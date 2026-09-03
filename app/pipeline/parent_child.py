"""Build parent evidence blocks and child retrieval blocks from parsed structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from app.chunkers.recursive import estimate_tokens
from app.parsers.base import StructuredBlock


@dataclass(slots=True)
class ParentChunk:
    parent_id: str
    parent_index: int
    text: str
    metadata: dict


@dataclass(slots=True)
class ChildChunk:
    text: str
    parent_id: str
    parent_index: int
    metadata: dict


def _block_metadata(block: StructuredBlock) -> dict:
    metadata = dict(block.metadata or {})
    if block.kind:
        metadata.setdefault("block_kind", block.kind)
    if block.section_title:
        metadata.setdefault("section_title", block.section_title)
    if block.heading_path:
        metadata.setdefault("heading_path", " > ".join(block.heading_path))
    if block.table_headers:
        metadata.setdefault("table_headers", " | ".join(block.table_headers))
    if block.source_anchor:
        metadata.setdefault("source_anchor", block.source_anchor)
    if block.page is not None:
        metadata.setdefault("page", block.page)
    return metadata


def _context_prefix(title: str, metadata: dict) -> str:
    parts = []
    if title:
        parts.append(title)
    heading_path = str(metadata.get("heading_path") or "")
    if heading_path and heading_path != title:
        parts.append(heading_path)
    # Keep retrieval prefixes bounded independently from the full parent text.
    return "\n".join(parts)[:96]


TokenCounter = Callable[[str], Awaitable[int]]


async def split_text_by_token_budget(
    text: str,
    *,
    prefix: str,
    target_tokens: int,
    hard_limit_tokens: int,
    continuity_tokens: int,
    token_counter: TokenCounter,
) -> list[str]:
    """Split text while accounting for the exact retrieval payload shape.

    ``token_counter`` measures the exact payload which will be sent to the
    embedding service.  It deliberately includes title/heading prefixes, so
    a 512-token server is never handed a 513-token retrieval payload.
    """
    target = max(1, min(int(target_tokens), int(hard_limit_tokens)))

    async def payload_tokens(body: str) -> int:
        payload = f"{prefix}\n{body}".strip() if prefix else body
        return await token_counter(payload)

    # Preserve natural boundaries first.  Keeping separators in the units
    # avoids silently dropping punctuation while merging neighbouring units.
    units: list[str] = []
    remaining = (text or "").strip()
    for separator in ("\n\n", "\n", "。", ".", " "):
        if not remaining:
            break
        if separator not in remaining:
            continue
        parts = remaining.split(separator)
        units = [part.strip() for part in parts if part.strip()]
        if len(units) > 1:
            break
    if not units:
        units = [remaining] if remaining else []

    async def split_oversized(unit: str) -> list[str]:
        result: list[str] = []
        pending = unit.strip()
        while pending:
            if await payload_tokens(pending) <= target:
                result.append(pending)
                break
            # Find the largest character prefix that satisfies the token
            # target. This is only the final fallback after natural splits.
            low, high, best = 1, len(pending), 0
            while low <= high:
                midpoint = (low + high) // 2
                candidate = pending[:midpoint].strip()
                if candidate and await payload_tokens(candidate) <= target:
                    best = midpoint
                    low = midpoint + 1
                else:
                    high = midpoint - 1
            if best <= 0:
                # A prefix alone can exceed a pathological hard limit. Keep
                # progress deterministic and let ingestion reject the payload
                # through the hard-limit assertion below.
                best = 1
            result.append(pending[:best].strip())
            pending = pending[best:].strip()
        return [item for item in result if item]

    normalized_units: list[str] = []
    for unit in units:
        normalized_units.extend(await split_oversized(unit))

    chunks: list[str] = []
    current = ""
    for unit in normalized_units:
        candidate = f"{current}\n{unit}".strip() if current else unit
        if current and await payload_tokens(candidate) > target:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)

    # Prefer sentence-level continuity over a fixed character overlap.  The
    # continuity prefix is only retained when it still satisfies the hard
    # model limit.
    if continuity_tokens > 0 and len(chunks) > 1:
        enriched = [chunks[0]]
        for index in range(1, len(chunks)):
            previous = chunks[index - 1]
            tail = previous
            while tail and (
                await token_counter(tail) > continuity_tokens
                or await payload_tokens(f"{tail}\n{chunks[index]}".strip()) > target
            ):
                cut = max(1, len(tail) // 8)
                tail = tail[cut:].strip()
            candidate = f"{tail}\n{chunks[index]}".strip() if tail else chunks[index]
            # The user target controls the actual embedding payload, not only
            # its body before continuity context is attached.  The hard model
            # limit remains a non-negotiable final assertion below.
            enriched.append(candidate if await payload_tokens(candidate) <= target else chunks[index])
        chunks = enriched

    for chunk in chunks:
        if await payload_tokens(chunk) > hard_limit_tokens:
            raise ValueError("child chunk exceeds embedding model token limit")
    return chunks


async def build_parent_child_chunks(
    doc_id: str,
    blocks: list[StructuredBlock],
    *,
    title: str = "",
    parent_max_tokens: int = 1024,
    child_max_tokens: int | None = None,
    child_overlap: int | None = None,
    child_target_tokens: int | None = None,
    child_hard_limit_tokens: int | None = None,
    child_continuity_tokens: int | None = None,
    token_counter: TokenCounter | None = None,
    index_profile_hash: str = "legacy",
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Create section-preserving parents and contextualized child retrieval text."""
    parents: list[ParentChunk] = []
    pending: list[StructuredBlock] = []
    pending_tokens = 0
    pending_path: tuple[str, ...] = ()

    def flush() -> None:
        nonlocal pending, pending_tokens, pending_path
        if not pending:
            return
        first = pending[0]
        evidence_block = next(
            (block for block in reversed(pending) if block.kind not in {"heading", "title"}),
            first,
        )
        text = "\n".join(block.text.strip() for block in pending if block.text.strip()).strip()
        if not text:
            pending, pending_tokens, pending_path = [], 0, ()
            return
        parent_index = len(parents)
        metadata = _block_metadata(evidence_block)
        metadata["parent_kind"] = evidence_block.kind
        # Parent IDs are physical evidence IDs.  Profile scoping lets a
        # candidate rebuild coexist with the active rendition of one source
        # document without overwriting parent evidence.
        parent_id = (
            f"{doc_id}:p{parent_index}"
            if index_profile_hash == "legacy"
            else f"{index_profile_hash}:{doc_id}:p{parent_index}"
        )
        parents.append(ParentChunk(parent_id, parent_index, text, metadata))
        pending, pending_tokens, pending_path = [], 0, ()

    for block in blocks:
        text = (block.text or "").strip()
        if not text:
            continue
        # Headings are context for following content, not independently retrieved facts.
        if block.kind in {"heading", "title"}:
            flush()
            pending = [block]
            pending_tokens = estimate_tokens(text)
            pending_path = block.heading_path
            continue

        block_tokens = estimate_tokens(text)
        block_path = block.heading_path
        isolated = block.kind in {"table", "code"}
        heading_context_for_isolated = (
            isolated
            and len(pending) == 1
            and pending[0].kind in {"heading", "title"}
            and pending_path == block_path
        )
        if pending and (
            (isolated and not heading_context_for_isolated)
            or (pending_path and block_path and pending_path != block_path)
            or pending_tokens + block_tokens > parent_max_tokens
        ):
            flush()
        if isolated:
            if heading_context_for_isolated:
                pending.append(block)
                pending_tokens += block_tokens
            else:
                pending = [block]
                pending_tokens = block_tokens
                pending_path = block_path
            flush()
            continue
        pending.append(block)
        pending_tokens += block_tokens
        pending_path = block_path or pending_path
    flush()

    target_tokens = int(child_target_tokens or child_max_tokens or 512)
    hard_limit_tokens = int(child_hard_limit_tokens or target_tokens)
    continuity_tokens = int(
        child_continuity_tokens
        if child_continuity_tokens is not None
        else (child_overlap or 0)
    )
    if token_counter is None:
        async def token_counter(payload: str) -> int:
            return estimate_tokens(payload)

    children: list[ChildChunk] = []
    for parent in parents:
        prefix = _context_prefix(title, parent.metadata)
        for child_text in await split_text_by_token_budget(
            parent.text,
            prefix=prefix,
            target_tokens=target_tokens,
            hard_limit_tokens=hard_limit_tokens,
            continuity_tokens=continuity_tokens,
            token_counter=token_counter,
        ):
            if not child_text:
                continue
            retrieval_text = f"{prefix}\n{child_text}".strip() if prefix else child_text
            children.append(
                ChildChunk(
                    text=retrieval_text,
                    parent_id=parent.parent_id,
                    parent_index=parent.parent_index,
                    metadata={
                        **parent.metadata,
                        "parent_id": parent.parent_id,
                        "parent_index": parent.parent_index,
                    },
                )
            )
    return parents, children
