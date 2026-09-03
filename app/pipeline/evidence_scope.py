"""Describe evidence coverage without guessing from question wording."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvidenceScopeView:
    """Machine-known boundaries for the text shown to a semantic judge."""

    scope: str
    has_more_before: bool | None
    has_more_after: bool | None
    view_complete: bool


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def evidence_scope_view(candidate: dict, *, max_candidate_chars: int) -> EvidenceScopeView:
    metadata = candidate.get("metadata") or {}
    text = str(candidate.get("text") or "")
    full_document = bool(metadata.get("full_context") or metadata.get("document_complete"))
    has_fragment_identity = bool(
        metadata.get("parent_id")
        or metadata.get("chunk_id")
        or candidate.get("chunk_id")
        or metadata.get("chunk_index") is not None
    )
    return EvidenceScopeView(
        scope=(
            "complete_document"
            if full_document
            else "document_fragment"
            if has_fragment_identity
            else "unknown_fragment"
        ),
        has_more_before=_optional_bool(metadata.get("has_more_before")),
        has_more_after=_optional_bool(metadata.get("has_more_after")),
        view_complete=len(text) <= max(1, int(max_candidate_chars)),
    )


def render_evidence_candidate(
    index: int,
    candidate: dict,
    *,
    max_candidate_chars: int,
) -> str:
    """Render one candidate with explicit, non-semantic coverage metadata.

    When the judge budget clips a long item, retain both ends.  A head-only
    view systematically hides conclusions, footnotes and final-state facts.
    """

    metadata = candidate.get("metadata") or {}
    scope = evidence_scope_view(candidate, max_candidate_chars=max_candidate_chars)
    body = str(candidate.get("text") or "")
    limit = max(1, int(max_candidate_chars))
    if len(body) > limit:
        marker = "\n…（本次判定展示已截断，中间内容未展示）…\n"
        available = max(2, limit - len(marker))
        head_chars = available // 2
        tail_chars = available - head_chars
        body = f"{body[:head_chars]}{marker}{body[-tail_chars:]}"

    heading = str(metadata.get("section_title") or metadata.get("heading_path") or "")

    def flag(value: bool | None) -> str:
        if value is None:
            return "unknown"
        return "yes" if value else "no"

    boundary = (
        f"scope={scope.scope}; "
        f"shown_text_complete={flag(scope.view_complete)}; "
        f"has_more_before={flag(scope.has_more_before)}; "
        f"has_more_after={flag(scope.has_more_after)}"
    )
    position = ""
    parent_index = metadata.get("parent_index")
    parent_count = metadata.get("parent_count")
    if isinstance(parent_index, int) and isinstance(parent_count, int) and parent_count > 0:
        position = f"; document_position={parent_index + 1}/{parent_count}"
    title_line = f"\nheading={heading}" if heading else ""
    return f"[{index}] {boundary}{position}{title_line}\n{body}".strip()
