"""Conversational chat orchestration."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from dataclasses import dataclass, field

from app.chunkers.recursive import estimate_tokens
from app.pipeline.answer_policy import (
    AnswerDecision,
    AnswerMode,
    AnswerPolicy,
    EvidenceOutcome,
    GroundingMode,
    resolve_answer_policy,
)
from app.pipeline.evidence_policy import (
    _ability_retrieval_expansion,
    _build_evidence_requirement,
    _definition_retrieval_expansion,
    _intent_retrieval_expansion,
    _negative_fact_retrieval_expansion,
)
from app.pipeline.evidence_controller import EvidenceControlDecision, control_evidence
from app.pipeline.evidence_resolution import resolve_retrieval_evidence
from app.pipeline.generation import resolve_generation_limits
from app.pipeline.query_rewriter import rewrite_query
from app.pipeline.retrieval import resolve_active_reranker, retrieve
from app.pipeline.retrieval_policy import RetrievalDecision, decide_retrieval, should_contextualize_with_history
from app.pipeline.retrieval_target import RetrievalTarget
from app.utils.runtime_errors import IndexUnavailableError
from app.pipeline.retrieval_trace import RetrievalTrace
from app.rag_scope import get_tenant_rag_collection_name, resolve_embedding_dimension
from app.settings.settings import AnswerQualityMode, AnswerQualityProfileConfig, AppConfig
from app.store.parent_chunk_store import ParentChunkStore
from app.utils.retrieval_match import normalize_exact_text, normalize_filename

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChatTurnResult:
    retrieval_query: str
    results: list[dict]
    sources: list[dict]
    decision: AnswerDecision
    reason: str
    fallback_used: bool = False
    grounding_mode: GroundingMode = "auto"
    answer_quality_mode: AnswerQualityMode = "normal"
    # ``results`` are the only citable, evidence-gated chunks.  When auto
    # routing finds material that is related but cannot prove the requested
    # fact, it is carried separately so generation can describe the boundary
    # without presenting it as a source or allowing citations.
    response_mode: AnswerMode = "auto"
    unverified_context: list[dict] = field(default_factory=list)
    route_plan: dict[str, str | bool] = field(default_factory=dict)
    evidence: dict[str, object] = field(default_factory=dict)
    evidence_guidance: dict[str, object] = field(default_factory=dict)
    answer_policy: dict[str, str | bool] = field(default_factory=dict)
    trace: dict | None = None


@dataclass(slots=True)
class RetrievalScopePlan:
    use_history_for_rewrite: bool
    scoped_doc_ids: list[str]
    allow_global_retry: bool = False
    scope_reason: str = ""


@dataclass(slots=True)
class QualityEvidenceResolution:
    results: list[dict]
    outcome: EvidenceOutcome
    decision: EvidenceControlDecision
    retrieval_query: str
    retry_count: int = 0
    context_expansion_count: int = 0
    actions: list[str] = field(default_factory=list)

    def to_guidance(self, **metadata: object) -> dict[str, object]:
        source_index_map = {
            original_index: final_index
            for final_index, original_index in enumerate(
                self.decision.selected_indices,
                start=1,
            )
        }
        return self.decision.to_guidance(
            source_index_map=source_index_map,
            **metadata,
        )


def _candidate_identity(candidate: dict) -> str:
    metadata = candidate.get("metadata") or {}
    return str(
        candidate.get("chunk_id")
        or metadata.get("chunk_id")
        or f"{metadata.get('doc_id', '')}:{metadata.get('chunk_index', '')}:{candidate.get('text', '')[:80]}"
    )


def _merge_quality_candidates(existing: list[dict], additions: list[dict]) -> list[dict]:
    """Merge correction rounds without allowing duplicates to crowd out evidence."""
    merged: dict[str, dict] = {}
    order: list[str] = []
    for candidate in [*existing, *additions]:
        key = _candidate_identity(candidate)
        previous = merged.get(key)
        if previous is None:
            merged[key] = candidate
            order.append(key)
            continue
        previous_score = float(previous.get("rank_score", previous.get("score", 0.0)) or 0.0)
        current_score = float(candidate.get("rank_score", candidate.get("score", 0.0)) or 0.0)
        if current_score > previous_score:
            merged[key] = candidate
    candidates = [merged[key] for key in order]
    return sorted(
        candidates,
        key=lambda item: (
            1 if (item.get("metadata") or {}).get("context_expansion") else 0,
            float(item.get("rank_score", item.get("score", 0.0)) or 0.0),
        ),
        reverse=True,
    )


def _has_expandable_context(candidates: list[dict]) -> bool:
    return any(
        (candidate.get("metadata") or {}).get("has_more_before") is True
        or (candidate.get("metadata") or {}).get("has_more_after") is True
        for candidate in candidates
    )


async def _resolve_quality_evidence(
    query: str,
    retrieval_query: str,
    candidates: list[dict],
    *,
    profile: AnswerQualityProfileConfig,
    llm_provider,
    retry_retrieval=None,
    expand_context=None,
    max_candidate_chars_override: int | None = None,
) -> QualityEvidenceResolution:
    """Run one semantic evidence controller with a bounded corrective loop."""
    aggregate = _merge_quality_candidates([], candidates)
    retry_count = 0
    context_expansion_count = 0
    actions: list[str] = []
    current_retrieval_query = retrieval_query

    while True:
        allow_retry = (
            retry_retrieval is not None
            and retry_count < profile.max_corrective_retrievals
        )
        controller_candidates = aggregate[:profile.evidence_max_candidates]
        allow_expand = (
            expand_context is not None
            and context_expansion_count < profile.max_context_expansions
            and _has_expandable_context(controller_candidates)
        )
        decision, judged_candidates = await control_evidence(
            query,
            current_retrieval_query,
            aggregate,
            llm_provider,
            allow_retry=allow_retry,
            allow_expand=allow_expand,
            timeout_seconds=profile.evidence_timeout_seconds,
            max_candidates=profile.evidence_max_candidates,
            max_tokens=profile.evidence_max_tokens,
            max_candidate_chars=(
                max_candidate_chars_override
                if max_candidate_chars_override is not None
                else profile.evidence_max_candidate_chars
            ),
            max_retries=profile.evidence_judge_max_retries,
        )
        actions.append(decision.action)

        if decision.action == "expand" and allow_expand:
            try:
                additions = await expand_context(
                    decision,
                    judged_candidates,
                    profile.context_expansion_radius,
                )
            except Exception as exc:
                logger.warning("相邻证据扩展失败: %s", exc)
                additions = []
            aggregate = _merge_quality_candidates(aggregate, additions)
            context_expansion_count += 1
            continue

        if decision.action == "retry" and allow_retry:
            next_query, new_candidates = await retry_retrieval(decision.retry_query)
            current_retrieval_query = next_query
            aggregate = _merge_quality_candidates(aggregate, new_candidates)
            retry_count += 1
            continue

        selected = [
            judged_candidates[index - 1]
            for index in decision.selected_indices
        ]
        candidate_count = len(aggregate)
        if decision.action == "answer":
            outcome = EvidenceOutcome.grounded(candidate_count, len(selected))
        elif decision.action == "boundary":
            outcome = EvidenceOutcome.boundary(
                decision.reason or "evidence_explicit_boundary",
                candidate_count,
                len(selected),
            )
        elif decision.action == "partial":
            outcome = EvidenceOutcome.partial(
                decision.reason or "evidence_partial",
                candidate_count,
                len(selected),
            )
        elif decision.action == "conflict":
            outcome = EvidenceOutcome.conflict(
                decision.reason or "evidence_conflict",
                candidate_count,
                len(selected),
            )
        elif decision.action == "unavailable":
            selected = []
            outcome = EvidenceOutcome.unavailable(decision.reason, candidate_count)
        else:
            selected = []
            outcome = EvidenceOutcome.missing(
                decision.reason or "evidence_insufficient",
                candidate_count,
            )
        return QualityEvidenceResolution(
            results=selected,
            outcome=outcome,
            decision=decision,
            retrieval_query=current_retrieval_query,
            retry_count=retry_count,
            context_expansion_count=context_expansion_count,
            actions=actions,
        )


def prepare_direct_chat_turn(
    query: str,
    *,
    grounding_mode: GroundingMode = "assistant",
    answer_quality_mode: AnswerQualityMode = "normal",
) -> ChatTurnResult:
    """Build a direct-answer turn without touching retrieval dependencies."""
    plan = RetrievalDecision("DIRECT", "assistant_mode_direct")
    evidence = EvidenceOutcome.not_checked()
    answer_policy = resolve_answer_policy(plan, evidence, grounding_mode=grounding_mode)
    return ChatTurnResult(
        retrieval_query=query,
        results=[],
        sources=[],
        decision=answer_policy.decision,
        reason=answer_policy.reason,
        grounding_mode=grounding_mode,
        answer_quality_mode=answer_quality_mode,
        response_mode=answer_policy.response_mode,
        route_plan=plan.to_dict(),
        evidence=evidence.to_dict(),
        answer_policy=answer_policy.to_dict(),
    )


_GLOBAL_RELATED_DOC_PATTERNS = (
    r"(知识库|全库|整个知识库).*(还有哪些|还有什么|更多|其他|别的).*(文档|资料)",
    r"(还有哪些|还有什么|更多|其他|别的).*(相关文档|相关资料)",
)
_CROSS_DOCUMENT_PATTERNS = (
    r"(和|与|跟).*(关系|关联|联系|区别|不同|对比|比较)",
    r"(关系|关联|联系|区别|不同|对比|比较).*(和|与|跟)",
)
_EXPLICIT_DOCUMENT_LOCAL_PATTERNS = (
    r"(文档|资料|原文).*(里|中|内)",
    r"(这篇|这份|该).*(文档|资料|原文)",
    r"(根据|结合|对照).*(文档|资料|原文)",
    r"上传(的)?文档",
)

async def _ensure_bm25_ready(
    settings: AppConfig,
    bm25_store,
    vector_store,
    document_store,
    embedding_provider,
    *,
    tenant_slug: str | None = None,
    embedding_model: str | None = None,
    collection_name: str | None = None,
):
    if not settings.retrieval.hybrid.enabled or not bm25_store:
        return None

    embedding_dimension = await resolve_embedding_dimension(embedding_provider)
    collection_name = collection_name or get_tenant_rag_collection_name(
        settings.vectorstore.collection_prefix,
        embedding_model or settings.embedding.openai.model_name,
        tenant_slug=tenant_slug,
        embedding_dimension=embedding_dimension,
    )
    if settings.observability.log_bm25_rebuild_reason and not getattr(bm25_store, "is_ready", False):
        logger.info(
            "BM25 尚未就绪，准备初始化: collection=%s tenant_slug=%s",
            collection_name,
            tenant_slug or "",
        )
    try:
        ready_kwargs = {}
        ensure_ready_params = inspect.signature(bm25_store.ensure_ready).parameters
        if "document_store" in ensure_ready_params:
            ready_kwargs["document_store"] = document_store
        await bm25_store.ensure_ready(vector_store, collection_name, **ready_kwargs)
    except Exception as exc:
        logger.warning("BM25 初始化失败，本次降级为向量检索: %s", exc)
        return None
    return bm25_store


async def _has_ready_knowledge(
    document_store,
    vector_store,
    *,
    tenant_id: str | None = None,
    collection_name: str | None = None,
    allowed_doc_ids: list[str] | None = None,
    allowed_kb_ids: list[str] | None = None,
) -> bool:
    if allowed_doc_ids is not None:
        return bool(allowed_doc_ids)
    if allowed_kb_ids:
        list_ready_doc_ids = getattr(document_store, "list_ready_doc_ids", None)
        if callable(list_ready_doc_ids):
            for kb_id in allowed_kb_ids:
                if await list_ready_doc_ids(kb_id, tenant_id=tenant_id):
                    return True
            return False
    has_ready_documents = getattr(document_store, "has_ready_documents", None)
    if callable(has_ready_documents):
        return await has_ready_documents(tenant_id=tenant_id)
    return await vector_store.count(collection_name=collection_name) > 0


def latest_grounded_sources(history: list[dict]) -> list[dict]:
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        if item.get("status") != "completed":
            continue
        sources = item.get("sources") or []
        if sources:
            return sources
    return []


async def _format_sources(
    results: list[dict],
    document_store,
    *,
    tenant_id: str | None = None,
) -> list[dict]:
    doc_cache = await _load_documents_by_ids(
        document_store,
        [result.get("metadata", {}).get("doc_id", "") for result in results],
        tenant_id=tenant_id,
    )
    sources = []
    for index, result in enumerate(results, 1):
        metadata = result.get("metadata") or {}
        doc_id = metadata.get("doc_id", "")
        doc = doc_cache.get(doc_id, {})
        score = _result_score(result)
        kb_id = metadata.get("kb_id") or doc.get("kb_id") or ""
        sources.append({
            "index": index,
            "doc_id": doc_id,
            "kb_id": kb_id,
            "filename": doc.get("filename", ""),
            "chunk_id": result.get("chunk_id") or metadata.get("chunk_id", ""),
            "chunk_index": metadata.get("chunk_index"),
            "score": score,
            "score_source": result.get("score_source") or "unknown",
            "text": result.get("text", ""),
            "metadata": metadata,
        })
    return sources


async def _hydrate_parent_results(results: list[dict], metadata_db: str) -> list[dict]:
    """Replace child retrieval text with its parent evidence text when available."""
    parent_ids = [str((result.get("metadata") or {}).get("parent_id") or "") for result in results]
    if not any(parent_ids):
        return results

    parents = await ParentChunkStore(metadata_db).get_many(parent_ids)
    hydrated: list[dict] = []
    seen_parent_ids: set[str] = set()
    for result in results:
        metadata = result.get("metadata") or {}
        parent_id = str(metadata.get("parent_id") or "")
        parent = parents.get(parent_id)
        if not parent:
            hydrated.append(result)
            continue
        # Multiple child hits from one parent should consume one source slot.
        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        parent_index = int(parent.get("parent_index") or 0)
        parent_count = max(1, int(parent.get("parent_count") or 1))
        hydrated.append({
            **result,
            "text": parent["text"],
            # Parent text is the citable evidence unit.  Keep the matched
            # child ID only as provenance so neighbour expansion deduplicates
            # the same parent reliably across rounds.
            "chunk_id": parent_id,
            "metadata": {
                **metadata,
                **(parent.get("metadata") or {}),
                "parent_id": parent_id,
                "parent_index": parent_index,
                "parent_count": parent_count,
                "document_complete": parent_count == 1,
                "has_more_before": parent_index > 0,
                "has_more_after": parent_index + 1 < parent_count,
                "matched_child_id": result.get("chunk_id") or metadata.get("chunk_id") or "",
                "matched_child_text": metadata.get("child_text") or result.get("text", ""),
            },
        })
    return hydrated


async def _expand_parent_context(
    decision: EvidenceControlDecision,
    judged_candidates: list[dict],
    metadata_db: str,
    radius: int,
) -> list[dict]:
    """Load model-requested neighbouring parent evidence in document order."""

    requested: list[tuple[str, dict]] = []
    for index in decision.expand_indices:
        if not 1 <= index <= len(judged_candidates):
            continue
        candidate = judged_candidates[index - 1]
        parent_id = str((candidate.get("metadata") or {}).get("parent_id") or "")
        if parent_id:
            requested.append((parent_id, candidate))
    if not requested:
        return []

    store = ParentChunkStore(metadata_db)
    anchors = await store.get_many(parent_id for parent_id, _candidate in requested)
    documents: dict[tuple[str, str], list[dict]] = {}
    for anchor in anchors.values():
        key = (str(anchor.get("doc_id") or ""), str(anchor.get("profile_hash") or "legacy"))
        if key not in documents and key[0]:
            documents[key] = await store.list_by_document(key[0], profile_hash=key[1])

    expanded: list[dict] = []
    seen_parent_ids = set(anchors)
    distance = max(1, int(radius))
    for parent_id, candidate in requested:
        anchor = anchors.get(parent_id)
        if not anchor:
            continue
        rows = documents.get(
            (str(anchor.get("doc_id") or ""), str(anchor.get("profile_hash") or "legacy")),
            [],
        )
        anchor_index = int(anchor.get("parent_index") or 0)
        wanted: set[int] = set()
        if decision.expand_direction in {"before", "both"}:
            wanted.update(range(max(0, anchor_index - distance), anchor_index))
        if decision.expand_direction in {"after", "both"}:
            wanted.update(range(anchor_index + 1, min(len(rows), anchor_index + distance + 1)))
        candidate_metadata = candidate.get("metadata") or {}
        candidate_score = float(candidate.get("rank_score", candidate.get("score", 0.0)) or 0.0)
        for row in rows:
            parent_index = int(row.get("parent_index") or 0)
            row_parent_id = str(row.get("parent_id") or "")
            if parent_index not in wanted or not row_parent_id or row_parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(row_parent_id)
            parent_count = len(rows)
            expanded.append({
                "text": str(row.get("text") or ""),
                "metadata": {
                    **candidate_metadata,
                    **(row.get("metadata") or {}),
                    "doc_id": row.get("doc_id") or candidate_metadata.get("doc_id") or "",
                    "parent_id": row_parent_id,
                    "parent_index": parent_index,
                    "parent_count": parent_count,
                    "chunk_id": row_parent_id,
                    "chunk_index": parent_index,
                    "document_complete": parent_count == 1,
                    "has_more_before": parent_index > 0,
                    "has_more_after": parent_index + 1 < parent_count,
                    "context_expansion": True,
                    "context_expansion_direction": decision.expand_direction,
                },
                "chunk_id": row_parent_id,
                "score": candidate_score,
                "retrieval_score": candidate_score,
                "rank_score": candidate_score,
                "score_source": "context_expansion",
            })
    return expanded


def _result_score(result: dict) -> float:
    for key in ("rerank_score", "retrieval_score", "score", "fusion_score", "rrf_score", "bm25_score", "vector_score"):
        value = result.get(key)
        if value is not None:
            return float(value)
    return 0.0


async def _load_documents_by_ids(
    document_store,
    doc_ids: list[str],
    *,
    tenant_id: str | None = None,
) -> dict[str, dict]:
    unique_ids = [doc_id for doc_id in dict.fromkeys(doc_ids) if doc_id]
    if not unique_ids:
        return {}

    loader = getattr(document_store, "list_by_doc_ids", None)
    if callable(loader):
        docs = await loader(unique_ids, tenant_id=tenant_id)
        return {doc["doc_id"]: doc for doc in docs}

    doc_cache: dict[str, dict] = {}
    for doc_id in unique_ids:
        doc_cache[doc_id] = await document_store.get(doc_id, tenant_id=tenant_id) or {}
    return doc_cache


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


async def _load_grounded_turn_contexts(
    history: list[dict],
    document_store,
    *,
    tenant_id: str | None = None,
) -> list[list[dict]]:
    grounded_turns: list[list[dict]] = []
    all_doc_ids: list[str] = []

    for item in history:
        if item.get("role") != "assistant" or item.get("status") != "completed":
            continue
        sources = item.get("sources") or []
        if not sources:
            continue
        grounded_turns.append(sources)
        all_doc_ids.extend(source.get("doc_id") or "" for source in sources)

    if not grounded_turns:
        return []

    doc_cache = await _load_documents_by_ids(document_store, all_doc_ids, tenant_id=tenant_id)
    ready_turns: list[list[dict]] = []
    for sources in grounded_turns:
        ready_sources: list[dict] = []
        for source in sources:
            doc_id = source.get("doc_id") or ""
            doc = doc_cache.get(doc_id)
            if not doc_id or not doc or doc.get("status") != "ready":
                continue
            ready_sources.append({
                "doc_id": doc_id,
                "doc": doc,
                "source": source,
            })
        if ready_sources:
            ready_turns.append(ready_sources)
    return ready_turns


def _doc_variants(doc: dict, source: dict) -> list[str]:
    variants: list[str] = []
    filename = str(doc.get("filename") or source.get("filename") or "").strip()
    doc_id = str(doc.get("doc_id") or source.get("doc_id") or "").strip()
    if doc_id:
        variants.append(doc_id)
    if filename:
        variants.append(filename)
        stem, _extension = normalize_filename(filename)
        if len(stem) >= 3:
            variants.append(stem)
    return [variant for variant in dict.fromkeys(variants) if variant]


def _query_mentions_variant(query: str, variant: str) -> bool:
    if not variant:
        return False
    normalized_query = normalize_exact_text(query)
    normalized_variant = normalize_exact_text(variant)
    if len(normalized_variant) < 3:
        return False
    return normalized_variant in normalized_query or variant.lower() in query.lower()


def _looks_like_global_related_docs_query(query: str) -> bool:
    return _matches_any(query, _GLOBAL_RELATED_DOC_PATTERNS)


def _looks_like_cross_document_query(query: str) -> bool:
    return _matches_any(query, _CROSS_DOCUMENT_PATTERNS)


def _looks_like_explicit_document_local(query: str) -> bool:
    return _matches_any(query, _EXPLICIT_DOCUMENT_LOCAL_PATTERNS)


async def _resolve_retrieval_scope_plan(
    query: str,
    history: list[dict],
    document_store,
    *,
    tenant_id: str | None = None,
) -> RetrievalScopePlan:
    use_history_for_rewrite = should_contextualize_with_history(query, history)
    grounded_turns = await _load_grounded_turn_contexts(history, document_store, tenant_id=tenant_id)
    if not grounded_turns:
        return RetrievalScopePlan(use_history_for_rewrite=use_history_for_rewrite, scoped_doc_ids=[])

    for turn in reversed(grounded_turns):
        matched_doc_ids = [
            ctx["doc_id"]
            for ctx in turn
            if any(_query_mentions_variant(query, variant) for variant in _doc_variants(ctx["doc"], ctx["source"]))
        ]
        if matched_doc_ids:
            return RetrievalScopePlan(
                use_history_for_rewrite=use_history_for_rewrite,
                scoped_doc_ids=list(dict.fromkeys(matched_doc_ids)),
                scope_reason="explicit_history_doc_match",
            )

    latest_doc_ids = list(dict.fromkeys(ctx["doc_id"] for ctx in grounded_turns[-1]))
    if use_history_for_rewrite and _looks_like_explicit_document_local(query):
        return RetrievalScopePlan(
            use_history_for_rewrite=True,
            scoped_doc_ids=latest_doc_ids,
            scope_reason="explicit_document_local",
        )

    if _looks_like_global_related_docs_query(query) or _looks_like_cross_document_query(query):
        return RetrievalScopePlan(
            use_history_for_rewrite=use_history_for_rewrite,
            scoped_doc_ids=[],
            scope_reason="global_or_cross_document_query",
        )

    if use_history_for_rewrite and latest_doc_ids:
        return RetrievalScopePlan(
            use_history_for_rewrite=True,
            scoped_doc_ids=latest_doc_ids,
            allow_global_retry=True,
            scope_reason="latest_grounded_followup",
        )

    return RetrievalScopePlan(use_history_for_rewrite=False, scoped_doc_ids=[])


def _apply_relevance_threshold(results: list[dict], threshold: float) -> list[dict]:
    if not results:
        return []
    # Reranker scores are calibrated relevance probabilities.  Previously the
    # threshold was used only to decide whether the *best* result existed,
    # then every near-zero candidate was still placed in the LLM context.
    # That lets a very large unrelated document dilute an otherwise correct
    # answer.  Keep every candidate only in the legacy path; when a reranker
    # actually ran, apply its configured floor to each candidate.
    if any(result.get("rerank_score") is not None for result in results):
        return [result for result in results if _result_score(result) >= threshold]
    best_score = max(_result_score(result) for result in results)
    if best_score < threshold:
        return []
    return results


def _effective_relevance_threshold(results: list[dict], settings: AppConfig) -> float:
    """Use calibrated reranker scores only when this request actually reranked."""
    if any(result.get("rerank_score") is not None for result in results):
        return float(settings.retrieval.reranker.min_score)
    return float(settings.retrieval.relevance_threshold)
async def _standalone_query(
    query: str,
    history: list[dict],
    llm_provider,
    *,
    history_limit: int,
    truncate: int,
    max_tokens: int = 256,
) -> str:
    if not history:
        return query

    history_lines = []
    for item in history[-history_limit:]:
        role = "用户" if item["role"] == "user" else "助手"
        content = (item.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content[:truncate]}")

    if not history_lines:
        return query

    messages = [
        {
            "role": "system",
            "content": (
                "将多轮对话中的当前问题改写成一个适合文档检索的独立问题。"
                "如果当前问题本身已经是完整的新问题，就保持原意，不要被无关历史带偏。"
                "只输出改写后的问题。"
            ),
        },
        {
            "role": "user",
            "content": "历史对话:\n"
            + "\n".join(history_lines)
            + f"\n\n当前问题: {query}",
        },
    ]
    try:
        rewritten = await llm_provider.chat(
            messages,
            max_tokens=max_tokens,
            temperature=0,
            thinking_mode="off",
        )
    except Exception as exc:
        logger.warning("历史问题改写失败，回退到原始问题: %s", exc)
        return query
    return rewritten.strip() or query


async def _retrieve_results(
    query: str,
    history: list[dict],
    settings: AppConfig,
    llm_provider,
    embedding_provider,
    vector_store,
    document_store,
    bm25_store,
    reranker=None,
    *,
    tenant_id: str | None = None,
    tenant_slug: str | None = None,
    trace: RetrievalTrace | None = None,
    allowed_doc_ids: list[str] | None = None,
    allowed_kb_ids: list[str] | None = None,
    retrieval_targets: list[RetrievalTarget] | None = None,
    semantic_only: bool = False,
) -> tuple[str, list[dict], list[dict]]:
    normalized_query = query.strip()
    if semantic_only:
        # Quality-mode retrieval keeps the user-selected document/KB boundary
        # but does not let question-shape regexes narrow or widen that scope.
        # The model receives history when it must make the question standalone.
        scope_plan = RetrievalScopePlan(
            use_history_for_rewrite=bool(history),
            scoped_doc_ids=[],
            allow_global_retry=False,
            scope_reason="semantic_scope",
        )
    else:
        scope_plan = await _resolve_retrieval_scope_plan(
            query,
            history,
            document_store,
            tenant_id=tenant_id,
        )
    use_history_for_rewrite = scope_plan.use_history_for_rewrite
    skip_standalone = False
    if (
        history
        and settings.retrieval.skip_standalone_rewrite_for_short_query
        and not use_history_for_rewrite
    ):
        if len(normalized_query) < settings.retrieval.standalone_rewrite_min_length:
            skip_standalone = True
            if settings.observability.log_query_rewrite_skips:
                logger.info(
                    "跳过 standalone rewrite: query_len=%d min_length=%d query=%r",
                    len(normalized_query),
                    settings.retrieval.standalone_rewrite_min_length,
                    normalized_query,
                )

    if skip_standalone or not use_history_for_rewrite:
        retrieval_query = normalized_query or query
    else:
        retrieval_query = await _standalone_query(
            query,
            history,
            llm_provider,
            history_limit=settings.chat.history_limit,
            truncate=settings.chat.rewrite_context_truncate,
            max_tokens=settings.chat.history_rewrite_max_tokens,
        )
    queries = [retrieval_query]
    overview_request = (
        False
        if semantic_only
        else _build_evidence_requirement(retrieval_query).kind == "collection_overview"
    )
    ranking_query = retrieval_query
    expansion_config = settings.retrieval.definition_query_expansion
    if not semantic_only and expansion_config.enabled and expansion_config.max_added_queries > 0:
        definition_expansion = _definition_retrieval_expansion(retrieval_query)
        if definition_expansion and definition_expansion not in queries:
            queries.append(definition_expansion)
            # The deterministic expansion makes the reranker compare the
            # definition intent against the same candidate pool.
            ranking_query = definition_expansion
        ability_expansion = _ability_retrieval_expansion(retrieval_query)
        if ability_expansion and ability_expansion not in queries:
            queries.append(ability_expansion)
            ranking_query = ability_expansion
    intent_config = settings.retrieval.intent_query_expansion
    if not semantic_only and intent_config.enabled and intent_config.max_added_queries > 0:
        intent_expansion = _intent_retrieval_expansion(
            retrieval_query,
            aliases=intent_config.aliases,
        )
        if intent_expansion and intent_expansion not in queries:
            queries.append(intent_expansion)
            ranking_query = intent_expansion
    if not semantic_only:
        negative_expansion = _negative_fact_retrieval_expansion(retrieval_query)
        if negative_expansion and negative_expansion not in queries:
            queries.append(negative_expansion)
            ranking_query = negative_expansion
    if settings.retrieval.query_rewrite.enabled:
        rewritten_queries = await rewrite_query(
            retrieval_query,
            llm_provider,
            strategy=settings.retrieval.query_rewrite.strategy,
            max_rewrites=settings.retrieval.query_rewrite.max_rewrites,
            timeout_seconds=settings.retrieval.query_rewrite.timeout_seconds,
            expand_max_tokens=getattr(settings.retrieval.query_rewrite, "expand_max_tokens", 200),
            decompose_max_tokens=getattr(settings.retrieval.query_rewrite, "decompose_max_tokens", 300),
        )
        queries = list(dict.fromkeys([*queries, *rewritten_queries]))
    max_effective_queries = max(1, settings.retrieval.max_effective_queries)
    if len(queries) > max_effective_queries:
        if settings.observability.log_query_rewrite_skips:
            logger.info(
                "限制检索 query 数量: original=%d limited=%d queries=%s",
                len(queries),
                max_effective_queries,
                queries,
            )
        queries = queries[:max_effective_queries]
    # A capped plan must never tell the reranker to use an expansion that was
    # dropped from the actual retrieval work.
    if ranking_query not in queries:
        ranking_query = retrieval_query if retrieval_query in queries else queries[0]

    embedding_dimension = await resolve_embedding_dimension(embedding_provider)
    legacy_collection_name = get_tenant_rag_collection_name(
        settings.vectorstore.collection_prefix,
        settings.embedding.openai.model_name,
        tenant_slug=tenant_slug,
        embedding_dimension=embedding_dimension,
    )
    targets = retrieval_targets or [
        RetrievalTarget(
            collection_name=legacy_collection_name,
            embedding_provider=embedding_provider,
            kb_ids=tuple(allowed_kb_ids) if allowed_kb_ids else None,
        )
    ]
    # Avoid an accidental duplicate collection from a caller composing
    # fallback targets.  The profile is part of the key because a malformed
    # configuration must not silently collapse incompatible providers.
    unique_targets: dict[tuple[str, str, tuple[str, ...] | None], RetrievalTarget] = {}
    for target in targets:
        key = (target.collection_name, target.profile_hash, target.kb_ids)
        unique_targets[key] = target
    targets = list(unique_targets.values())

    query_vectors_by_provider: dict[int, dict[str, list[float] | None]] = {}
    bm25_by_collection: dict[str, object | None] = {}

    async def _query_vectors_for(target: RetrievalTarget) -> dict[str, list[float] | None]:
        provider_key = id(target.embedding_provider)
        cached = query_vectors_by_provider.get(provider_key)
        if cached is not None:
            return cached
        vectors: dict[str, list[float] | None] = {}
        for planned_query in queries:
            try:
                vectors[planned_query] = await target.embedding_provider.embed_query(planned_query)
            except Exception as exc:
                vectors[planned_query] = None
                logger.warning(
                    "预计算检索 query embedding 失败: collection=%s query=%r error=%s",
                    target.collection_name, planned_query, exc,
                )
        query_vectors_by_provider[provider_key] = vectors
        return vectors

    async def _bm25_for(target: RetrievalTarget):
        if target.collection_name not in bm25_by_collection:
            bm25_by_collection[target.collection_name] = await _ensure_bm25_ready(
                settings,
                bm25_store,
                vector_store,
                document_store,
                target.embedding_provider,
                tenant_slug=tenant_slug,
                collection_name=target.collection_name,
            )
        return bm25_by_collection[target.collection_name]

    async def _run_retrieve(
        target: RetrievalTarget,
        scoped_doc_ids: list[str] | None,
        *,
        scope_type: str,
        kb_ids: list[str] | None = None,
        retrieval_config=None,
        branch: bool = False,
        apply_relevance_threshold: bool = True,
    ) -> list[dict]:
        target_kb_ids = list(target.kb_ids) if target.kb_ids else None
        # ``knowledge_scope=all`` is represented by an empty list at the API
        # boundary.  Empty does not mean "deny every KB": for an immutable
        # profile target it must retain that target's own KB scope.  Treating
        # it as an explicit filter made every all-KB chat request return
        # NO_EVIDENCE before it ever queried the active collection.
        requested_kb_ids = kb_ids if kb_ids else (allowed_kb_ids or None)
        if target_kb_ids is not None and requested_kb_ids is not None:
            effective_kb_ids = [kb_id for kb_id in requested_kb_ids if kb_id in set(target_kb_ids)]
        else:
            effective_kb_ids = target_kb_ids or requested_kb_ids
        if target_kb_ids is not None and not effective_kb_ids:
            return []
        attempt = trace.begin_attempt(retrieval_query, scope_type=scope_type) if trace is not None and not branch else None
        if attempt is not None:
            attempt.record_stage(
                "query_plan",
                candidate_count=0,
                details={
                    "query_count": len(queries),
                    "scoped": bool(scoped_doc_ids),
                    "definition_expansion": ranking_query != retrieval_query,
                },
            )
        retrieved = await retrieve(
            ranking_query,
            target.embedding_provider,
            vector_store,
            retrieval_config or settings.retrieval,
            document_store,
            await _bm25_for(target),
            queries=queries,
            collection_name=target.collection_name,
            tenant_id=tenant_id,
            allowed_doc_ids=scoped_doc_ids,
            allowed_kb_ids=effective_kb_ids,
            query_vectors=await _query_vectors_for(target),
            log_vector_query_timing=settings.observability.log_vector_query_timing,
            log_bm25_timing=settings.observability.log_bm25_timing,
            reranker=None if branch else reranker,
            trace_attempt=attempt,
            original_query=retrieval_query,
        )
        # In a multi-KB candidate branch the candidate set must stay broad
        # until the single global rerank is complete.  Applying the final
        # relevance gate here would discard a valid candidate before the
        # reranker can compare it with candidates from the other KBs.
        threshold = _effective_relevance_threshold(retrieved, settings)
        if semantic_only or overview_request or not apply_relevance_threshold:
            accepted = retrieved
        else:
            accepted = _apply_relevance_threshold(retrieved, threshold)
        if attempt is not None:
            attempt.record_stage(
                "relevance_threshold",
                candidates=accepted,
                reason=("accepted" if accepted else "rejected") if apply_relevance_threshold else "deferred",
                details={"threshold": threshold, "deferred": not apply_relevance_threshold},
            )
            attempt.finish(
                ("accepted" if accepted else "relevance_threshold_rejected")
                if apply_relevance_threshold
                else "candidate_pool"
            )
        return accepted

    allowed_set = set(allowed_doc_ids or []) if allowed_doc_ids is not None else None
    history_scope = scope_plan.scoped_doc_ids
    if allowed_kb_ids:
        # A KB filter is the hard boundary.  Do not let a document-local
        # history hint from a previously selected KB narrow this request to
        # the wrong document set.
        scoped_doc_ids = list(allowed_doc_ids or []) if allowed_set is not None else None
        initial_scope_type = "knowledge_bases"
    elif allowed_set is not None:
        narrowed_history_scope = [doc_id for doc_id in history_scope if doc_id in allowed_set]
        scoped_doc_ids = narrowed_history_scope or list(allowed_doc_ids or [])
        initial_scope_type = (
            "knowledge_base_history" if narrowed_history_scope else "knowledge_base"
        )
    else:
        scoped_doc_ids = history_scope or None
        initial_scope_type = scope_plan.scope_reason or ("scoped" if history_scope else "global")
    multi_kb_config = settings.retrieval.multi_knowledge_base
    selected_kb_ids = list(dict.fromkeys(kb_id for kb_id in (allowed_kb_ids or []) if kb_id))
    should_parallel_candidates = (
        len(selected_kb_ids) > 1
        and len(selected_kb_ids) <= multi_kb_config.max_parallel_knowledge_bases
        and multi_kb_config.strategy in {"parallel_candidates", "adaptive"}
    )
    needs_global_merge = len(targets) > 1 or should_parallel_candidates
    if needs_global_merge:
        branch_config = settings.retrieval.model_copy(deep=True)
        branch_config.top_k = max(
            settings.retrieval.top_k,
            min(
                settings.retrieval.reranker.candidate_pool_size,
                settings.retrieval.top_k * settings.retrieval.candidate_multiplier,
            ),
        )
        branch_specs: list[tuple[RetrievalTarget, list[str] | None]] = []
        if should_parallel_candidates:
            for target in targets:
                target_kb_ids = list(target.kb_ids or selected_kb_ids)
                for kb_id in target_kb_ids:
                    if kb_id in selected_kb_ids:
                        branch_specs.append((target.with_kb_ids([kb_id]), [kb_id]))
        else:
            branch_specs = [(target, list(target.kb_ids) if target.kb_ids else allowed_kb_ids) for target in targets]
        branch_semaphore = asyncio.Semaphore(
            max(1, multi_kb_config.max_parallel_knowledge_bases)
        )

        async def _bounded_branch(target: RetrievalTarget, kb_ids: list[str] | None) -> list[dict]:
            async with branch_semaphore:
                return await asyncio.wait_for(
                    _run_retrieve(
                        target,
                        scoped_doc_ids,
                        scope_type="knowledge_base_parallel" if selected_kb_ids else "profile_targets",
                        kb_ids=kb_ids,
                        retrieval_config=branch_config,
                        branch=True,
                        apply_relevance_threshold=False,
                    ),
                    timeout=multi_kb_config.branch_timeout_seconds,
                )

        branch_tasks = [
            _bounded_branch(target, kb_ids)
            for target, kb_ids in branch_specs
        ]
        branch_outcomes = await asyncio.gather(*branch_tasks, return_exceptions=True)
        merged_candidates: dict[str, dict] = {}
        for (target, kb_ids), outcome in zip(branch_specs, branch_outcomes):
            if isinstance(outcome, Exception):
                if isinstance(outcome, IndexUnavailableError):
                    # A target selected by the immutable index resolver is a
                    # required source of truth, not an optional retrieval
                    # branch. Never convert its disappearance into a partial
                    # answer from a different KB or a direct fallback.
                    raise outcome
                logger.warning(
                    "知识库候选召回失败: collection=%s kb_ids=%s error=%s",
                    target.collection_name, kb_ids, outcome,
                )
                continue
            for candidate in outcome:
                metadata = candidate.get("metadata") or {}
                key = candidate.get("chunk_id") or metadata.get("chunk_id") or (
                    f"{target.profile_hash}:{metadata.get('doc_id', '')}:{metadata.get('chunk_index', '')}"
                )
                previous = merged_candidates.get(key)
                if previous is None or float(candidate.get("rank_score", 0.0)) > float(previous.get("rank_score", 0.0)):
                    merged_candidates[key] = candidate
        results = sorted(merged_candidates.values(), key=lambda item: item.get("rank_score", 0.0), reverse=True)
        rerank_pool_size = max(settings.retrieval.top_k, settings.retrieval.reranker.candidate_pool_size)
        rerank_candidates = results[:rerank_pool_size]
        active_reranker = resolve_active_reranker(reranker)
        if active_reranker is not None and rerank_candidates:
            try:
                rerank_scores = await active_reranker.rerank(
                    ranking_query,
                    [candidate.get("text", "") for candidate in rerank_candidates],
                )
                if len(rerank_scores) != len(rerank_candidates):
                    raise ValueError("reranker returned an unexpected number of scores")
                for candidate, score in zip(rerank_candidates, rerank_scores):
                    candidate["rerank_score"] = float(score)
                    candidate["rank_score"] = float(score)
                results = sorted(rerank_candidates, key=lambda item: item.get("rank_score", float("-inf")), reverse=True)
            except Exception as exc:
                logger.warning("多知识库全局 reranker 失败，保留融合排序: %s", exc)
                results = rerank_candidates
        else:
            results = rerank_candidates
        final_threshold = _effective_relevance_threshold(results, settings)
        if not semantic_only and not overview_request:
            results = _apply_relevance_threshold(results, final_threshold)
        results = results[:settings.retrieval.top_k]
        if trace is not None:
            attempt = trace.begin_attempt(retrieval_query, scope_type="knowledge_bases_parallel")
            attempt.record_stage(
                "multi_kb_candidates",
                candidates=results,
                details={
                    "knowledge_base_count": len(selected_kb_ids) or len(targets),
                    "branch_candidate_limit": branch_config.top_k,
                    "relevance_threshold": final_threshold,
                },
            )
            attempt.finish("accepted" if results else "no_candidates")
    else:
        results = await _run_retrieve(
            targets[0],
            scoped_doc_ids,
            scope_type=initial_scope_type,
            apply_relevance_threshold=not semantic_only,
        )
    if not results and scope_plan.allow_global_retry and history_scope:
        logger.info(
            "局部检索未命中，回退全库重试: reason=%s query=%r scoped_doc_ids=%s",
            scope_plan.scope_reason,
            retrieval_query,
            scope_plan.scoped_doc_ids,
        )
        retry_scope = list(allowed_doc_ids or []) if allowed_set is not None else None
        if len(targets) == 1:
            results = await _run_retrieve(
                targets[0], retry_scope,
                scope_type="knowledge_base_retry" if allowed_set is not None else "global_retry",
            )
        else:
            retry_candidates = await asyncio.gather(*[
                _run_retrieve(
                    target, retry_scope,
                    scope_type="knowledge_base_retry" if allowed_set is not None else "global_retry",
                    branch=True, apply_relevance_threshold=False,
                )
                for target in targets
            ])
            results = _apply_relevance_threshold(
                sorted(
                    [item for candidates in retry_candidates for item in candidates],
                    key=lambda item: item.get("rank_score", 0.0), reverse=True,
                )[:settings.retrieval.top_k],
                _effective_relevance_threshold(
                    [item for candidates in retry_candidates for item in candidates], settings,
                ),
            )

    results = await _hydrate_parent_results(results, settings.storage.metadata_db)
    if trace is not None and trace.attempts:
        trace.attempts[-1].record_stage("parent_hydration", candidates=results)

    sources = await _format_sources(results, document_store, tenant_id=tenant_id)
    for result, source in zip(results, sources):
        result["source_index"] = source["index"]
    return retrieval_query, results, sources


async def _reuse_results(
    history: list[dict],
    document_store,
    *,
    tenant_id: str | None = None,
    allowed_doc_ids: list[str] | None = None,
    allowed_kb_ids: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    sources = latest_grounded_sources(history)
    if not sources:
        return [], []

    results: list[dict] = []
    reusable_sources: list[dict] = []
    doc_cache = await _load_documents_by_ids(
        document_store,
        [source.get("doc_id") or "" for source in sources],
        tenant_id=tenant_id,
    )

    for source in sources:
        doc_id = source.get("doc_id") or ""
        if allowed_doc_ids is not None and doc_id not in set(allowed_doc_ids):
            return [], []
        doc = doc_cache.get(doc_id)
        if doc_id and (not doc or doc.get("status") != "ready"):
            return [], []
        if allowed_kb_ids and (not doc or doc.get("kb_id") not in set(allowed_kb_ids)):
            return [], []

        metadata = source.get("metadata") or {}
        results.append({
            "text": source.get("text", ""),
            "metadata": metadata,
            "score": 1.0,
            "retrieval_score": 1.0,
            "rank_score": 1.0,
            "score_source": "reuse",
            "chunk_id": source.get("chunk_id") or metadata.get("chunk_id", ""),
            "source_index": source.get("index"),
        })
        reusable_sources.append(source)

    return results, reusable_sources


def _log_decision(
    settings: AppConfig,
    route_plan: RetrievalDecision,
    execution_plan: RetrievalDecision,
    answer_policy: AnswerPolicy,
    evidence: EvidenceOutcome,
    query: str,
    retrieval_query: str,
    sources: list[dict],
    results: list[dict],
    elapsed_ms: int,
) -> None:
    if not settings.retrieval.decision.log_decisions:
        return
    logger.info(
        "chat route=%s route_intent=%s route_origin=%s route_confidence=%s route_policy=%s final=%s reason=%s evidence=%s evidence_reason=%s llm_gate=%s fallback=%s query=%r retrieval_query=%r reusable_sources=%d results=%d elapsed_ms=%d",
        route_plan.decision,
        route_plan.intent,
        route_plan.origin,
        route_plan.confidence,
        route_plan.evidence_policy,
        answer_policy.decision,
        answer_policy.reason,
        evidence.status,
        evidence.reason,
        execution_plan.used_llm_gate,
        execution_plan.fallback_used,
        query,
        retrieval_query,
        len(sources),
        len(results),
        elapsed_ms,
    )


def _trace_payload(
    trace: RetrievalTrace | None,
    route_plan: RetrievalDecision,
    execution_plan: RetrievalDecision,
    evidence: EvidenceOutcome,
    answer_policy: AnswerPolicy | None = None,
) -> dict | None:
    """Attach bounded route/evidence state to an optional retrieval trace."""
    if trace is None:
        return None
    payload = trace.to_dict()
    payload["route_plan"] = route_plan.to_dict()
    payload["execution_plan"] = execution_plan.to_dict()
    payload["evidence"] = evidence.to_dict()
    if answer_policy is not None:
        payload["answer_policy"] = answer_policy.to_dict()
    return payload


async def prepare_chat_turn(
    query: str,
    history: list[dict],
    settings: AppConfig,
    llm_provider,
    embedding_provider,
    vector_store,
    document_store,
    bm25_store=None,
    reranker=None,
    tenant_id: str | None = None,
    tenant_slug: str | None = None,
    grounding_mode: GroundingMode = "auto",
    answer_quality_mode: AnswerQualityMode | None = None,
    diagnostics: bool = False,
    allowed_doc_ids: list[str] | None = None,
    allowed_kb_ids: list[str] | None = None,
    retrieval_targets: list[RetrievalTarget] | None = None,
) -> ChatTurnResult:
    started = time.perf_counter()
    effective_quality_mode = answer_quality_mode or settings.chat.answer_quality.default_mode
    semantic_quality = answer_quality_mode is not None
    quality_profile = settings.chat.answer_quality.profile(effective_quality_mode)
    # The chat route resolves active targets from local index metadata before
    # calling us.  An explicit empty target list means there is no searchable
    # corpus, so never probe Embedding merely to construct a collection name.
    # That probe can itself retry for minutes and used to make an empty library
    # look like a stuck generation.
    if retrieval_targets is not None:
        has_vector_data = bool(retrieval_targets)
    else:
        embedding_dimension = await resolve_embedding_dimension(embedding_provider)
        collection_name = get_tenant_rag_collection_name(
            settings.vectorstore.collection_prefix,
            settings.embedding.openai.model_name,
            tenant_slug=tenant_slug,
            embedding_dimension=embedding_dimension,
        )
        has_vector_data = await _has_ready_knowledge(
            document_store,
            vector_store,
            tenant_id=tenant_id,
            collection_name=collection_name,
            allowed_doc_ids=allowed_doc_ids,
            allowed_kb_ids=allowed_kb_ids,
        )
    reusable_sources = latest_grounded_sources(history) if settings.retrieval.decision.reuse_last_sources else []
    if allowed_doc_ids is not None:
        allowed_set = set(allowed_doc_ids)
        reusable_sources = [source for source in reusable_sources if source.get("doc_id") in allowed_set]
    elif allowed_kb_ids:
        allowed_kb_set = set(allowed_kb_ids)
        reusable_sources = [
            source for source in reusable_sources
            if (source.get("kb_id") or (source.get("metadata") or {}).get("kb_id")) in allowed_kb_set
        ]

    if not has_vector_data and not reusable_sources:
        # No route can retrieve evidence when the selected corpus is empty.
        # Skip the optional LLM gate as well: it cannot change the eventual
        # evidence boundary and would add one more slow model request before
        # the actual direct/no-evidence answer.
        decision = RetrievalDecision(
            "RETRIEVE",
            "knowledge_index_empty",
            intent="knowledge",
            evidence_policy="required",
            origin="availability",
        )
    else:
        decision = await decide_retrieval(
            query,
            history,
            bool(reusable_sources),
            settings.retrieval.decision,
            llm_provider,
            has_vector_data=has_vector_data,
            semantic_only=semantic_quality,
        )
    route_plan = decision
    evidence = EvidenceOutcome.not_checked()
    evidence_guidance: dict[str, object] = {}
    unverified_context: list[dict] = []

    if decision.decision == "RETRIEVE" and not has_vector_data:
        # No index is available.  Keep the route as RETRIEVE and represent the
        # unavailable evidence independently; this also avoids touching the
        # embedding/vector providers for an explicitly empty target set.
        evidence = EvidenceOutcome.missing(decision.reason)

    # 知识库模式必须先检索和校验证据。路由器即使将问题误判成实时查询，
    # 也不能绕过本地知识库直接生成通用答案；仍保留 REUSE，它只会复用
    # 已落库的带来源回答，复用失败会重新检索。
    if grounding_mode == "knowledge" and decision.decision in {"DIRECT", "LIVE_UNSUPPORTED"}:
        decision = decision.transition(
            "RETRIEVE",
            "knowledge_mode_forced_retrieve",
            intent="knowledge",
            evidence_policy="required",
            origin="guardrail",
        )

    retrieval_query = query
    results: list[dict] = []
    sources: list[dict] = []
    trace = RetrievalTrace() if diagnostics or settings.observability.log_retrieval_trace else None

    if decision.decision == "REUSE":
        results, _reused_sources = await _reuse_results(
            history,
            document_store,
            tenant_id=tenant_id,
            allowed_doc_ids=allowed_doc_ids,
            allowed_kb_ids=allowed_kb_ids,
        )
        if not results:
            decision = decision.transition(
                "RETRIEVE",
                "reuse_invalid_retrieve",
                intent="knowledge",
                evidence_policy="required",
                origin="guardrail",
            )
        elif not semantic_quality:
            evidence = EvidenceOutcome.grounded(len(results), len(results))

    if (
        decision.decision == "RETRIEVE"
        and not has_vector_data
        and evidence.status == "not_checked"
    ):
        evidence = EvidenceOutcome.missing("knowledge_index_empty")

    if decision.decision == "RETRIEVE" and has_vector_data:
        retrieval_query, results, _retrieval_sources = await _retrieve_results(
            query,
            history,
            settings,
            llm_provider,
            embedding_provider,
            vector_store,
            document_store,
            bm25_store,
            reranker,
            tenant_id=tenant_id,
            tenant_slug=tenant_slug,
            trace=trace,
            allowed_doc_ids=allowed_doc_ids,
            allowed_kb_ids=allowed_kb_ids,
            # 对话入口必须和 Agent 检索、诊断接口使用同一组活动索引。
            # 否则知识库已切换到候选代际后，聊天仍会误查旧 collection。
            retrieval_targets=retrieval_targets,
            semantic_only=semantic_quality,
        )

    if (
        not semantic_quality
        and evidence.status == "not_checked"
        and decision.decision == "RETRIEVE"
    ):
        legacy_resolution = await resolve_retrieval_evidence(
            query,
            retrieval_query,
            results,
            grounding_mode=grounding_mode,
            support_grader_config=settings.retrieval.support_grader,
            llm_provider=llm_provider,
        )
        results = legacy_resolution.results
        evidence = legacy_resolution.outcome
        unverified_context = legacy_resolution.unverified_context
        if trace is not None and trace.attempts:
            trace.attempts[-1].record_stage(
                "evidence_gate",
                candidate_count=len(results),
                reason=evidence.reason or "not_applicable",
                details=legacy_resolution.trace_details(),
            )

    if (
        semantic_quality
        and evidence.status == "not_checked"
        and decision.decision in {"RETRIEVE", "REUSE"}
    ):
        async def retry_retrieval(retry_query: str) -> tuple[str, list[dict]]:
            next_query, next_results, _next_sources = await _retrieve_results(
                retry_query,
                [],
                settings,
                llm_provider,
                embedding_provider,
                vector_store,
                document_store,
                bm25_store,
                reranker,
                tenant_id=tenant_id,
                tenant_slug=tenant_slug,
                trace=trace,
                allowed_doc_ids=allowed_doc_ids,
                allowed_kb_ids=allowed_kb_ids,
                retrieval_targets=retrieval_targets,
                semantic_only=True,
            )
            return next_query, next_results

        async def expand_context(
            expand_decision: EvidenceControlDecision,
            judged_candidates: list[dict],
            radius: int,
        ) -> list[dict]:
            return await _expand_parent_context(
                expand_decision,
                judged_candidates,
                settings.storage.metadata_db,
                radius,
            )

        quality_resolution = await _resolve_quality_evidence(
            query,
            retrieval_query,
            results,
            profile=quality_profile,
            llm_provider=llm_provider,
            retry_retrieval=retry_retrieval if has_vector_data else None,
            expand_context=expand_context,
        )
        results = quality_resolution.results
        evidence = quality_resolution.outcome
        evidence_guidance = quality_resolution.to_guidance(
            context_expansions=quality_resolution.context_expansion_count,
        )
        retrieval_query = quality_resolution.retrieval_query
        if quality_resolution.retry_count or quality_resolution.context_expansion_count:
            decision = decision.transition(
                "RETRIEVE",
                (
                    "semantic_corrective_retrieval"
                    if quality_resolution.retry_count
                    else "semantic_context_expansion"
                ),
                intent="knowledge",
                evidence_policy="required",
                origin="llm",
            )
        if trace is not None and trace.attempts:
            trace.attempts[-1].record_stage(
                "semantic_evidence_controller",
                candidate_count=evidence.candidate_count,
                reason=evidence.reason or "not_applicable",
                details={
                    "quality_mode": effective_quality_mode,
                    "controller_action": quality_resolution.decision.action,
                    "corrective_retrievals": quality_resolution.retry_count,
                    "context_expansions": quality_resolution.context_expansion_count,
                    "controller_actions": ",".join(quality_resolution.actions),
                    "accepted_count": evidence.accepted_count,
                },
            )

    if results:
        sources = await _format_sources(results, document_store, tenant_id=tenant_id)
        for result, source in zip(results, sources):
            result["source_index"] = source["index"]
    elif decision.decision == "DIRECT" or not results:
        sources = []
        if decision.decision == "DIRECT":
            results = []

    answer_policy = resolve_answer_policy(
        decision,
        evidence,
        grounding_mode=grounding_mode,
    )
    response_mode = answer_policy.response_mode

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    trace_payload = None
    if trace is not None:
        trace.finish(answer_policy.decision)
        trace_payload = _trace_payload(trace, route_plan, decision, evidence, answer_policy)
    _log_decision(
        settings,
        route_plan,
        decision,
        answer_policy,
        evidence,
        query,
        retrieval_query,
        sources,
        results,
        elapsed_ms,
    )
    return ChatTurnResult(
        retrieval_query=retrieval_query,
        results=results,
        sources=sources,
        decision=answer_policy.decision,
        reason=answer_policy.reason,
        fallback_used=decision.fallback_used,
        grounding_mode=grounding_mode,
        answer_quality_mode=effective_quality_mode,
        response_mode=response_mode,
        unverified_context=unverified_context,
        route_plan=route_plan.to_dict(),
        evidence=evidence.to_dict(),
        evidence_guidance=evidence_guidance,
        answer_policy=answer_policy.to_dict(),
        trace=trace_payload,
    )


async def prepare_full_context_turn(
    query: str,
    settings: AppConfig,
    document_store,
    *,
    doc_id: str,
    tenant_id: str | None = None,
    profile_hash: str = "legacy",
    llm_provider=None,
    answer_quality_mode: AnswerQualityMode = "normal",
) -> ChatTurnResult:
    """Use one short document as a single, untruncated RAG source.

    The persisted parent chunks are used rather than re-parsing uploads, so the
    text is exactly the indexed document version and works after restarts.
    """
    document = await document_store.get(doc_id, tenant_id=tenant_id)
    if not document or document.get("status") != "ready":
        raise ValueError("全文模式的文档不存在或尚未准备完成")
    text, parents, _tokens, _source_budget = await validate_full_context_document(
        settings, doc_id, profile_hash=profile_hash,
    )
    metadata = dict((parents[0].get("metadata") or {}) if parents else {})
    metadata.update({
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}:full",
        "chunk_index": 0,
        "full_context": True,
        "document_complete": True,
        "has_more_before": False,
        "has_more_after": False,
    })
    result = {
        "text": text,
        "metadata": metadata,
        "chunk_id": metadata["chunk_id"],
        "score": 1.0,
        "retrieval_score": 1.0,
        "rank_score": 1.0,
        "score_source": "full_context",
    }
    route_plan = RetrievalDecision("RETRIEVE", "full_context_document", origin="configuration")
    quality_profile = settings.chat.answer_quality.profile(answer_quality_mode)
    resolution = await _resolve_quality_evidence(
        query,
        query,
        [result],
        profile=quality_profile,
        llm_provider=llm_provider,
        max_candidate_chars_override=len(text),
    )
    results = resolution.results
    evidence = resolution.outcome
    sources = await _format_sources(results, document_store, tenant_id=tenant_id)
    for accepted, source in zip(results, sources):
        accepted["source_index"] = source["index"]
    answer_policy = resolve_answer_policy(route_plan, evidence, grounding_mode="knowledge")
    return ChatTurnResult(
        retrieval_query=query,
        results=results,
        sources=sources,
        decision=answer_policy.decision,
        reason=answer_policy.reason,
        grounding_mode="knowledge",
        answer_quality_mode=answer_quality_mode,
        response_mode=answer_policy.response_mode,
        route_plan=route_plan.to_dict(),
        evidence=evidence.to_dict(),
        evidence_guidance=resolution.to_guidance(),
        answer_policy=answer_policy.to_dict(),
    )


async def validate_full_context_document(
    settings: AppConfig,
    doc_id: str,
    *,
    profile_hash: str = "legacy",
) -> tuple[str, list[dict], int, int]:
    """Return a document only when its complete indexed text fits the model budget."""
    parents = await ParentChunkStore(settings.storage.metadata_db).list_by_document(
        doc_id, profile_hash=profile_hash,
    )
    text = "\n\n".join(str(parent.get("text") or "").strip() for parent in parents).strip()
    if not text:
        raise ValueError("该文档没有可用的全文索引，请重新摄入后再使用全文模式")
    effective_context, output_budget, history_budget = resolve_generation_limits(
        settings.llm.context_window,
        settings.llm.max_tokens,
    )
    source_budget = max(0, effective_context - output_budget - history_budget - 600)
    tokens = estimate_tokens(text)
    if tokens > source_budget:
        raise ValueError(
            f"该文档约 {tokens} tokens，超过当前模型可用于全文的 {source_budget} tokens；请改用智能检索"
        )
    return text, parents, tokens, source_budget


async def prepare_retrieval_only(
    query: str,
    history: list[dict],
    settings: AppConfig,
    llm_provider,
    embedding_provider,
    vector_store,
    document_store,
    bm25_store=None,
    reranker=None,
    tenant_id: str | None = None,
    tenant_slug: str | None = None,
    diagnostics: bool = False,
    allowed_kb_ids: list[str] | None = None,
    retrieval_targets: list[RetrievalTarget] | None = None,
    answer_quality_mode: AnswerQualityMode | None = None,
) -> ChatTurnResult:
    effective_quality_mode = answer_quality_mode or settings.chat.answer_quality.default_mode
    semantic_quality = answer_quality_mode is not None
    trace = RetrievalTrace() if diagnostics or settings.observability.log_retrieval_trace else None
    route_plan = RetrievalDecision("RETRIEVE", "forced_retrieve", origin="configuration")
    retrieval_query, candidates, _sources = await _retrieve_results(
        query,
        history,
        settings,
        llm_provider,
        embedding_provider,
        vector_store,
        document_store,
        bm25_store,
        reranker,
        tenant_id=tenant_id,
        tenant_slug=tenant_slug,
        trace=trace,
        allowed_kb_ids=allowed_kb_ids,
        retrieval_targets=retrieval_targets,
        semantic_only=semantic_quality,
    )

    async def retry_retrieval(retry_query: str) -> tuple[str, list[dict]]:
        next_query, next_candidates, _next_sources = await _retrieve_results(
            retry_query,
            [],
            settings,
            llm_provider,
            embedding_provider,
            vector_store,
            document_store,
            bm25_store,
            reranker,
            tenant_id=tenant_id,
            tenant_slug=tenant_slug,
            trace=trace,
            allowed_kb_ids=allowed_kb_ids,
            retrieval_targets=retrieval_targets,
            semantic_only=True,
        )
        return next_query, next_candidates

    async def expand_context(
        expand_decision: EvidenceControlDecision,
        judged_candidates: list[dict],
        radius: int,
    ) -> list[dict]:
        return await _expand_parent_context(
            expand_decision,
            judged_candidates,
            settings.storage.metadata_db,
            radius,
        )

    if semantic_quality:
        resolution = await _resolve_quality_evidence(
            query,
            retrieval_query,
            candidates,
            profile=settings.chat.answer_quality.profile(effective_quality_mode),
            llm_provider=llm_provider,
            retry_retrieval=retry_retrieval,
            expand_context=expand_context,
        )
    else:
        resolution = await resolve_retrieval_evidence(
            query,
            retrieval_query,
            candidates,
            grounding_mode="knowledge",
            support_grader_config=settings.retrieval.support_grader,
            llm_provider=llm_provider,
            retain_supplementary_evidence=True,
        )
    if semantic_quality:
        retrieval_query = resolution.retrieval_query
    results = resolution.results
    evidence = resolution.outcome
    sources = await _format_sources(results, document_store, tenant_id=tenant_id)
    for result, source in zip(results, sources):
        result["source_index"] = source["index"]
    if semantic_quality and trace is not None and trace.attempts:
        trace.attempts[-1].record_stage(
            "semantic_evidence_controller",
            candidate_count=evidence.candidate_count,
            reason=evidence.reason or "not_applicable",
            details={
                "quality_mode": effective_quality_mode,
                "controller_action": resolution.decision.action,
                "corrective_retrievals": resolution.retry_count,
                "context_expansions": resolution.context_expansion_count,
                "accepted_count": evidence.accepted_count,
            },
        )
    elif trace is not None and trace.attempts:
        trace.attempts[-1].record_stage(
            "evidence_gate",
            candidate_count=len(results),
            reason=evidence.reason or "not_applicable",
            details=resolution.trace_details(),
        )
    answer_policy = resolve_answer_policy(
        route_plan,
        evidence,
        grounding_mode="knowledge",
    )
    if trace is not None:
        trace.finish(answer_policy.decision)
    return ChatTurnResult(
        retrieval_query=retrieval_query,
        results=results,
        sources=sources,
        decision=answer_policy.decision,
        reason=answer_policy.reason,
        grounding_mode="knowledge",
        answer_quality_mode=effective_quality_mode,
        response_mode=answer_policy.response_mode,
        route_plan=route_plan.to_dict(),
        evidence=evidence.to_dict(),
        evidence_guidance=(
            resolution.to_guidance(
                context_expansions=resolution.context_expansion_count,
            )
            if semantic_quality
            else {}
        ),
        answer_policy=answer_policy.to_dict(),
        trace=_trace_payload(trace, route_plan, route_plan, evidence, answer_policy),
    )
