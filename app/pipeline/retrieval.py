"""检索管道"""

from __future__ import annotations

import inspect
import logging
import re
import time
from typing import Dict, List, Optional

from app.providers.embedding.base import BaseEmbeddingProvider
from app.settings.settings import RetrievalConfig
from app.store.bm25_store import BM25Store
from app.store.document_store import DocumentStore
from app.store.vector_store import VectorStore
from app.utils.runtime_errors import IndexUnavailableError
from app.pipeline.retrieval_trace import AttemptTrace
from app.utils.text_utils import STOP_WORDS, tokenize_mixed

logger = logging.getLogger(__name__)
_ENUMERATIVE_QUERY_RE = re.compile(r"(哪些|列出|罗列|支持哪些|适配|有哪些|分别|清单|列表|表格|字段|包含)")
_PLATFORM_HINT_RE = re.compile(r"(tiktok|instagram|reddit|twitter/x|twitter|x)", re.IGNORECASE)
_RELATION_QUERY_RE = re.compile(
    r"((和|与|跟|及|以及).*(关系|关联|联系|区别|不同|差异|对比|比较))|((关系|关联|联系|区别|不同|差异|对比|比较).*(和|与|跟|及|以及))"
)
_GENERIC_QUERY_TERMS = {
    "什么",
    "哪些",
    "如何",
    "怎么",
    "为何",
    "为什么",
    "是谁",
    "是什么",
    "是啥",
    "什么意思",
    "一个",
    "一款",
    "一种",
    "东西",
    "玩意",
    "技能",
    "功能",
    "情况",
    "内容",
    "信息",
    "介绍",
}
_RELATION_STOP_TERMS = {
    "关系",
    "关联",
    "联系",
    "区别",
    "不同",
    "差异",
    "对比",
    "比较",
}
_ASCII_ANCHOR_RE = re.compile(r"^(?:[a-z]\d{1,3}|[a-z][a-z0-9_.-]{2,31})$", re.IGNORECASE)
_DEFINITION_INTENT_QUERY_RE = re.compile(r"产品定位|目标用户|核心能力")
_SHORT_IDENTIFIER_QUERY_RE = re.compile(r"^[A-Za-z]\d{1,3}$")
_HARD_QUERY_PHRASES = (
    "目标用户",
    "产品定位",
    "核心用户",
    "核心能力",
    "字段契约",
    "规则资产",
    "批量私信",
    "自动建联",
    "不包含",
    "不做",
)


def _get_chunk_key(result: dict) -> str:
    """统一 chunk 标识，优先使用显式 chunk_id。"""
    if result.get("chunk_id"):
        return result["chunk_id"]

    metadata = result.get("metadata", {})
    if metadata.get("chunk_id"):
        return metadata["chunk_id"]

    doc_id = metadata.get("doc_id", "")
    chunk_index = metadata.get("chunk_index", "")
    if doc_id and chunk_index != "":
        return f"{doc_id}_{chunk_index}"

    return result.get("text", "")[:50]


def _merge_chunk_payload(existing: dict | None, incoming: dict) -> dict:
    if not existing:
        return incoming.copy()

    merged = existing.copy()
    merged.update(incoming)

    existing_meta = existing.get("metadata", {})
    incoming_meta = incoming.get("metadata", {})
    if existing_meta or incoming_meta:
        merged["metadata"] = {**existing_meta, **incoming_meta}

    if existing.get("text"):
        merged["text"] = existing["text"]

    return merged


def _result_score(result: dict, *keys: str) -> float:
    for key in keys:
        value = result.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _annotate_lexical_scores(results: List[dict]) -> List[dict]:
    if not results:
        return results

    bm25_scores = [float(result.get("bm25_score", 0.0)) for result in results]
    max_bm25 = max(bm25_scores)
    min_bm25 = min(bm25_scores)
    if max_bm25 <= 0:
        for result in results:
            result["bm25_score_normalized"] = 0.0
            result["lexical_score"] = min(1.0, float(result.get("exact_match_bonus", 0.0)))
        return results

    if max_bm25 == min_bm25:
        for result in results:
            normalized = 1.0
            lexical_score = min(1.0, normalized + float(result.get("exact_match_bonus", 0.0)))
            result["bm25_score_normalized"] = normalized
            result["lexical_score"] = lexical_score
        return results

    range_bm25 = max_bm25 - min_bm25
    for result in results:
        normalized = (float(result.get("bm25_score", 0.0)) - min_bm25) / range_bm25
        lexical_score = min(1.0, normalized + float(result.get("exact_match_bonus", 0.0)))
        result["bm25_score_normalized"] = normalized
        result["lexical_score"] = lexical_score
    return results


def _finalize_retrieval_scores(
    results: List[dict],
    hybrid_config,
) -> List[dict]:
    finalized = []
    weight_sum = max(hybrid_config.vector_weight + hybrid_config.bm25_weight, 1e-10)
    for result in results:
        vector_score = result.get("vector_score")
        lexical_score = result.get("lexical_score")
        if vector_score is not None and lexical_score is not None:
            retrieval_score = (
                float(vector_score) * hybrid_config.vector_weight
                + float(lexical_score) * hybrid_config.bm25_weight
            ) / weight_sum
            score_source = "hybrid"
        elif vector_score is not None:
            retrieval_score = float(vector_score)
            score_source = "vector"
        elif lexical_score is not None:
            retrieval_score = float(lexical_score)
            score_source = "bm25"
        else:
            retrieval_score = _result_score(result, "fusion_score", "rrf_score", "score", "bm25_score")
            score_source = "unknown"

        rank_score = _result_score(
            result,
            "fusion_score",
            "rrf_score",
            "bm25_rank_score",
            "vector_score",
            "bm25_score",
            "score",
        )
        result["retrieval_score"] = retrieval_score
        result["rank_score"] = rank_score
        result["score"] = retrieval_score
        result["score_source"] = score_source
        finalized.append(result)
    finalized.sort(key=lambda item: item.get("rank_score", 0), reverse=True)
    return finalized


def _apply_structured_rerank(results: List[dict], query: str) -> List[dict]:
    if not results:
        return results

    normalized_query = (query or "").strip().lower()
    is_enumerative = bool(_ENUMERATIVE_QUERY_RE.search(normalized_query))
    query_terms = {term for term in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]+", normalized_query) if len(term) > 1}
    boosted_results: List[dict] = []

    for result in results:
        metadata = result.get("metadata") or {}
        kind = str(metadata.get("block_kind") or metadata.get("kind") or "").lower()
        section_title = str(metadata.get("section_title") or "")
        heading_path = str(metadata.get("heading_path") or "")
        table_headers = str(metadata.get("table_headers") or "")
        text = str(result.get("text") or "")

        boost = 0.0
        if is_enumerative:
            if kind in {"table", "list_item"}:
                boost += 0.14
            elif kind in {"heading", "title"}:
                boost += 0.08
            elif kind == "paragraph" and len(text) > 280:
                boost -= 0.04

            structural_markers = sum(marker in text for marker in ("|", "、", "；", "\n- ", "\n* "))
            if structural_markers >= 2:
                boost += 0.05

            platform_hits = len(_PLATFORM_HINT_RE.findall(text))
            if platform_hits >= 2:
                boost += 0.18
                if kind == "list_item":
                    boost += 0.08
                if platform_hits >= 4:
                    boost += 0.12
            if platform_hits == 0 and any(marker in text for marker in ("distribution_channels", "channel_drafts[]")):
                boost -= 0.08

            for field_text in (section_title, heading_path, table_headers):
                if not field_text:
                    continue
                normalized_field = field_text.lower()
                if any(term in normalized_field for term in query_terms):
                    boost += 0.05
                if any(token in normalized_field for token in ("渠道", "适配", "平台")):
                    boost += 0.08
                if "渠道适配" in normalized_field or "分发渠道" in normalized_field:
                    boost += 0.08

        if boost:
            result["structured_boost"] = boost
            base_retrieval = float(result.get("retrieval_score", result.get("score", 0.0)))
            result["retrieval_score"] = min(1.0, base_retrieval + boost)
            result["score"] = result["retrieval_score"]
            result["rank_score"] = float(result.get("rank_score", 0.0)) + boost
        boosted_results.append(result)

    boosted_results.sort(key=lambda item: item.get("rank_score", 0), reverse=True)
    return boosted_results


def _query_terms(query: str) -> List[str]:
    terms: List[str] = []
    seen: set[str] = set()
    for token in tokenize_mixed(query or ""):
        normalized = token.strip().lower()
        if (
            not normalized
            or len(normalized) <= 1
            or normalized in seen
            or normalized in STOP_WORDS
            or normalized in _GENERIC_QUERY_TERMS
        ):
            continue
        terms.append(normalized)
        seen.add(normalized)
    return terms


def _is_relation_query(query: str) -> bool:
    return bool(_RELATION_QUERY_RE.search((query or "").strip().lower()))


def _relation_anchor_terms(query: str) -> List[str]:
    if not _is_relation_query(query):
        return []

    anchors: List[str] = []
    seen: set[str] = set()
    for term in _query_terms(query):
        if term in _RELATION_STOP_TERMS or term in seen:
            continue
        if not _is_anchor_term(term):
            continue
        anchors.append(term)
        seen.add(term)
        if len(anchors) >= 3:
            break
    return anchors


def _augment_relation_queries(
    query: str,
    queries: List[str],
    retrieval_config: RetrievalConfig,
) -> tuple[List[str], dict[str, str]]:
    relation_config = retrieval_config.relation_query
    normalized_queries: List[str] = []
    seen: set[str] = set()
    for candidate in queries:
        normalized = (candidate or "").strip()
        if not normalized or normalized in seen:
            continue
        normalized_queries.append(normalized)
        seen.add(normalized)

    if not relation_config.enabled:
        return normalized_queries, {}

    anchors = _relation_anchor_terms(query)
    if len(anchors) < 2:
        return normalized_queries, {}

    max_queries = max(1, retrieval_config.max_effective_queries)
    support_budget = max(0, relation_config.support_query_limit)
    support_query_terms: dict[str, str] = {}
    for anchor in anchors[1:]:
        if support_budget <= 0 or len(normalized_queries) >= max_queries:
            break
        support_query = f"{anchor} 是什么"
        if support_query in seen:
            continue
        normalized_queries.append(support_query)
        seen.add(support_query)
        support_query_terms[support_query] = anchor
        support_budget -= 1

    return normalized_queries[:max_queries], support_query_terms


def _is_anchor_term(term: str) -> bool:
    if not term:
        return False
    if _ASCII_ANCHOR_RE.match(term):
        return True
    return bool(re.fullmatch(r"[\u4e00-\u9fff]{2,8}", term))


def _result_search_space(result: dict) -> str:
    metadata = result.get("metadata") or {}
    haystacks = [
        str(result.get("text") or ""),
        str(metadata.get("filename") or ""),
        str(metadata.get("doc_id") or ""),
        str(metadata.get("section_title") or ""),
        str(metadata.get("heading_path") or ""),
        str(metadata.get("table_headers") or ""),
        str(metadata.get("source_anchor") or ""),
    ]
    return "\n".join(haystacks).lower()


def _contains_exact_query_term(query: str, value: str) -> bool:
    normalized_query = str(query or "").lower()
    normalized_value = str(value or "").lower()
    if not normalized_query or not normalized_value:
        return False
    if re.search(r"[\u4e00-\u9fff]", normalized_value):
        chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", normalized_value)
        return any(term in normalized_query for term in chinese_terms)
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_value)}(?![a-z0-9])", normalized_query))


def _has_hard_exact_query_match(result: dict, query: str) -> bool:
    """Keep exact section/identifier hits from being erased by a reranker."""
    reasons = result.get("exact_match_reasons") or []
    queries = [query, *(result.get("retrieval_queries") or [])]
    search_space = _result_search_space(result)
    for current_query in queries:
        if any(phrase in current_query.lower() and phrase in search_space for phrase in _HARD_QUERY_PHRASES):
            return True
    for reason in reasons:
        field, separator, value = str(reason).partition(":")
        if not separator or field not in {"identifier", "section_title", "heading_path", "table_headers", "source_anchor"}:
            continue
        value = value.strip()
        if field == "identifier" and not _SHORT_IDENTIFIER_QUERY_RE.fullmatch(value):
            continue
        if any(_contains_exact_query_term(current_query, value) for current_query in queries):
            return True
    return False


def _promote_hard_exact_candidates(
    reranked: list[dict],
    legacy_ranked: list[dict],
    query: str,
    top_k: int,
) -> list[dict]:
    original_query, *_expanded_queries = str(query or "").splitlines()
    hard_candidates = [
        candidate
        for candidate in legacy_ranked
        if _has_hard_exact_query_match(candidate, query)
    ]

    def _strength(candidate: dict) -> tuple[float, float, float]:
        text = str(candidate.get("text") or "").lower()
        primary_hits = sum(
            1
            for phrase in _HARD_QUERY_PHRASES
            if phrase in original_query.lower() and phrase in text
        )
        expanded_hits = sum(
            1
            for phrase in _HARD_QUERY_PHRASES
            if phrase in query.lower() and phrase in text
        )
        return (
            float(primary_hits),
            float(expanded_hits),
            float(candidate.get("exact_match_bonus", 0.0))
            + float(candidate.get("rank_score", 0.0)),
        )

    # A long document often contains many generic sections mentioning an
    # intent term such as “目标用户”.  Promote only the strongest exact
    # evidence instead of flooding the top-k with those generic sections.
    hard_candidates.sort(key=_strength, reverse=True)
    promoted = hard_candidates[: max(1, min(top_k, 2))]
    selected = {_get_chunk_key(item) for item in promoted}
    promoted.extend(item for item in reranked if _get_chunk_key(item) not in selected)
    return promoted


def _normalize_doc_id_scope(doc_ids: Optional[List[str]]) -> List[str]:
    return [doc_id for doc_id in dict.fromkeys(doc_ids or []) if doc_id]


def _normalize_kb_id_scope(kb_ids: Optional[List[str]]) -> List[str]:
    return [kb_id for kb_id in dict.fromkeys(kb_ids or []) if kb_id]


def _build_vector_where_for_doc_scope(
    doc_ids: Optional[List[str]],
    kb_ids: Optional[List[str]] = None,
) -> dict | None:
    normalized = _normalize_doc_id_scope(doc_ids)
    normalized_kb_ids = _normalize_kb_id_scope(kb_ids)
    clauses: list[dict] = []
    if normalized:
        clauses.append({"doc_id": normalized[0]} if len(normalized) == 1 else {"doc_id": {"$in": normalized}})
    if normalized_kb_ids:
        clauses.append(
            {"kb_id": normalized_kb_ids[0]}
            if len(normalized_kb_ids) == 1
            else {"kb_id": {"$in": normalized_kb_ids}}
        )
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _combine_vector_where(base_where: dict | None, extra_where: dict) -> dict:
    if not base_where:
        return extra_where
    return {"$and": [base_where, extra_where]}


def _filter_results_by_doc_scope(results: List[dict], doc_ids: Optional[List[str]]) -> List[dict]:
    allowed = set(_normalize_doc_id_scope(doc_ids))
    if not allowed:
        return results
    return [
        result
        for result in results
        if (result.get("metadata") or {}).get("doc_id") in allowed
    ]


def _filter_results_by_kb_scope(results: List[dict], kb_ids: Optional[List[str]]) -> List[dict]:
    allowed = set(_normalize_kb_id_scope(kb_ids))
    if not allowed:
        return results
    return [
        result
        for result in results
        if (result.get("metadata") or {}).get("kb_id") in allowed
    ]


def _promote_anchor_result(results: List[dict], query: str, terms: List[str]) -> List[dict]:
    if len(results) <= 1 or len(terms) < 2:
        return results

    anchor_term = terms[0]
    if not _is_anchor_term(anchor_term):
        return results

    anchor_index = None
    for index, result in enumerate(results):
        if anchor_term in _result_search_space(result):
            anchor_index = index
            break

    if anchor_index in (None, 0):
        return results

    promoted = results[anchor_index]
    gap = float(results[0].get("rank_score", 0.0)) - float(promoted.get("rank_score", 0.0))
    boost = max(0.0, gap + 0.01)
    if boost:
        promoted["anchor_alignment_boost"] = round(boost, 6)
        promoted["rank_score"] = float(promoted.get("rank_score", 0.0)) + boost
        promoted["retrieval_score"] = min(1.0, float(promoted.get("retrieval_score", promoted.get("score", 0.0))) + min(boost, 0.35))
        promoted["score"] = promoted["retrieval_score"]

    reordered = [promoted]
    reordered.extend(result for idx, result in enumerate(results) if idx != anchor_index)
    return reordered


def _apply_query_term_rerank(results: List[dict], query: str) -> List[dict]:
    if not results:
        return results

    terms = _query_terms(query)
    if not terms:
        return results

    primary_terms = terms[:2]
    reranked: List[dict] = []

    for result in results:
        metadata = result.get("metadata") or {}
        search_space = _result_search_space(result)

        boost = 0.0
        matched_terms = [term for term in terms if term in search_space]
        if matched_terms:
            boost += 0.18
            if matched_terms[0] in search_space:
                boost += 0.08
            if any(term in str(metadata.get("section_title") or "").lower() for term in terms):
                boost += 0.05
            if any(term in str(metadata.get("heading_path") or "").lower() for term in terms):
                boost += 0.04
        elif any(term in search_space for term in primary_terms):
            boost += 0.08

        if boost:
            result["query_alignment_boost"] = boost
            base_retrieval = float(result.get("retrieval_score", result.get("score", 0.0)))
            result["retrieval_score"] = min(1.0, base_retrieval + boost)
            result["score"] = result["retrieval_score"]
            result["rank_score"] = float(result.get("rank_score", 0.0)) + boost
        reranked.append(result)

    reranked.sort(key=lambda item: item.get("rank_score", 0), reverse=True)
    return _promote_anchor_result(reranked, query, terms)


def _promote_definition_title_candidate(results: List[dict], query: str) -> List[dict]:
    """Keep the title/prelude candidate available for entity definitions."""
    if not results or not _DEFINITION_INTENT_QUERY_RE.search(query or ""):
        return results
    candidate_indexes = [
        index for index, result in enumerate(results)
        if result.get("definition_title_candidate")
    ]
    if not candidate_indexes:
        return results
    best_index = max(
        candidate_indexes,
        key=lambda index: (
            float(results[index].get("rerank_score", results[index].get("rank_score", 0.0))),
            -index,
        ),
    )
    if best_index == 0:
        return results
    promoted = results.pop(best_index)
    promoted["definition_evidence_boost"] = True
    results.insert(0, promoted)
    return results


def _result_term_coverage(result: dict, terms: List[str]) -> set[str]:
    search_space = _result_search_space(result)
    coverage = {term for term in terms if term in search_space}
    support_terms = {
        str(term).lower()
        for term in (result.get("relation_support_terms") or [])
        if term
    }
    return coverage | {term for term in terms if term in support_terms}


def _apply_relation_evidence_rerank(
    results: List[dict],
    query: str,
    top_k: int,
    retrieval_config: RetrievalConfig,
) -> List[dict]:
    relation_config = retrieval_config.relation_query
    if not relation_config.enabled or not relation_config.promote_missing_evidence:
        return results

    anchors = _relation_anchor_terms(query)
    if len(anchors) < 2 or len(results) <= 1:
        return results

    target_window = min(
        len(results),
        max(1, top_k),
        max(1, relation_config.diversify_top_n),
    )
    coverage = [_result_term_coverage(result, anchors) for result in results]
    covered_terms = set().union(*coverage[:target_window])
    missing_terms = [term for term in anchors if term not in covered_terms]
    if not missing_terms:
        return results

    doc_ids_in_window = {
        str((result.get("metadata") or {}).get("doc_id") or "")
        for result in results[:target_window]
    }
    best_index = None
    best_score = None
    best_terms: set[str] = set()
    for index in range(target_window, len(results)):
        matched_missing = coverage[index] & set(missing_terms)
        if not matched_missing:
            continue
        metadata = results[index].get("metadata") or {}
        doc_id = str(metadata.get("doc_id") or "")
        field_text = " ".join(
            value.lower()
            for value in (
                str(metadata.get("filename") or ""),
                str(metadata.get("section_title") or ""),
                str(metadata.get("heading_path") or ""),
            )
            if value
        )
        field_match = any(term in field_text for term in matched_missing)
        candidate_score = (
            len(matched_missing),
            1 if doc_id and doc_id not in doc_ids_in_window else 0,
            1 if field_match else 0,
            float(results[index].get("rank_score", 0.0)),
        )
        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_index = index
            best_terms = matched_missing

    if best_index is None:
        return results

    promoted = results.pop(best_index)
    promoted["relation_evidence_terms"] = sorted(best_terms)
    insert_at = 1 if target_window > 1 else 0
    results.insert(insert_at, promoted)
    return results


def resolve_active_reranker(reranker):
    """Return an already-active reranker without blocking foreground retrieval.

    Runtime-managed providers may start a background recovery probe while
    inactive. Plain adapters and test doubles remain compatible without
    implementing this lifecycle protocol.
    """
    if reranker is None:
        return None
    active = getattr(reranker, "is_active", None)
    if active is False:
        schedule_probe = getattr(reranker, "schedule_probe", None)
        if callable(schedule_probe):
            try:
                schedule_probe()
            except Exception as exc:
                logger.warning("reranker background probe scheduling failed: %s", exc)
        return None
    if active is True:
        return reranker

    # Adapters without runtime state are treated as ready for backward
    # compatibility. Readiness methods are intentionally not awaited here.
    return reranker


async def _resolve_active_reranker(reranker):
    """Compatibility wrapper for older internal callers and tests."""
    return resolve_active_reranker(reranker)


def rrf_fusion(
    vector_results: List[dict],
    bm25_results: List[dict],
    k: int = 60,
) -> List[dict]:
    """RRF (Reciprocal Rank Fusion) 融合排序。"""
    all_chunks: Dict[str, dict] = {}
    rrf_scores: Dict[str, float] = {}

    for rank, result in enumerate(vector_results):
        key = _get_chunk_key(result)
        all_chunks[key] = _merge_chunk_payload(all_chunks.get(key), result)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)

    for rank, result in enumerate(bm25_results):
        key = _get_chunk_key(result)
        all_chunks[key] = _merge_chunk_payload(all_chunks.get(key), result)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)

    sorted_keys = sorted(rrf_scores.keys(), key=lambda item: rrf_scores[item], reverse=True)

    return [
        {**all_chunks[key], "rrf_score": rrf_scores[key]}
        for key in sorted_keys
    ]


async def retrieve(
    query: str,
    embedding_provider: BaseEmbeddingProvider,
    vector_store: VectorStore,
    retrieval_config: RetrievalConfig,
    document_store: Optional[DocumentStore] = None,
    bm25_store: Optional[BM25Store] = None,
    queries: Optional[List[str]] = None,
    collection_name: str | None = None,
    tenant_id: str | None = None,
    allowed_doc_ids: Optional[List[str]] = None,
    allowed_kb_ids: Optional[List[str]] = None,
    query_vectors: Optional[Dict[str, List[float] | None]] = None,
    log_vector_query_timing: bool = True,
    log_bm25_timing: bool = True,
    reranker=None,
    trace_attempt: AttemptTrace | None = None,
    original_query: str | None = None,
) -> List[dict]:
    """检索相关文档片段（支持混合检索 + 多查询）。"""
    started_at = time.perf_counter()
    hybrid_config = retrieval_config.hybrid
    top_k = retrieval_config.top_k
    candidate_multiplier = max(1, retrieval_config.candidate_multiplier)
    active_reranker = await _resolve_active_reranker(reranker)
    rerank_pool_size = (
        max(top_k, int(retrieval_config.reranker.candidate_pool_size))
        if active_reranker is not None
        else top_k * candidate_multiplier
    )
    candidate_k = max(top_k * candidate_multiplier, rerank_pool_size)
    bm25_candidate_k = max(
        hybrid_config.bm25_top_k,
        top_k * max(1, hybrid_config.bm25_search_limit_multiplier),
        rerank_pool_size,
    )
    structured_query = bool(_ENUMERATIVE_QUERY_RE.search(query or ""))
    if structured_query:
        structured_candidate_k = max(top_k * 8, candidate_k)
        candidate_k = structured_candidate_k
        bm25_candidate_k = max(bm25_candidate_k, structured_candidate_k)
    all_queries, relation_support_queries = _augment_relation_queries(query, queries or [query], retrieval_config)
    scoped_doc_ids = _normalize_doc_id_scope(allowed_doc_ids)
    scoped_kb_ids = _normalize_kb_id_scope(allowed_kb_ids)
    vector_where = _build_vector_where_for_doc_scope(scoped_doc_ids, scoped_kb_ids)
    vector_started = time.perf_counter()

    async def _safe_embed_query(current_query: str):
        if query_vectors is not None and current_query in query_vectors:
            return query_vectors[current_query]
        try:
            return await embedding_provider.embed_query(current_query)
        except Exception as exc:
            logger.warning("embedding query failed for %r: %s", current_query, exc)
            return None

    vector_results_map: Dict[str, dict] = {}
    definition_title_candidates = bool(_DEFINITION_INTENT_QUERY_RE.search(query or ""))
    title_candidate_limit = max(1, int(retrieval_config.definition_query_expansion.title_candidate_limit))
    for current_query in all_queries:
        query_vector = await _safe_embed_query(current_query)
        if query_vector is None:
            continue
        vector_results_raw = await _query_vector_store(
            vector_store,
            query_vector,
            candidate_k,
            collection_name,
            where=vector_where,
        )
        vector_searches = [(vector_results_raw, False)]
        if definition_title_candidates and current_query == query:
            title_results_raw = await _query_vector_store(
                vector_store,
                query_vector,
                title_candidate_limit,
                collection_name,
                where=_combine_vector_where(vector_where, {"block_kind": "title"}),
            )
            vector_searches.append((title_results_raw, True))

        for search_results, is_definition_title_candidate in vector_searches:
            for result in _parse_vector_results(search_results, retrieval_config.score_threshold):
                if is_definition_title_candidate:
                    result["definition_title_candidate"] = True
                result["retrieval_queries"] = [current_query]
                support_term = relation_support_queries.get(current_query)
                if support_term:
                    result["relation_support_terms"] = [support_term]
                key = _get_chunk_key(result)
                existing = vector_results_map.get(key)
                if not existing or result["score"] > existing["score"]:
                    vector_results_map[key] = _merge_chunk_payload(existing, result)
                    vector_results_map[key]["score"] = result["score"]
                    vector_results_map[key]["vector_score"] = result["score"]
                else:
                    merged = _merge_chunk_payload(existing, result)
                    vector_results_map[key] = merged

                merged_queries = list(dict.fromkeys((vector_results_map[key].get("retrieval_queries") or []) + [current_query]))
                vector_results_map[key]["retrieval_queries"] = merged_queries
                if support_term:
                    merged_support_terms = list(dict.fromkeys((vector_results_map[key].get("relation_support_terms") or []) + [support_term]))
                    vector_results_map[key]["relation_support_terms"] = merged_support_terms

    vector_results = sorted(
        _filter_results_by_kb_scope(
            _filter_results_by_doc_scope(list(vector_results_map.values()), scoped_doc_ids),
            scoped_kb_ids,
        ),
        key=lambda item: item.get("vector_score", item.get("score", 0)),
        reverse=True,
    )
    if retrieval_config.log_retrieval_timings and log_vector_query_timing:
        logger.info(
            "检索耗时: vector_ms=%d vector_candidate_k=%d queries=%d",
            int((time.perf_counter() - vector_started) * 1000),
            candidate_k,
            len(all_queries),
        )
    if trace_attempt is not None:
        trace_attempt.record_stage(
            "vector",
            candidates=vector_results,
            details={
                "query_count": len(all_queries),
                "candidate_limit": candidate_k,
            },
        )

    if hybrid_config.enabled and bm25_store and bm25_store.is_ready:
        bm25_started = time.perf_counter()
        all_bm25_results = []
        bm25_search_params = inspect.signature(bm25_store.search).parameters
        for current_query in all_queries:
            search_kwargs = {
                "top_k": bm25_candidate_k,
                "collection_name": collection_name,
            }
            if "exact_match_config" in bm25_search_params:
                search_kwargs["exact_match_config"] = retrieval_config.exact_match
            if "allowed_doc_ids" in bm25_search_params:
                search_kwargs["allowed_doc_ids"] = scoped_doc_ids
            if "allowed_kb_ids" in bm25_search_params:
                search_kwargs["allowed_kb_ids"] = scoped_kb_ids
            bm25_hits = await bm25_store.search(current_query, **search_kwargs)
            support_term = relation_support_queries.get(current_query)
            for result in bm25_hits:
                result["retrieval_queries"] = list(dict.fromkeys((result.get("retrieval_queries") or []) + [current_query]))
                if support_term:
                    result["relation_support_terms"] = list(dict.fromkeys((result.get("relation_support_terms") or []) + [support_term]))
            all_bm25_results.extend(bm25_hits)

        all_bm25_results = _filter_results_by_kb_scope(
            _filter_results_by_doc_scope(all_bm25_results, scoped_doc_ids),
            scoped_kb_ids,
        )

        bm25_results_map: Dict[str, dict] = {}
        for result in all_bm25_results:
            key = _get_chunk_key(result)
            bm25_rank_score = _result_score(result, "bm25_rank_score", "bm25_score")
            existing = bm25_results_map.get(key)
            existing_rank_score = _result_score(existing or {}, "bm25_rank_score", "bm25_score")
            if not existing or bm25_rank_score > existing_rank_score:
                bm25_results_map[key] = _merge_chunk_payload(existing, result)
                bm25_results_map[key]["bm25_score"] = result["bm25_score"]
                bm25_results_map[key]["bm25_rank_score"] = bm25_rank_score
                bm25_results_map[key]["exact_match_bonus"] = result.get("exact_match_bonus", 0.0)
                bm25_results_map[key]["exact_match_reasons"] = result.get("exact_match_reasons", [])

        bm25_results = _annotate_lexical_scores(sorted(
            bm25_results_map.values(),
            key=lambda item: item.get("bm25_rank_score", item.get("bm25_score", 0)),
            reverse=True,
        ))

        if hybrid_config.fusion_method == "rrf":
            merged = rrf_fusion(vector_results, bm25_results, k=hybrid_config.rrf_k)
        else:
            merged = _weighted_fusion(
                vector_results,
                bm25_results,
                vector_weight=hybrid_config.vector_weight,
                bm25_weight=hybrid_config.bm25_weight,
            )

        logger.info(f"混合检索: 向量 {len(vector_results)} + BM25 {len(bm25_results)} → 融合 {len(merged)}")
        if retrieval_config.log_retrieval_timings and log_bm25_timing:
            logger.info(
                "检索耗时: bm25_ms=%d bm25_candidate_k=%d vector_candidate_k=%d queries=%d",
                int((time.perf_counter() - bm25_started) * 1000),
                bm25_candidate_k,
                candidate_k,
                len(all_queries),
            )
        if trace_attempt is not None:
            trace_attempt.record_stage(
                "bm25",
                candidates=bm25_results,
                details={
                    "query_count": len(all_queries),
                    "candidate_limit": bm25_candidate_k,
                },
            )
    else:
        merged = vector_results

    merged = _finalize_retrieval_scores(merged, hybrid_config)
    merged = _apply_structured_rerank(merged, query)
    merged = _apply_query_term_rerank(merged, query)
    if trace_attempt is not None:
        trace_attempt.record_stage("fusion", candidates=merged)
    candidate_threshold = (
        retrieval_config.reranker.candidate_prefilter_threshold
        if active_reranker is not None
        else float(retrieval_config.score_threshold)
    )
    if candidate_threshold is not None:
        merged = [
            result
            for result in merged
            if result.get("retrieval_score", 0) >= float(candidate_threshold)
        ]

    filtered = _filter_results_by_kb_scope(merged, scoped_kb_ids)
    filtered = await _filter_by_doc_status(filtered, document_store, tenant_id=tenant_id)
    filtered.sort(key=lambda item: item.get("rank_score", 0), reverse=True)
    if trace_attempt is not None:
        trace_attempt.record_stage("document_status", candidates=filtered)
    legacy_ranked = list(filtered)
    if active_reranker is not None and filtered:
        rerank_candidates = filtered[:rerank_pool_size]
        try:
            scores = await active_reranker.rerank(
                query,
                [item.get("text", "") for item in rerank_candidates],
            )
            if len(scores) != len(rerank_candidates):
                raise ValueError("reranker returned an unexpected number of scores")
            for result, rerank_score in zip(rerank_candidates, scores):
                result["rerank_score"] = float(rerank_score)
                result["rank_score"] = float(rerank_score)
            filtered = sorted(
                rerank_candidates,
                key=lambda item: item.get("rerank_score", float("-inf")),
                reverse=True,
            )
            if trace_attempt is not None:
                trace_attempt.record_stage(
                    "reranker",
                    candidates=filtered,
                    details={"active": True, "candidate_limit": rerank_pool_size},
                )
        except Exception as exc:
            logger.warning("reranker failed; using legacy retrieval order: %s", exc)
            filtered = legacy_ranked
            for result in filtered:
                result["reranker_fallback"] = True
            if trace_attempt is not None:
                trace_attempt.record_stage(
                    "reranker",
                    candidates=filtered,
                    reason="fallback",
                    details={"active": False, "candidate_limit": rerank_pool_size},
                )
    elif trace_attempt is not None:
        trace_attempt.record_stage(
            "reranker",
            candidate_count=0,
            reason="inactive_or_no_candidates",
            details={"active": False, "candidate_limit": rerank_pool_size},
        )

    # Preserve exact section titles and short identifiers (for example L4)
    # after semantic reranking.
    # ``query`` may be a deterministic expansion used only to improve recall
    # (for example "Atlas 产品定位 目标用户 核心能力").  Those helper terms
    # must not be treated as user-authored exact anchors: otherwise an
    # unrelated document containing "核心能力" can be inserted ahead of the
    # reranker's entity match.  Preserve exact sections based on the original
    # user wording when the caller provides it.
    # Keep exact evidence from deterministic intent expansions (for example
    # ``核心用户`` -> ``目标用户`` or an exclusion-boundary expansion), while
    # excluding broad definition helper terms that would over-promote generic
    # sections such as ``核心能力``.
    hard_anchor_queries = [original_query or query]
    for candidate_query in queries or []:
        if candidate_query == (original_query or query):
            continue
        if "产品定位" in candidate_query and "目标用户" in candidate_query and "核心能力" in candidate_query:
            continue
        hard_anchor_queries.append(candidate_query)
    filtered = _promote_hard_exact_candidates(
        filtered,
        legacy_ranked,
        "\n".join(dict.fromkeys(hard_anchor_queries)),
        top_k,
    )

    # Relation/entity coverage is a final-result constraint.  Apply it after
    # semantic reranking so a cross-encoder cannot erase the second entity.
    filtered = _promote_definition_title_candidate(filtered, query)
    filtered = _apply_relation_evidence_rerank(filtered, query, top_k, retrieval_config)

    # Parent-child indexing can yield several precise children for one parent;
    # retain the best child so one section cannot crowd out all evidence slots.
    unique_results: list[dict] = []
    seen_parent_ids: set[str] = set()
    for result in filtered:
        parent_id = str((result.get("metadata") or {}).get("parent_id") or "")
        if parent_id and parent_id in seen_parent_ids:
            continue
        if parent_id:
            seen_parent_ids.add(parent_id)
        unique_results.append(result)
    filtered = unique_results
    if trace_attempt is not None:
        trace_attempt.record_stage("final", candidates=filtered[:top_k])
    logger.info(f"检索到 {len(filtered)} 条结果")
    if retrieval_config.log_retrieval_timings:
        logger.info(
            "检索总耗时: total_ms=%d vector_hits=%d bm25_enabled=%s final_hits=%d queries=%d",
            int((time.perf_counter() - started_at) * 1000),
            len(vector_results),
            bool(hybrid_config.enabled and bm25_store and bm25_store.is_ready),
            len(filtered),
            len(all_queries),
        )
    return filtered[:top_k]


def _weighted_fusion(
    vector_results: List[dict],
    bm25_results: List[dict],
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[dict]:
    """加权融合（需要归一化 BM25 分数）。"""
    all_chunks: Dict[str, dict] = {}
    scores: Dict[str, float] = {}

    for result in vector_results:
        key = _get_chunk_key(result)
        all_chunks[key] = _merge_chunk_payload(all_chunks.get(key), result)
        scores[key] = result.get("vector_score", result.get("score", 0)) * vector_weight

    for result in bm25_results:
        key = _get_chunk_key(result)
        all_chunks[key] = _merge_chunk_payload(all_chunks.get(key), result)
        scores[key] = scores.get(key, 0) + result.get("lexical_score", result.get("bm25_score_normalized", 0)) * bm25_weight

    sorted_keys = sorted(scores.keys(), key=lambda item: scores[item], reverse=True)
    return [{**all_chunks[key], "fusion_score": scores[key]} for key in sorted_keys]


def _parse_vector_results(vector_results_raw: dict, score_threshold: float) -> List[dict]:
    results = []
    if not vector_results_raw or not vector_results_raw.get("documents"):
        return results

    docs = vector_results_raw["documents"][0]
    metas = vector_results_raw["metadatas"][0] if vector_results_raw.get("metadatas") else [{}] * len(docs)
    dists = vector_results_raw["distances"][0] if vector_results_raw.get("distances") else [0] * len(docs)

    for doc, meta, dist in zip(docs, metas, dists):
        metadata = meta or {}
        score = 1 - dist
        chunk_id = metadata.get("chunk_id")
        if not chunk_id and metadata.get("doc_id") and metadata.get("chunk_index") is not None:
            chunk_id = f"{metadata['doc_id']}_{metadata['chunk_index']}"
        results.append({
            "text": doc,
            "metadata": metadata,
            "score": score,
            "vector_score": score,
            "rank_score": score,
            "chunk_id": chunk_id,
        })

    return results


async def _filter_by_doc_status(
    results: List[dict],
    document_store: Optional[DocumentStore],
    tenant_id: str | None = None,
) -> List[dict]:
    """过滤非 ready 状态的文档。"""
    if not document_store:
        return results

    doc_ids = set()
    for result in results:
        metadata = result.get("metadata", {})
        if metadata and metadata.get("doc_id"):
            doc_ids.add(metadata["doc_id"])

    ready_docs = set()
    batch_get = getattr(document_store, "list_by_doc_ids", None)
    if callable(batch_get):
        ready_rows = await batch_get(doc_ids, tenant_id=tenant_id, status="ready")
        ready_docs = {doc["doc_id"] for doc in ready_rows}
    else:
        for doc_id in doc_ids:
            doc = await document_store.get(doc_id, tenant_id=tenant_id)
            if doc and doc.get("status") == "ready":
                ready_docs.add(doc_id)

    return [
        result
        for result in results
        if result.get("metadata", {}).get("doc_id", "") in ready_docs
    ]


async def _query_vector_store(
    vector_store,
    vector,
    top_k: int,
    collection_name: str | None,
    *,
    where: dict | None = None,
):
    query_kwargs = {
        "vector": vector,
        "top_k": top_k,
        "where": where,
        "collection_name": collection_name,
    }
    # Lightweight test doubles and third-party adapters may predate the
    # guarded-read argument. Production VectorStore always supports it.
    if collection_name and "require_existing" in inspect.signature(vector_store.query).parameters:
        query_kwargs["require_existing"] = True
    try:
        return await vector_store.query(**query_kwargs)
    except IndexUnavailableError:
        # An immutable retrieval target disappeared. This is not a low-score
        # result and must reach the caller so recovery/UI can act on it.
        raise
    except Exception as exc:
        logger.warning("vector store query failed: %s", exc)
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
