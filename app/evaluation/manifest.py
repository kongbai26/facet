"""Non-secret configuration snapshot attached to every evaluation report."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from app.utils.model_labels import display_model_name
from app.utils.user_errors import sanitize_diagnostic_detail


def build_evaluation_manifest(
    settings,
    *,
    embedding_dimension: int | None = None,
    reranker_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture only quality-relevant, non-secret configuration."""
    payload = {
        "embedding": {
            "provider": settings.embedding.provider,
            "model": display_model_name(settings.embedding.openai.model_name),
            "dimension": embedding_dimension,
            "max_tokens": settings.embedding.openai.max_tokens,
        },
        "llm": {
            "provider": settings.llm.provider,
            "model": display_model_name(settings.llm.model_name),
            "context_window": settings.llm.context_window,
            "max_tokens": settings.llm.max_tokens,
            "temperature": settings.llm.temperature,
        },
        "chunking": settings.chunking.model_dump(),
        "retrieval": {
            "top_k": settings.retrieval.top_k,
            "score_threshold": settings.retrieval.score_threshold,
            "relevance_threshold": settings.retrieval.relevance_threshold,
            "candidate_multiplier": settings.retrieval.candidate_multiplier,
            "hybrid": settings.retrieval.hybrid.model_dump(),
            "exact_match": settings.retrieval.exact_match.model_dump(),
            "query_rewrite": settings.retrieval.query_rewrite.model_dump(),
            "definition_query_expansion": settings.retrieval.definition_query_expansion.model_dump(),
            "relation_query": settings.retrieval.relation_query.model_dump(),
            "support_grader": settings.retrieval.support_grader.model_dump(),
            "reranker": {
                "enabled": settings.retrieval.reranker.enabled,
                "mode": settings.retrieval.reranker.mode,
                "provider": settings.retrieval.reranker.provider,
                "expected_model": display_model_name(settings.retrieval.reranker.expected_model),
                "probe_timeout_seconds": settings.retrieval.reranker.probe_timeout_seconds,
                "request_timeout": settings.retrieval.reranker.request_timeout,
                "candidate_pool_size": settings.retrieval.reranker.candidate_pool_size,
                "candidate_prefilter_threshold": settings.retrieval.reranker.candidate_prefilter_threshold,
                "score_mode": settings.retrieval.reranker.score_mode,
                "min_score": settings.retrieval.reranker.min_score,
            },
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        "config": payload,
        "reranker_status": {
            **dict(reranker_status or {}),
            "model_name": display_model_name((reranker_status or {}).get("model_name")),
            "expected_model": display_model_name((reranker_status or {}).get("expected_model")),
            "last_error": sanitize_diagnostic_detail((reranker_status or {}).get("last_error")),
        },
    }
