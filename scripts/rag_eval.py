"""Run an end-to-end offline RAG quality suite against the local index.

The script intentionally invokes the same ``prepare_chat_turn`` and
``generate`` functions used by the API.  It is a release tool, not an online
feature: no report data is written to the knowledge base and no extra model is
required beyond the configured local LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

from app.config import get_config
from app.evaluation.manifest import build_evaluation_manifest
from app.evaluation.metrics import (
    answer_metrics,
    evaluate_answer,
    find_ranks,
    retrieval_metrics,
)
from app.evaluation.models import load_evaluation_cases
from app.evaluation.release_gate import validate_minimum_metrics
from app.pipeline.chat_flow import prepare_chat_turn
from app.pipeline.generation import generate
from app.prompt_profile import resolve_prompt_profile
from app.providers.embedding.registry import get_embedding_provider
from app.providers.llm.registry import get_llm_provider
from app.providers.reranker.registry import get_reranker
from app.rag_scope import resolve_embedding_dimension
from app.store.bm25_store import BM25Store
from app.store.document_store import DocumentStore
from app.store.vector_store import VectorStore


def _parse_minimums(parser: argparse.ArgumentParser, values: list[str]) -> dict[str, float]:
    minimums: dict[str, float] = {}
    for item in values:
        name, separator, raw_value = item.partition("=")
        if not name or not separator:
            parser.error("--min-metric must use NAME=VALUE")
        try:
            minimums[name.strip()] = float(raw_value)
        except ValueError:
            parser.error("--min-metric value must be numeric")
    return minimums


def _safe_candidates(results: list[dict]) -> list[dict]:
    """Persist identities and scores, never document bodies, in the report."""
    candidates: list[dict] = []
    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}
        candidates.append(
            {
                "rank": rank,
                "doc_id": metadata.get("doc_id"),
                "chunk_id": result.get("chunk_id") or metadata.get("chunk_id"),
                "chunk_index": metadata.get("chunk_index"),
                "block_kind": metadata.get("block_kind"),
                "retrieval_score": result.get("retrieval_score"),
                "rank_score": result.get("rank_score"),
                "rerank_score": result.get("rerank_score"),
            }
        )
    return candidates


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 2)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the current Facet index end to end.")
    parser.add_argument("dataset", type=Path, help="Versioned JSON evaluation dataset.")
    parser.add_argument("--tenant-id", default=None, help="Optional tenant id of the evaluated corpus.")
    parser.add_argument("--tenant-slug", default=None, help="Optional tenant slug of the evaluated corpus.")
    parser.add_argument("--output", type=Path, required=True, help="Path for the complete JSON report.")
    parser.add_argument(
        "--case-timeout-seconds",
        type=int,
        default=0,
        help="Bound one end-to-end case; 0 uses the configured LLM request timeout.",
    )
    parser.add_argument(
        "--reranker",
        choices=("configured", "off"),
        default="configured",
        help="Use the configured optional reranker or force the legacy path for A/B runs.",
    )
    parser.add_argument(
        "--grounding-mode",
        choices=("knowledge", "auto"),
        default="knowledge",
        help="Evaluate explicit knowledge grounding or the user-facing automatic route.",
    )
    parser.add_argument(
        "--min-metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Fail the command when a deterministic aggregate metric is below VALUE.",
    )
    args = parser.parse_args()
    minimums = _parse_minimums(parser, args.min_metric)
    cases = load_evaluation_cases(args.dataset)

    settings = get_config()
    if args.reranker == "off":
        settings.retrieval.reranker.enabled = False
        settings.retrieval.reranker.mode = "off"

    embedding_provider = get_embedding_provider(settings.embedding, settings.vectorstore)
    llm_provider = get_llm_provider(settings.llm)
    reranker = get_reranker(settings.retrieval.reranker)
    if reranker is not None and hasattr(reranker, "initialize"):
        await reranker.initialize()
    vector_store = VectorStore(settings.vectorstore, settings.embedding.openai.model_name)
    document_store = DocumentStore(settings.storage.metadata_db)
    bm25_store = BM25Store(
        cache_dir=settings.retrieval.hybrid.bm25_cache_dir,
        lexical_metadata_fields=settings.retrieval.exact_match.lexical_metadata_fields,
    )

    answer_rows: list[dict] = []
    retrieval_rows: list[dict] = []
    case_timeout = args.case_timeout_seconds or settings.llm.request_timeout
    if case_timeout < 1:
        parser.error("--case-timeout-seconds must be positive")
    for case in cases:
        print(f"[{case.case_id}] evaluating")
        case_started = time.perf_counter()
        retrieval_started = case_started
        turn = await prepare_chat_turn(
            case.query,
            [],
            settings,
            llm_provider,
            embedding_provider,
            vector_store,
            document_store,
            bm25_store,
            reranker=reranker,
            tenant_id=args.tenant_id,
            tenant_slug=args.tenant_slug,
            grounding_mode=args.grounding_mode,
            diagnostics=True,
        )
        retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 2)
        response_mode = turn.response_mode
        generation_error = ""
        generation_started = time.perf_counter()
        try:
            answer = await asyncio.wait_for(
                generate(
                    case.query,
                    turn.results,
                    llm_provider,
                    prompt_profile=resolve_prompt_profile(settings),
                    response_mode=response_mode,
                    context_window=settings.llm.context_window,
                    max_output_tokens=settings.llm.max_tokens,
                    relevance_threshold=settings.retrieval.relevance_threshold,
                    history_limit=settings.chat.history_limit,
                    history_truncate=settings.chat.history_truncate,
                    validation_max_retries=settings.chat.answer_validation_max_retries,
                    unverified_context=turn.unverified_context,
                ),
                timeout=case_timeout,
            )
        except asyncio.TimeoutError:
            answer = ""
            generation_error = f"generation timed out after {case_timeout}s"
        except Exception as exc:
            answer = ""
            generation_error = f"{exc.__class__.__name__}: {exc}"
        generation_ms = round((time.perf_counter() - generation_started) * 1000, 2)
        evaluated = evaluate_answer(
            case,
            answer,
            turn.results,
            context_window=settings.llm.context_window,
            max_output_tokens=settings.llm.max_tokens,
            relevance_threshold=settings.retrieval.relevance_threshold,
        )
        expected_decision = case.expected_decisions.get(args.grounding_mode) or case.expected_decision
        decision_matches = not expected_decision or turn.decision == expected_decision
        row = {
            "case_id": case.case_id,
            "query": case.query,
            "category": case.category,
            "difficulty": case.difficulty,
            "decision": turn.decision,
            "decision_reason": turn.reason,
            "expected_decision": expected_decision,
            "decision_matches": decision_matches,
            "fallback_used": turn.fallback_used,
            "route_plan": turn.route_plan,
            "evidence": turn.evidence,
            "answer_policy": turn.answer_policy,
            "generation_error": generation_error,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": round((time.perf_counter() - case_started) * 1000, 2),
            "retrieval_query": turn.retrieval_query,
            "candidates": _safe_candidates(turn.results),
            "answer": answer,
            **evaluated,
        }
        answer_rows.append(row)
        if not case.should_abstain:
            top_ids = [candidate.get("doc_id") or "" for candidate in row["candidates"][:3]]
            retrieval_rows.append(
                {
                    "expected_ids": list(case.expected_doc_ids),
                    "ranks": find_ranks(turn.results, case.expected_doc_ids),
                    "top_ids": top_ids,
                }
            )

    aggregate = {
        **retrieval_metrics(retrieval_rows, top_k=3),
        **answer_metrics(answer_rows),
        "decision_match_rate": sum(bool(row["decision_matches"]) for row in answer_rows) / max(len(answer_rows), 1),
        "generation_success_rate": sum(not row["generation_error"] for row in answer_rows) / max(len(answer_rows), 1),
        "latency_ms": {
            "retrieval_p50": _percentile([row["retrieval_ms"] for row in answer_rows], 0.50),
            "retrieval_p95": _percentile([row["retrieval_ms"] for row in answer_rows], 0.95),
            "generation_p50": _percentile([row["generation_ms"] for row in answer_rows], 0.50),
            "generation_p95": _percentile([row["generation_ms"] for row in answer_rows], 0.95),
            "total_p50": _percentile([row["total_ms"] for row in answer_rows], 0.50),
            "total_p95": _percentile([row["total_ms"] for row in answer_rows], 0.95),
        },
    }
    failures = validate_minimum_metrics(aggregate, minimums)
    dimension = await resolve_embedding_dimension(embedding_provider)
    reranker_status = reranker.status() if reranker is not None and hasattr(reranker, "status") else {}
    report = {
        "schema_version": 1,
        "benchmark": "facet_end_to_end",
        "dataset": str(args.dataset),
        "tenant": {"tenant_id": args.tenant_id, "tenant_slug": args.tenant_slug},
        "manifest": build_evaluation_manifest(
            settings,
            embedding_dimension=dimension,
            reranker_status=reranker_status,
        ),
        "metrics": aggregate,
        "gate": {"minimums": minimums, "passed": not failures, "failures": failures},
        "rows": answer_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": aggregate, "gate": report["gate"], "report": str(args.output)}, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
