"""Evaluate the conversational router against a real local knowledge base."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from app.config import get_config
from app.pipeline.chat_flow import prepare_chat_turn, prepare_direct_chat_turn
from app.providers.embedding.registry import get_embedding_provider
from app.providers.llm.registry import get_llm_provider
from app.providers.reranker.registry import get_reranker
from app.store.bm25_store import BM25Store
from app.store.document_store import DocumentStore
from app.store.vector_store import VectorStore


def _history_from_row(row: dict) -> list[dict]:
    return [
        {"role": "user", "content": row["query"]},
        {
            "role": "assistant",
            "status": "completed",
            "content": "根据资料完成了上一轮回答。",
            "sources": row.get("sources", []),
        },
    ]


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate auto/manual conversational routing.")
    parser.add_argument("dataset", type=Path, help="Routing JSON dataset.")
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--tenant-slug", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    settings = get_config()
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

    rows: list[dict] = []
    by_id: dict[str, dict] = {}
    for case in cases:
        history_row = by_id.get(case.get("history_case_id"))
        history = _history_from_row(history_row) if history_row else []
        mode = case.get("mode", "auto")
        if mode == "assistant":
            turn = prepare_direct_chat_turn(case["query"], grounding_mode="assistant")
        else:
            turn = await prepare_chat_turn(
                case["query"],
                history,
                settings,
                llm_provider,
                embedding_provider,
                vector_store,
                document_store,
                bm25_store,
                reranker=reranker,
                tenant_id=args.tenant_id,
                tenant_slug=args.tenant_slug,
                grounding_mode=mode,
            )
        row = {
            "id": case["id"],
            "query": case["query"],
            "mode": mode,
            "decision": turn.decision,
            "reason": turn.reason,
            "expected_decision": case.get("expected_decision"),
            "expected_reason": case.get("expected_reason"),
            "decision_match": not case.get("expected_decision") or turn.decision == case["expected_decision"],
            "reason_match": (
                not case.get("expected_reason") and not case.get("expected_reasons")
            ) or (
                turn.reason == case.get("expected_reason")
                if case.get("expected_reason")
                else turn.reason in set(case.get("expected_reasons") or [])
            ),
            "result_count": len(turn.results),
            "source_count": len(turn.sources),
            "retrieval_query": turn.retrieval_query,
            "sources": turn.sources,
            "route_plan": turn.route_plan,
            "evidence": turn.evidence,
            "answer_policy": turn.answer_policy,
        }
        rows.append(row)
        by_id[case["id"]] = row
        print(f"[{case['id']}] mode={mode} decision={turn.decision} reason={turn.reason} results={len(turn.results)}")

    decision_matches = sum(row["decision_match"] for row in rows)
    reason_matches = sum(row["reason_match"] for row in rows)
    report = {
        "dataset": str(args.dataset),
        "case_count": len(rows),
        "decision_match_rate": decision_matches / max(len(rows), 1),
        "reason_match_rate": reason_matches / max(len(rows), 1),
        "route_intents": dict(Counter(
            str((row.get("route_plan") or {}).get("intent") or "unknown")
            for row in rows
        )),
        "route_origins": dict(Counter(
            str((row.get("route_plan") or {}).get("origin") or "unknown")
            for row in rows
        )),
        "evidence_statuses": dict(Counter(
            str((row.get("evidence") or {}).get("status") or "unknown")
            for row in rows
        )),
        "answer_modes": dict(Counter(
            str((row.get("answer_policy") or {}).get("response_mode") or "unknown")
            for row in rows
        )),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_count", "decision_match_rate", "reason_match_rate")}, ensure_ascii=False, indent=2))
    return 0 if decision_matches == len(rows) and reason_matches == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
