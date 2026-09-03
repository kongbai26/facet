"""Benchmark the configured LLM's semantic evidence judgements.

This is intentionally narrower than the end-to-end RAG evaluator: every case
contains a compact, labelled candidate batch so an operator can distinguish a
retrieval miss from an LLM evidence-judgement miss.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from app.config import get_config
from app.evaluation.manifest import build_evaluation_manifest
from app.evaluation.release_gate import validate_minimum_metrics
from app.evaluation.semantic_support import (
    load_semantic_support_cases,
    score_semantic_support_case,
    semantic_support_metrics,
)
from app.pipeline.support_grader import grade_candidate_support
from app.providers.llm.registry import get_llm_provider


def _parse_minimums(parser: argparse.ArgumentParser, values: list[str]) -> dict[str, float]:
    minimums: dict[str, float] = {}
    for value in values:
        name, separator, raw_threshold = value.partition("=")
        if not name or not separator:
            parser.error("--min-metric must use NAME=VALUE")
        try:
            minimums[name.strip()] = float(raw_threshold)
        except ValueError:
            parser.error("--min-metric VALUE must be numeric")
    return minimums


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic support grading with labelled candidate batches."
    )
    parser.add_argument("dataset", type=Path, help="Semantic-support JSON dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Path for the JSON report.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Override support_grader.timeout_seconds for this evaluation.",
    )
    parser.add_argument(
        "--min-metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Fail when a reported metric is below VALUE.",
    )
    args = parser.parse_args()
    minimums = _parse_minimums(parser, args.min_metric)
    cases = load_semantic_support_cases(args.dataset)
    settings = get_config()
    grader_config = settings.retrieval.support_grader
    timeout_seconds = args.timeout_seconds or grader_config.timeout_seconds
    if timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    llm_provider = get_llm_provider(settings.llm)

    rows: list[dict] = []
    category_rows: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        verdict = await grade_candidate_support(
            case.query,
            list(case.candidates),
            llm_provider,
            timeout_seconds=timeout_seconds,
            max_tokens=grader_config.max_tokens,
            max_candidate_chars=grader_config.max_candidate_chars,
        )
        row = score_semantic_support_case(
            case,
            actual_verdict=verdict.status,
            actual_supported_indices=verdict.supported_indices,
        )
        row["reason"] = verdict.reason
        rows.append(row)
        category_rows[case.category].append(row)
        print(
            f"[{case.case_id}] category={case.category} "
            f"expected={case.expected_verdict} actual={verdict.status} passed={row['passed']}"
        )

    metrics = semantic_support_metrics(rows)
    category_metrics = {
        category: semantic_support_metrics(category_cases)
        for category, category_cases in sorted(category_rows.items())
    }
    failures = validate_minimum_metrics(metrics, minimums)
    report = {
        "schema_version": 1,
        "benchmark": "facet_semantic_support",
        "dataset": str(args.dataset),
        "case_count": len(rows),
        "manifest": build_evaluation_manifest(settings),
        "metrics": metrics,
        "category_metrics": category_metrics,
        "gate": {"minimums": minimums, "passed": not failures, "failures": failures},
        # Keep source text out of reports: case ids link back to the supplied
        # corpus, while reports remain safe to attach to issue trackers.
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "gate": report["gate"]}, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
