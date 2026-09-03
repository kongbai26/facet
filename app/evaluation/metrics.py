"""Deterministic metrics for offline retrieval and answer regression checks."""

from __future__ import annotations

import re
from typing import Any, Iterable

from app.evaluation.models import EvaluationCase
from app.pipeline.generation import (
    detect_answer_constraint_violations,
    extract_citation_indexes,
)


def find_ranks(results: Iterable[dict], expected_doc_ids: Iterable[str]) -> list[int]:
    """Return one-based ranks of every expected document in a result list."""
    expected = set(expected_doc_ids)
    return [
        index
        for index, result in enumerate(results, start=1)
        if (result.get("metadata") or {}).get("doc_id") in expected
    ]


def retrieval_metrics(rows: list[dict], *, top_k: int = 3) -> dict[str, float]:
    """Calculate document-level metrics while retaining multi-document labels."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    metric_names = {
        "hit@1": 0.0,
        f"hit@{top_k}": 0.0,
        "mrr": 0.0,
        f"context_precision@{top_k}": 0.0,
        f"context_recall@{top_k}": 0.0,
    }
    if not rows:
        return metric_names

    total = len(rows)
    hit_at_one = sum(1 for row in rows if any(rank == 1 for rank in row["ranks"]))
    hit_at_k = sum(1 for row in rows if any(rank <= top_k for rank in row["ranks"]))
    reciprocal_rank = sum(
        1.0 / min(row["ranks"]) if row["ranks"] else 0.0
        for row in rows
    )
    precision = 0.0
    recall = 0.0
    for row in rows:
        expected = set(row["expected_ids"])
        top_ids = list(dict.fromkeys(
            doc_id for doc_id in list(row.get("top_ids", row.get("top3", [])))[:top_k] if doc_id
        ))
        relevant = {doc_id for doc_id in top_ids if doc_id in expected}
        precision += len(relevant) / max(len(top_ids), 1)
        recall += len(relevant) / max(len(expected), 1)

    return {
        "hit@1": hit_at_one / total,
        f"hit@{top_k}": hit_at_k / total,
        "mrr": reciprocal_rank / total,
        f"context_precision@{top_k}": precision / total,
        f"context_recall@{top_k}": recall / total,
    }


def _normalize_text(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", (value or "").lower(), flags=re.UNICODE)


def _matches_abstention(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return False
    refusal_signals = (
        ("知识库", "资料不足"),
        ("知识库", "没有找到", "依据"),
        ("参考资料", "未提及"),
        ("当前资料", "无法"),
        ("补充", "相关资料"),
    )
    return any(all(signal in text for signal in group) for group in refusal_signals)


def _fact_coverage(answer: str, fact_groups: tuple[tuple[str, ...], ...]) -> tuple[float, list[bool]]:
    if not fact_groups:
        return 1.0, []
    normalized_answer = _normalize_text(answer)
    matched = [
        any(_normalize_text(choice) in normalized_answer for choice in group)
        for group in fact_groups
    ]
    return sum(matched) / len(matched), matched


def evaluate_answer(
    case: EvaluationCase,
    answer: str,
    results: list[dict],
    *,
    context_window: int,
    max_output_tokens: int,
    relevance_threshold: float,
) -> dict[str, Any]:
    """Evaluate deterministic release properties of one generated answer.

    This intentionally does not claim semantic faithfulness.  That requires a
    separately configured judge and is reported as advisory in a later phase.
    """
    answer = (answer or "").strip()
    abstained = _matches_abstention(answer)
    citations = extract_citation_indexes(answer)
    violations = detect_answer_constraint_violations(
        answer,
        results,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        relevance_threshold=relevance_threshold,
    )
    fact_coverage, fact_matches = _fact_coverage(answer, case.expected_facts)
    citation_valid = not violations["invalid_citations"] and not violations["missing_citation_paragraphs"]

    if case.should_abstain:
        passed = abstained
    else:
        passed = (
            not abstained
            and fact_coverage == 1.0
            and citation_valid
            and (not case.must_cite or bool(citations))
            and not violations["has_internal_notes"]
        )
    return {
        "case_id": case.case_id,
        "passed": passed,
        "abstained": abstained,
        "abstention_expected": case.should_abstain,
        "fact_coverage": fact_coverage,
        "fact_matches": fact_matches,
        "citation_indexes": citations,
        "citation_valid": citation_valid,
        "invalid_citations": violations["invalid_citations"],
        "missing_citation_paragraphs": violations["missing_citation_paragraphs"],
        "has_internal_notes": violations["has_internal_notes"],
    }


def answer_metrics(rows: list[dict]) -> dict[str, float]:
    """Aggregate deterministic answer checks without hiding failed cases."""
    if not rows:
        return {
            "answer_pass_rate": 0.0,
            "fact_coverage": 0.0,
            "citation_valid_rate": 0.0,
            "abstention_accuracy": 0.0,
        }
    total = len(rows)
    abstention_rows = [row for row in rows if row.get("abstention_expected")]
    return {
        "answer_pass_rate": sum(bool(row.get("passed")) for row in rows) / total,
        "fact_coverage": sum(float(row.get("fact_coverage", 0.0)) for row in rows) / total,
        "citation_valid_rate": sum(bool(row.get("citation_valid")) for row in rows) / total,
        "abstention_accuracy": (
            sum(bool(row.get("abstained")) for row in abstention_rows) / len(abstention_rows)
            if abstention_rows else 1.0
        ),
    }
