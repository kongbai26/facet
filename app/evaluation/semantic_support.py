"""Versioned semantic-support benchmark cases and deterministic scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


ExpectedSupportVerdict = Literal["supported", "contradicted", "insufficient"]
_EXPECTED_VERDICTS = frozenset({"supported", "contradicted", "insufficient"})
_ANSWERABLE_VERDICTS = frozenset({"supported", "contradicted"})


def _normalized_indices(value: Any, candidate_count: int, *, field_name: str) -> tuple[int, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of candidate indexes")

    indexes: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{field_name} entries must be integers")
        if item < 1 or item > candidate_count:
            raise ValueError(f"{field_name} index {item} is outside the candidate range")
        if item not in indexes:
            indexes.append(item)
    return tuple(indexes)


@dataclass(frozen=True)
class SemanticSupportCase:
    """One labelled candidate-set judgement for the semantic support grader."""

    case_id: str
    category: str
    query: str
    candidates: tuple[dict[str, Any], ...]
    expected_verdict: ExpectedSupportVerdict
    expected_supported_indices: tuple[int, ...]
    difficulty: str = "standard"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SemanticSupportCase":
        case_id = str(raw.get("id") or raw.get("case_id") or "").strip()
        category = str(raw.get("category") or "").strip()
        query = str(raw.get("query") or "").strip()
        if not case_id or not category or not query:
            raise ValueError("semantic support cases need id, category, and query")

        raw_candidates = raw.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"case {case_id}: candidates must be a non-empty list")
        candidates: list[dict[str, Any]] = []
        for position, raw_candidate in enumerate(raw_candidates, start=1):
            if not isinstance(raw_candidate, dict):
                raise ValueError(f"case {case_id}: candidate {position} must be an object")
            text = str(raw_candidate.get("text") or "").strip()
            if not text:
                raise ValueError(f"case {case_id}: candidate {position} needs text")
            metadata = raw_candidate.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValueError(f"case {case_id}: candidate {position} metadata must be an object")
            candidates.append({**raw_candidate, "text": text, "metadata": dict(metadata)})

        expected_verdict = str(raw.get("expected_verdict") or "").strip().lower()
        if expected_verdict not in _EXPECTED_VERDICTS:
            raise ValueError(f"case {case_id}: unsupported expected_verdict {expected_verdict!r}")
        expected_indices = _normalized_indices(
            raw.get("expected_supported_indices"),
            len(candidates),
            field_name="expected_supported_indices",
        )
        if expected_verdict in _ANSWERABLE_VERDICTS and not expected_indices:
            raise ValueError(f"case {case_id}: answerable verdicts need supporting indexes")
        if expected_verdict == "insufficient" and expected_indices:
            raise ValueError(f"case {case_id}: insufficient verdicts cannot cite candidates")

        return cls(
            case_id=case_id,
            category=category,
            query=query,
            candidates=tuple(candidates),
            expected_verdict=expected_verdict,  # type: ignore[arg-type]
            expected_supported_indices=expected_indices,
            difficulty=str(raw.get("difficulty") or "standard").strip() or "standard",
        )


def load_semantic_support_cases(path: str | Path) -> list[SemanticSupportCase]:
    """Load a labelled semantic-support corpus without touching a knowledge base."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("semantic support dataset must be a list or contain a cases list")
    cases = [SemanticSupportCase.from_dict(row) for row in rows if isinstance(row, dict)]
    if len(cases) != len(rows):
        raise ValueError("every semantic support dataset entry must be an object")
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("semantic support case ids must be unique")
    return cases


def score_semantic_support_case(
    case: SemanticSupportCase,
    *,
    actual_verdict: str,
    actual_supported_indices: Iterable[int],
) -> dict[str, Any]:
    """Score one structured verdict without treating wording as evidence."""
    verdict = str(actual_verdict or "").strip().lower()
    indexes = tuple(dict.fromkeys(
        index for index in actual_supported_indices if isinstance(index, int) and not isinstance(index, bool)
    ))
    verdict_match = verdict == case.expected_verdict
    expected_answerable = case.expected_verdict in _ANSWERABLE_VERDICTS
    actual_answerable = verdict in _ANSWERABLE_VERDICTS
    index_match = set(indexes) == set(case.expected_supported_indices)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "difficulty": case.difficulty,
        "expected_verdict": case.expected_verdict,
        "actual_verdict": verdict,
        "expected_supported_indices": list(case.expected_supported_indices),
        "actual_supported_indices": list(indexes),
        "verdict_match": verdict_match,
        "answerability_match": actual_answerable == expected_answerable,
        "evidence_index_match": verdict_match and index_match,
        "passed": verdict_match and index_match,
    }


def semantic_support_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Return strict, per-case semantic-support benchmark metrics."""
    if not rows:
        return {
            "case_pass_rate": 0.0,
            "verdict_accuracy": 0.0,
            "answerability_accuracy": 0.0,
            "evidence_index_exact_match_rate": 0.0,
        }
    total = len(rows)
    return {
        "case_pass_rate": sum(bool(row.get("passed")) for row in rows) / total,
        "verdict_accuracy": sum(bool(row.get("verdict_match")) for row in rows) / total,
        "answerability_accuracy": sum(bool(row.get("answerability_match")) for row in rows) / total,
        "evidence_index_exact_match_rate": sum(
            bool(row.get("evidence_index_match")) for row in rows
        ) / total,
    }
