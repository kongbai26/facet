"""Versioned, human-labelled cases for offline RAG evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _as_nonempty_strings(value: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list):
        values = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        raise ValueError(f"{field_name} must be a string or a list of strings")
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return values


def _fact_groups(value: Any) -> tuple[tuple[str, ...], ...]:
    """Normalize fact labels into ANDed groups with ORed alternatives.

    ``["A", "B"]`` means both A and B must appear.  ``[["A", "A 的别名"]]``
    means either wording is acceptable.  Exact phrase checks are intentional:
    they are a deterministic release signal, not a substitute for semantic
    evaluation by a separately configured judge.
    """
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ValueError("expected_facts must be a list")
    groups: list[tuple[str, ...]] = []
    for item in value:
        if isinstance(item, str):
            group = (item.strip(),)
        elif isinstance(item, list):
            group = tuple(str(choice).strip() for choice in item if str(choice).strip())
        else:
            raise ValueError("expected_facts entries must be strings or lists of strings")
        if not group:
            raise ValueError("expected_facts must not contain empty entries")
        groups.append(group)
    return tuple(groups)


@dataclass(frozen=True)
class EvaluationCase:
    """One knowledge-scoped question and its release expectations."""

    case_id: str
    query: str
    category: str
    expected_doc_ids: tuple[str, ...] = ()
    expected_facts: tuple[tuple[str, ...], ...] = ()
    should_abstain: bool = False
    must_cite: bool = True
    expected_decision: str = ""
    expected_decisions: dict[str, str] = field(default_factory=dict)
    difficulty: str = "standard"
    document_types: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvaluationCase":
        case_id = str(raw.get("id") or raw.get("case_id") or "").strip()
        query = str(raw.get("query") or "").strip()
        category = str(raw.get("category") or "").strip()
        if not case_id or not query or not category:
            raise ValueError("every evaluation case needs id, query, and category")

        should_abstain = bool(raw.get("should_abstain", False))
        expected_doc_ids_raw = raw.get("expected_doc_ids", [])
        expected_doc_ids = () if expected_doc_ids_raw in (None, []) else _as_nonempty_strings(
            expected_doc_ids_raw,
            field_name="expected_doc_ids",
        )
        if should_abstain and expected_doc_ids:
            raise ValueError(f"case {case_id}: abstention cases cannot declare expected_doc_ids")
        if not should_abstain and not expected_doc_ids:
            raise ValueError(f"case {case_id}: knowledge cases need expected_doc_ids")
        expected_decisions_raw = raw.get("expected_decisions") or {}
        if not isinstance(expected_decisions_raw, dict):
            raise ValueError("expected_decisions must be an object mapping grounding mode to decision")

        document_types_raw = raw.get("document_types", [])
        document_types = () if document_types_raw in (None, []) else _as_nonempty_strings(
            document_types_raw,
            field_name="document_types",
        )
        return cls(
            case_id=case_id,
            query=query,
            category=category,
            expected_doc_ids=expected_doc_ids,
            expected_facts=_fact_groups(raw.get("expected_facts")),
            should_abstain=should_abstain,
            must_cite=bool(raw.get("must_cite", not should_abstain)),
            expected_decision=str(raw.get("expected_decision") or "").strip(),
            expected_decisions={
                str(mode).strip(): str(decision).strip()
                for mode, decision in expected_decisions_raw.items()
                if str(mode).strip() and str(decision).strip()
            },
            difficulty=str(raw.get("difficulty") or "standard").strip() or "standard",
            document_types=document_types,
        )


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load a JSON list or a ``{\"cases\": [...]}`` versioned dataset."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("evaluation dataset must be a list or contain a cases list")
    cases = [EvaluationCase.from_dict(item) for item in rows if isinstance(item, dict)]
    if len(cases) != len(rows):
        raise ValueError("every evaluation dataset entry must be an object")
    identifiers = [case.case_id for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("evaluation case ids must be unique")
    return cases
