"""Typed question requirements used by the evidence-support stage.

The contract describes what retrieved material must contain; it does not make
the final support decision and it never infers an answer from missing text.
Keeping this state separate from route selection prevents a successful search
for a nearby entity from being mistaken for proof of the requested claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


EvidenceKind = Literal[
    "ability",
    "binding",
    "collection_overview",
    "definition",
    "factual",
    "numeric",
    "relation",
    "type",
]


_GENERIC_ABILITY_OBJECTS = {
    "什么",
    "些什么",
    "哪些",
    "啥",
    "什么技能",
    "什么能力",
    "哪些技能",
    "哪些能力",
}
_ABILITY_PREDICATE_PATTERNS = (
    re.compile(r"(?:会不会|能不能|可不可以|是否会|是否能|是否可以)(?P<predicate>.+)$"),
    re.compile(r"(?:能够|可以|擅长|会|能)(?P<predicate>.+)$"),
)
_TRAILING_PUNCTUATION_RE = re.compile(r"[？?。！!\s]*$")
_OPEN_LOCATION_PATTERNS = (
    re.compile(r"^(?P<subject>.+?)(?:住在|居住在|位于|来自)(?:哪里|哪儿|何处)$"),
    re.compile(r"^(?P<subject>.+?)(?:住哪里|住哪儿|在哪里|在哪儿|在何处)$"),
)
_LOCATION_EVIDENCE_TERMS = (
    "住在",
    "居住",
    "位于",
    "来自",
    "家在",
    "生活在",
    "定居",
    "地址",
    "所在地",
)


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    """Minimum independently verifiable fields required by a question."""

    kind: EvidenceKind
    required_entities: tuple[str, ...] = ()
    required_facets: tuple[str, ...] = ()
    alternative_fact_groups: tuple[tuple[str, ...], ...] = ()
    required_claim_terms: tuple[str, ...] = ()


def extract_ability_claim_terms(query: str) -> tuple[str, ...]:
    """Extract a concrete capability from Chinese yes/no ability questions.

    Open-ended inventory questions such as ``林小北都会什么`` intentionally
    return no claim term.  A closed question such as ``林小北会开飞机吗``
    returns ``开飞机`` so entity-only evidence can no longer pass the gate.
    The extractor is deliberately conservative: missing a paraphrase yields an
    insufficient-evidence response instead of manufacturing a positive fact.
    """

    normalized = (query or "").strip().lower()
    if not normalized:
        return ()
    for pattern in _ABILITY_PREDICATE_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        predicate = _TRAILING_PUNCTUATION_RE.sub(
            "",
            match.group("predicate").strip(),
        ).strip("：:，,；;、 ")
        if predicate in _GENERIC_ABILITY_OBJECTS:
            return ()
        if predicate.endswith(("吗", "嘛", "呢")):
            predicate = predicate[:-1].strip()
        elif predicate.endswith("么") and not predicate.endswith("什么"):
            predicate = predicate[:-1].strip()
        if not predicate or predicate in _GENERIC_ABILITY_OBJECTS:
            return ()
        if any(token in predicate for token in ("什么", "哪些", "多少")):
            return ()
        return (predicate,)
    return ()


def extract_open_location_contract(
    query: str,
) -> tuple[str, tuple[tuple[str, ...], ...]] | None:
    """Return the subject and location-relation alternatives for open questions."""

    normalized = _TRAILING_PUNCTUATION_RE.sub("", (query or "").strip().lower())
    for pattern in _OPEN_LOCATION_PATTERNS:
        match = pattern.fullmatch(normalized)
        if not match:
            continue
        subject = match.group("subject").strip("：:，,；;、 ")
        if subject:
            return subject, (_LOCATION_EVIDENCE_TERMS,)
    return None
