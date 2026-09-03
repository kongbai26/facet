"""Resolve the final answer boundary from a route plan and evidence outcome."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.pipeline.grounding_contract import route_requires_grounded_evidence
from app.pipeline.route_plan import RoutePlan


EvidenceStatus = Literal[
    "not_checked",
    "grounded",
    "boundary",
    "partial",
    "conflict",
    "missing",
    "unavailable",
]
GroundingMode = Literal["auto", "knowledge", "assistant"]
AnswerDecision = Literal["RETRIEVE", "REUSE", "DIRECT", "LIVE_UNSUPPORTED", "NO_EVIDENCE"]
AnswerMode = Literal[
    "auto",
    "auto_fallback",
    "auto_partial",
    "evidence_boundary",
    "evidence_partial",
    "evidence_conflict",
    "verification_unavailable",
    "knowledge_no_evidence",
    "live_unsupported",
]


@dataclass(frozen=True, slots=True)
class EvidenceOutcome:
    """Result of judging retrieved candidates against the user question."""

    status: EvidenceStatus
    reason: str = ""
    candidate_count: int = 0
    accepted_count: int = 0

    @classmethod
    def not_checked(cls) -> "EvidenceOutcome":
        return cls("not_checked")

    @classmethod
    def grounded(cls, candidate_count: int, accepted_count: int) -> "EvidenceOutcome":
        return cls("grounded", "evidence_accepted", candidate_count, accepted_count)

    @classmethod
    def boundary(
        cls,
        reason: str,
        candidate_count: int,
        accepted_count: int,
    ) -> "EvidenceOutcome":
        return cls("boundary", reason, candidate_count, accepted_count)

    @classmethod
    def partial(
        cls,
        reason: str,
        candidate_count: int,
        accepted_count: int = 0,
    ) -> "EvidenceOutcome":
        return cls("partial", reason, candidate_count, accepted_count)

    @classmethod
    def conflict(cls, reason: str, candidate_count: int, accepted_count: int) -> "EvidenceOutcome":
        return cls("conflict", reason, candidate_count, accepted_count)

    @classmethod
    def missing(cls, reason: str, candidate_count: int = 0) -> "EvidenceOutcome":
        return cls("missing", reason, candidate_count, 0)

    @classmethod
    def unavailable(cls, reason: str, candidate_count: int = 0) -> "EvidenceOutcome":
        return cls("unavailable", reason, candidate_count, 0)

    def to_dict(self) -> dict[str, str | int]:
        return {
            "status": self.status,
            "reason": self.reason,
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
        }


@dataclass(frozen=True, slots=True)
class AnswerPolicy:
    """The answer boundary chosen after routing and evidence evaluation."""

    decision: AnswerDecision
    reason: str
    response_mode: AnswerMode
    include_unverified_context: bool = False

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "response_mode": self.response_mode,
            "include_unverified_context": self.include_unverified_context,
        }


def resolve_answer_policy(
    plan: RoutePlan,
    evidence: EvidenceOutcome,
    *,
    grounding_mode: GroundingMode,
) -> AnswerPolicy:
    """Map independent route and evidence states to the existing response modes.

    Automatic chat can answer normally when the semantic router selects
    ``DIRECT``.  When that router selected evidence-required retrieval,
    missing evidence remains a knowledge boundary in both auto and knowledge
    modes; otherwise model memory could overwrite the evidence verdict.
    """
    if plan.decision == "LIVE_UNSUPPORTED":
        return AnswerPolicy("LIVE_UNSUPPORTED", plan.reason, "live_unsupported")

    if evidence.status == "unavailable":
        return AnswerPolicy(
            "NO_EVIDENCE",
            evidence.reason or "evidence_verification_unavailable",
            "verification_unavailable",
        )

    if evidence.status == "boundary":
        return AnswerPolicy(
            "RETRIEVE",
            evidence.reason or "evidence_explicit_boundary",
            "evidence_boundary",
        )

    if evidence.status == "conflict":
        return AnswerPolicy(
            "RETRIEVE",
            evidence.reason or "evidence_conflict",
            "evidence_conflict",
        )

    if evidence.status == "missing":
        if grounding_mode == "knowledge" or route_requires_grounded_evidence(plan):
            return AnswerPolicy(
                "NO_EVIDENCE",
                evidence.reason or plan.reason,
                "knowledge_no_evidence",
            )
        return AnswerPolicy(
            "DIRECT",
            (
                f"{evidence.reason}_direct"
                if evidence.reason in {"knowledge_index_empty", "empty_store_no_evidence"}
                else evidence.reason
            ),
            "auto_fallback",
        )

    if evidence.status == "partial":
        if evidence.accepted_count > 0:
            return AnswerPolicy(
                "RETRIEVE",
                evidence.reason,
                "evidence_partial",
            )
        if grounding_mode == "knowledge" or route_requires_grounded_evidence(plan):
            return AnswerPolicy("NO_EVIDENCE", evidence.reason, "knowledge_no_evidence")
        return AnswerPolicy(
            "DIRECT",
            evidence.reason,
            "auto_partial",
            include_unverified_context=True,
        )

    if plan.decision == "RETRIEVE" and evidence.status == "grounded":
        return AnswerPolicy("RETRIEVE", plan.reason, "auto")
    if plan.decision == "REUSE":
        return AnswerPolicy("REUSE", plan.reason, "auto")
    return AnswerPolicy("DIRECT", plan.reason, "auto")
