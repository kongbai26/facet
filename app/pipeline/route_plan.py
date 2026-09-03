"""Typed state shared by conversational routing and evidence resolution.

The routing model can suggest an action, but it must not decide whether the
retrieved material is sufficient to answer.  ``RoutePlan`` therefore records
only the pre-retrieval plan.  ``EvidenceOutcome`` and ``AnswerPolicy`` live in
their own module and resolve the final response boundary afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


RouteAction = Literal["RETRIEVE", "REUSE", "DIRECT", "LIVE_UNSUPPORTED"]
RouteIntent = Literal["knowledge", "follow_up", "conversation", "live", "ambiguous"]
EvidencePolicy = Literal["required", "probe", "reuse_verified", "not_required"]
RouteOrigin = Literal["heuristic", "llm", "fallback", "configuration", "guardrail", "availability"]
RouteConfidence = Literal["high", "medium", "low"]


def _intent_for_action(action: RouteAction) -> RouteIntent:
    if action == "RETRIEVE":
        return "knowledge"
    if action == "REUSE":
        return "follow_up"
    if action == "LIVE_UNSUPPORTED":
        return "live"
    return "conversation"


def _evidence_policy_for_action(action: RouteAction) -> EvidencePolicy:
    if action == "RETRIEVE":
        return "required"
    if action == "REUSE":
        return "reuse_verified"
    return "not_required"


@dataclass(slots=True)
class RoutePlan:
    """Pre-retrieval routing state with a stable compatibility surface.

    ``decision`` intentionally retains the previous public name so callers,
    diagnostics, and saved evaluation fixtures remain compatible while the
    implementation moves to a two-stage route/evidence architecture.
    """

    decision: RouteAction
    reason: str
    intent: RouteIntent | None = None
    evidence_policy: EvidencePolicy | None = None
    origin: RouteOrigin | None = None
    confidence: RouteConfidence = "high"
    used_llm_gate: bool = False
    fallback_used: bool = False

    def __post_init__(self) -> None:
        if self.intent is None:
            self.intent = _intent_for_action(self.decision)
        if self.evidence_policy is None:
            self.evidence_policy = _evidence_policy_for_action(self.decision)
        if self.origin is None:
            if self.fallback_used:
                self.origin = "fallback"
            elif self.used_llm_gate:
                self.origin = "llm"
            elif self.reason.startswith("mode_"):
                self.origin = "configuration"
            elif "guard" in self.reason or "invalid" in self.reason:
                self.origin = "guardrail"
            else:
                self.origin = "heuristic"

    def transition(
        self,
        decision: RouteAction,
        reason: str,
        *,
        evidence_policy: EvidencePolicy | None = None,
        intent: RouteIntent | None = None,
        origin: RouteOrigin | None = None,
    ) -> "RoutePlan":
        """Return a later-stage plan while retaining route provenance."""
        return replace(
            self,
            decision=decision,
            reason=reason,
            evidence_policy=evidence_policy or _evidence_policy_for_action(decision),
            intent=intent or self.intent,
            origin=origin or self.origin,
        )

    def to_dict(self) -> dict[str, str | bool]:
        """Return bounded diagnostic fields; no query or document content."""
        return {
            "decision": self.decision,
            "reason": self.reason,
            "intent": self.intent or "ambiguous",
            "evidence_policy": self.evidence_policy or "not_required",
            "origin": self.origin or "heuristic",
            "confidence": self.confidence,
            "used_llm_gate": self.used_llm_gate,
            "fallback_used": self.fallback_used,
        }
