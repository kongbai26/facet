"""Resolve retrieved candidates into citable, partial, or missing evidence."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.pipeline.answer_policy import EvidenceOutcome, GroundingMode
from app.pipeline.evidence_contract import EvidenceRequirement
from app.pipeline.evidence_policy import (
    _build_evidence_requirement,
    _extend_with_supplementary_evidence,
    _filter_results_to_question_evidence,
    _has_sufficient_evidence_for_queries,
    _select_support_grader_candidates,
    _select_unverified_context_candidates,
)
from app.pipeline.support_grader import SupportVerdict, grade_candidate_support
from app.providers.llm.base import BaseLLMProvider
from app.settings.settings import EvidenceSupportGraderConfig


@dataclass(slots=True)
class EvidenceResolution:
    """Bounded result of the post-retrieval evidence stage."""

    results: list[dict]
    outcome: EvidenceOutcome
    requirement: EvidenceRequirement
    unverified_context: list[dict] = field(default_factory=list)
    support_verdict: SupportVerdict | None = None

    @property
    def used_semantic_grader(self) -> bool:
        return self.support_verdict is not None

    def trace_details(self) -> dict[str, str | int]:
        """Return content-free diagnostic fields for the retrieval trace."""
        return {
            "status": self.outcome.status,
            "kind": self.requirement.kind,
            "required_entity_count": len(self.requirement.required_entities),
            "required_facet_count": len(self.requirement.required_facets),
            "required_claim_count": len(self.requirement.required_claim_terms),
            "alternative_group_count": len(self.requirement.alternative_fact_groups),
            "candidate_count": self.outcome.candidate_count,
            "accepted_count": self.outcome.accepted_count,
            "unverified_context_count": len(self.unverified_context),
            "support_grader_status": (
                self.support_verdict.status if self.support_verdict is not None else "not_used"
            ),
            "support_grader_reason": (
                self.support_verdict.reason if self.support_verdict is not None else ""
            ),
        }


async def resolve_retrieval_evidence(
    query: str,
    retrieval_query: str,
    candidates: list[dict],
    *,
    grounding_mode: GroundingMode,
    support_grader_config: EvidenceSupportGraderConfig,
    llm_provider: BaseLLMProvider | None,
    retain_supplementary_evidence: bool = False,
) -> EvidenceResolution:
    """Resolve evidence with lexical checks as a fast path, not a veto.

    Retrieval and reranking already provide a bounded, scoped candidate set.
    Exact-term checks remain useful when they can prove support cheaply, but
    they cannot reliably model paraphrases, narrative outcomes, or headings.
    When that fast path is inconclusive, the semantic grader makes the final
    support decision over the retrieved candidates. Quality-first deployments
    can make the semantic verdict authoritative for every candidate set.
    """
    retrieval_candidates = list(candidates)
    requirement = _build_evidence_requirement(query)
    evidence_gate_enabled = grounding_mode in {"auto", "knowledge"}
    strict_grounding = grounding_mode == "knowledge"
    had_retrieval_candidates = bool(retrieval_candidates)
    results = list(retrieval_candidates)
    support_verdict: SupportVerdict | None = None
    semantic_support_accepted = False
    unverified_context: list[dict] = []

    if evidence_gate_enabled:
        # Follow-up rewrites contain resolved antecedents, so filter using the
        # standalone retrieval wording rather than the raw pronoun question.
        results = _filter_results_to_question_evidence(retrieval_query, results)
        lexical_support_sufficient = bool(results) and _has_sufficient_evidence_for_queries(
            [query, retrieval_query],
            results,
        )

        # Lexical matching is intentionally not a rejection gate. A source
        # may answer “what happened in the end?” through a chapter title such
        # as “新的起点”, even when it does not repeat the word “结局”. Let the
        # model judge the semantic relation whenever the inexpensive lexical
        # fast path cannot establish it; ``always`` makes that model verdict
        # authoritative for quality-first deployments.
        if (
            (support_grader_config.mode == "always" or not lexical_support_sufficient)
            and support_grader_config.mode in {"auto", "always"}
            and llm_provider is not None
        ):
            grader_candidates = _select_support_grader_candidates(
                retrieval_query,
                retrieval_candidates,
                limit=support_grader_config.max_candidates,
            )
            if grader_candidates:
                support_verdict = await grade_candidate_support(
                    retrieval_query,
                    grader_candidates,
                    llm_provider,
                    timeout_seconds=support_grader_config.timeout_seconds,
                    max_tokens=support_grader_config.max_tokens,
                    max_candidate_chars=support_grader_config.max_candidate_chars,
                )
                if support_verdict.answerable:
                    results = [
                        grader_candidates[index - 1]
                        for index in support_verdict.supported_indices
                    ]
                    semantic_support_accepted = True
                elif (
                    support_grader_config.mode == "always"
                    and support_verdict.status == "insufficient"
                ):
                    # A syntactically valid semantic refusal is a meaningful
                    # decision. Unlike an unavailable grader, it must not be
                    # overridden by coincidental keyword overlap.
                    results = []

        if results and retain_supplementary_evidence:
            results = _extend_with_supplementary_evidence(
                requirement,
                results,
                retrieval_candidates,
            )

    if not results:
        if strict_grounding:
            outcome = EvidenceOutcome.missing(
                "knowledge_no_evidence",
                len(retrieval_candidates),
            )
        elif grounding_mode == "auto":
            unverified_context = _select_unverified_context_candidates(
                retrieval_query,
                retrieval_candidates,
            )
            reason = (
                "auto_evidence_gate_rejected"
                if had_retrieval_candidates
                else "auto_no_evidence"
            )
            outcome = (
                EvidenceOutcome.partial(reason, len(retrieval_candidates))
                if unverified_context
                else EvidenceOutcome.missing(reason, len(retrieval_candidates))
            )
        else:
            outcome = EvidenceOutcome.missing(
                "retrieve_empty_fallback",
                len(retrieval_candidates),
            )
    elif (
        evidence_gate_enabled
        and not semantic_support_accepted
        and not _has_sufficient_evidence_for_queries(
            [query, retrieval_query],
            results,
        )
    ):
        if grounding_mode == "auto":
            unverified_context = _select_unverified_context_candidates(
                retrieval_query,
                retrieval_candidates,
            )
        results = []
        reason = (
            "knowledge_evidence_gate_rejected"
            if strict_grounding
            else "auto_evidence_gate_rejected"
        )
        outcome = (
            EvidenceOutcome.partial(reason, len(retrieval_candidates))
            if grounding_mode == "auto" and unverified_context
            else EvidenceOutcome.missing(reason, len(retrieval_candidates))
        )
    else:
        outcome = EvidenceOutcome.grounded(len(retrieval_candidates), len(results))

    return EvidenceResolution(
        results=results,
        outcome=outcome,
        requirement=requirement,
        unverified_context=unverified_context,
        support_verdict=support_verdict,
    )
