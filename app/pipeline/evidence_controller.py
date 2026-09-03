"""LLM-led evidence decisions and bounded corrective retrieval actions."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Literal

from app.pipeline.evidence_scope import render_evidence_candidate
from app.pipeline.grounding_contract import (
    CORE_EVIDENCE_PRINCIPLES,
    EvidenceClaim,
    parse_evidence_claims,
)
from app.providers.llm.base import BaseLLMProvider


EvidenceAction = Literal[
    "answer",
    "boundary",
    "expand",
    "retry",
    "partial",
    "conflict",
    "abstain",
    "unavailable",
]
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_EVIDENCE_CONTROLLER_PROMPT = f"""你是知识证据控制器。请判断候选资料能否支持用户原问题，并决定下一步。

{CORE_EVIDENCE_PRINCIPLES}

每条资料都带有系统生成的覆盖边界：
- scope=complete_document 表示该候选覆盖完整文档；document_fragment/unknown_fragment 都不是全文。
- shown_text_complete=no 表示本次展示本身被截断，不能据此判断被省略部分没有某项事实。
- has_more_before/has_more_after 表示同一文档是否存在可补充的相邻内容。
仅仅“当前片段没有出现某事实”不能支持“原文、全文或整个资料没有提及”的结论。只有资料明确陈述该事实不存在/未披露，或已完整展示了所判断的范围，才可支持相应的否定或缺失结论。

action 定义：
- answer：资料足以回答问题，supported_indices 列出真正支持答案的资料。
- boundary：资料没有给出用户所求的具体值或结论，但明确说明该信息未披露、未知、尚未确定、不可获得或不在其覆盖范围；
  此时资料足以回答“为什么不能给出该具体事实”，supported_indices 必须列出直接支持这一边界的资料。
  不能因为局部片段没有出现某事实就选择 boundary；普通的否定事实能够直接回答时仍选择 answer。
- expand：资料相关，但局部边界或展示截断使必要上下文不完整；用 expand_indices 指定锚点，并用 expand_direction 指定 before、after 或 both。
- retry：当前资料不够，但可以针对缺失事实改写检索问句；retry_query 必须具体且保持原问题含义。
- partial：只能回答一部分；supported_indices 和 missing_facets 都必须给出。
- conflict：资料在同一事实和适用范围下互相冲突；列出冲突资料。
- abstain：当前范围内既没有足以回答问题的证据，也没有明确说明该信息边界的证据，而且不应继续检索。

claims 是已经获得资料支持、允许进入最终答案的主张清单：
- kind=fact：资料直接表达或合理释义；kind=derived：可由资料完整输入唯一推导；
  kind=boundary：资料明确表达的信息边界；kind=conflict：资料间无法消解的冲突。
- source_indices 必须精确列出支持该主张的候选编号。数值推导应把可复核算式写入 expression。
- claims 不是最终答案，最多 8 条；不要为没有证据的内容创建 claim。

只输出一个 JSON 对象：
{{"action":"answer|boundary|expand|retry|partial|conflict|abstain","supported_indices":[1],"conflicting_indices":[],"missing_facets":[],"retry_query":"","expand_indices":[],"expand_direction":"both","claims":[{{"statement":"","kind":"fact|derived|boundary|conflict","source_indices":[1],"expression":""}}],"reason_code":""}}

不要输出分数、解释文字或 Markdown。是否还能补上下文或重试分别由 allow_expand、allow_retry 决定；为 no 时不得选择对应动作。
"""


@dataclass(frozen=True, slots=True)
class EvidenceControlDecision:
    action: EvidenceAction
    supported_indices: tuple[int, ...] = ()
    conflicting_indices: tuple[int, ...] = ()
    missing_facets: tuple[str, ...] = ()
    retry_query: str = ""
    expand_indices: tuple[int, ...] = ()
    expand_direction: Literal["before", "after", "both"] = "both"
    claims: tuple[EvidenceClaim, ...] = ()
    reason: str = ""

    @property
    def selected_indices(self) -> tuple[int, ...]:
        claim_indices = tuple(
            index
            for claim in self.claims
            for index in claim.source_indices
        )
        return tuple(
            dict.fromkeys((*self.supported_indices, *self.conflicting_indices, *claim_indices))
        )

    def to_guidance(
        self,
        *,
        source_index_map: dict[int, int] | None = None,
        **metadata: object,
    ) -> dict[str, object]:
        """Return the semantic claim boundary consumed by generation and verification."""
        claims: list[dict[str, object]] = []
        for claim in self.claims:
            item = claim.to_dict()
            if source_index_map is not None:
                item["source_indices"] = list(
                    dict.fromkeys(
                        source_index_map[index]
                        for index in claim.source_indices
                        if index in source_index_map
                    )
                )
                if not item["source_indices"]:
                    continue
            claims.append(item)
        return {
            "action": self.action,
            "missing_facets": list(self.missing_facets),
            "claims": claims,
            "reason": self.reason,
            **metadata,
        }


def _valid_indices(value: object, candidate_count: int) -> tuple[int, ...]:
    values = value if isinstance(value, list) else [value]
    normalized: list[int] = []
    for item in values:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            index = item
        elif isinstance(item, str) and item.strip().isdigit():
            index = int(item.strip())
        else:
            continue
        if 1 <= index <= candidate_count:
            normalized.append(index)
    return tuple(
        dict.fromkeys(normalized)
    )


def parse_evidence_control_decision(
    value: str,
    candidate_count: int,
    *,
    allow_retry: bool,
    allow_expand: bool = False,
) -> EvidenceControlDecision:
    """Parse a compact controller response without inventing a semantic result."""

    normalized = (value or "").strip()
    match = _JSON_OBJECT_RE.search(normalized)
    if not match:
        # Small local models sometimes follow the semantic choice but ignore
        # the JSON envelope. Accept only a complete, unambiguous label; never
        # fish a verdict out of explanatory prose.
        bare_label = re.sub(r"[`*_#\s。.！!]+", "", normalized).lower()
        if bare_label in {"answer", "supported", "可回答", "证据充分"} and candidate_count:
            return EvidenceControlDecision(
                "answer",
                supported_indices=tuple(range(1, candidate_count + 1)),
                reason="evidence_answer_bare_label",
            )
        if bare_label in {"boundary", "explicitboundary", "明确边界", "明确未披露"}:
            if candidate_count == 1:
                return EvidenceControlDecision(
                    "boundary",
                    supported_indices=(1,),
                    reason="evidence_boundary_bare_label",
                )
            return EvidenceControlDecision(
                "unavailable",
                reason="evidence_boundary_bare_label_missing_support",
            )
        if bare_label in {"abstain", "insufficient", "unsupported", "无法回答", "证据不足"}:
            return EvidenceControlDecision("abstain", reason="evidence_abstain_bare_label")
        if bare_label in {"conflict", "conflicting", "冲突", "证据冲突"} and candidate_count >= 2:
            return EvidenceControlDecision(
                "conflict",
                conflicting_indices=tuple(range(1, candidate_count + 1)),
                reason="evidence_conflict_bare_label",
            )
        if bare_label in {"expand", "context", "补上下文", "扩展上下文"} and candidate_count:
            if not allow_expand:
                return EvidenceControlDecision("abstain", reason="evidence_expand_exhausted")
            return EvidenceControlDecision(
                "expand",
                expand_indices=tuple(range(1, candidate_count + 1)),
                reason="evidence_expand_bare_label",
            )
        return EvidenceControlDecision("unavailable", reason="evidence_controller_invalid_json")
    try:
        payload = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return EvidenceControlDecision("unavailable", reason="evidence_controller_invalid_json")
    if not isinstance(payload, dict):
        return EvidenceControlDecision("unavailable", reason="evidence_controller_invalid_payload")

    action = str(payload.get("action") or "").strip().lower()
    action = {
        "supported": "answer",
        "sufficient": "answer",
        "answer_boundary": "boundary",
        "explicit_boundary": "boundary",
        "known_unknown": "boundary",
        "insufficient": "abstain",
        "unsupported": "abstain",
        "conflicting": "conflict",
        "context": "expand",
        "expand_context": "expand",
    }.get(action, action)
    if action not in {"answer", "boundary", "expand", "retry", "partial", "conflict", "abstain"}:
        return EvidenceControlDecision("unavailable", reason="evidence_controller_invalid_action")

    supported = _valid_indices(payload.get("supported_indices"), candidate_count)
    conflicting = _valid_indices(payload.get("conflicting_indices"), candidate_count)
    raw_facets = payload.get("missing_facets")
    facets: tuple[str, ...] = ()
    if isinstance(raw_facets, list):
        facets = tuple(
            dict.fromkeys(
                text[:120]
                for item in raw_facets[:6]
                if (text := str(item).strip())
            )
        )
    retry_query = str(payload.get("retry_query") or "").strip()[:1000]
    expand_indices = _valid_indices(payload.get("expand_indices"), candidate_count)
    expand_direction = str(payload.get("expand_direction") or "both").strip().lower()
    expand_direction = {
        "previous": "before",
        "next": "after",
        "around": "both",
        "前": "before",
        "后": "after",
        "前后": "both",
    }.get(expand_direction, expand_direction)
    if expand_direction not in {"before", "after", "both"}:
        expand_direction = "both"
    reason_code = str(payload.get("reason_code") or "").strip()[:80]
    claims = parse_evidence_claims(payload.get("claims"), candidate_count)
    claim_supported = tuple(
        dict.fromkeys(
            index
            for claim in claims
            if claim.kind != "conflict"
            for index in claim.source_indices
        )
    )
    claim_conflicting = tuple(
        dict.fromkeys(
            index
            for claim in claims
            if claim.kind == "conflict"
            for index in claim.source_indices
        )
    )
    if not supported:
        supported = claim_supported
    if not conflicting:
        conflicting = claim_conflicting

    if action in {"answer", "boundary"} and not supported:
        return EvidenceControlDecision("unavailable", reason="evidence_controller_missing_support")
    if action == "partial" and (not supported or not facets):
        return EvidenceControlDecision("unavailable", reason="evidence_controller_invalid_partial")
    if action == "conflict" and len(set((*supported, *conflicting))) < 2:
        return EvidenceControlDecision("unavailable", reason="evidence_controller_invalid_conflict")
    if action == "retry":
        if not allow_retry:
            if supported and facets:
                return EvidenceControlDecision(
                    "partial",
                    supported_indices=supported,
                    missing_facets=facets,
                    reason="evidence_retry_exhausted_partial",
                )
            return EvidenceControlDecision(
                "abstain",
                missing_facets=facets,
                reason="evidence_retry_exhausted",
            )
        if not retry_query:
            return EvidenceControlDecision("unavailable", reason="evidence_controller_missing_retry_query")
    if action == "expand":
        if not allow_expand:
            if supported and facets:
                return EvidenceControlDecision(
                    "partial",
                    supported_indices=supported,
                    missing_facets=facets,
                    reason="evidence_expand_exhausted_partial",
                )
            return EvidenceControlDecision(
                "abstain",
                missing_facets=facets,
                reason="evidence_expand_exhausted",
            )
        if not expand_indices:
            return EvidenceControlDecision("unavailable", reason="evidence_controller_missing_expand_indices")

    return EvidenceControlDecision(
        action,  # type: ignore[arg-type]
        supported_indices=supported,
        conflicting_indices=conflicting,
        missing_facets=facets,
        retry_query=retry_query,
        expand_indices=expand_indices,
        expand_direction=expand_direction,  # type: ignore[arg-type]
        claims=claims,
        reason=reason_code or f"evidence_{action}",
    )


async def control_evidence(
    query: str,
    retrieval_query: str,
    candidates: list[dict],
    llm_provider: BaseLLMProvider,
    *,
    allow_retry: bool,
    allow_expand: bool = False,
    timeout_seconds: float,
    max_candidates: int,
    max_tokens: int,
    max_candidate_chars: int,
    max_retries: int = 0,
) -> tuple[EvidenceControlDecision, list[dict]]:
    """Judge one ranked candidate pool and return the exact rendered subset."""

    judged_candidates = list(candidates[:max_candidates])
    rendered: list[str] = []
    for index, candidate in enumerate(judged_candidates, start=1):
        rendered.append(
            render_evidence_candidate(
                index,
                candidate,
                max_candidate_chars=max_candidate_chars,
            )
        )

    messages = [
        {"role": "system", "content": _EVIDENCE_CONTROLLER_PROMPT},
        {
            "role": "user",
            "content": (
                f"allow_retry: {'yes' if allow_retry else 'no'}\n"
                f"allow_expand: {'yes' if allow_expand else 'no'}\n"
                f"用户原问题：{query}\n"
                f"当前检索问句：{retrieval_query}\n\n"
                "候选资料：\n"
                + ("\n\n".join(rendered) if rendered else "（没有召回候选资料）")
            ),
        },
    ]
    decision = EvidenceControlDecision("unavailable", reason="evidence_controller_unavailable")
    attempts = max(1, int(max_retries) + 1)
    attempt_messages = messages
    for attempt in range(attempts):
        try:
            response = await asyncio.wait_for(
                llm_provider.chat(
                    attempt_messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    thinking_mode="off",
                ),
                timeout=timeout_seconds,
            )
            decision = parse_evidence_control_decision(
                response,
                len(judged_candidates),
                allow_retry=allow_retry,
                allow_expand=allow_expand,
            )
        except Exception as exc:
            response = ""
            decision = EvidenceControlDecision(
                "unavailable",
                reason=f"evidence_controller_{exc.__class__.__name__.lower()}",
            )
        if decision.action != "unavailable" or attempt + 1 >= attempts:
            break
        attempt_messages = [
            *messages,
            {"role": "assistant", "content": response},
            {
                "role": "user",
                "content": (
                    "上一条输出无法解析。请重新判断，并且只输出系统要求的单个 JSON 对象；"
                    "不要输出解释、Markdown 或其他文本。"
                ),
            },
        ]
    return decision, judged_candidates
