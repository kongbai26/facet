"""Claim-aware semantic verification for knowledge-grounded answers."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Literal

from app.pipeline.evidence_scope import render_evidence_candidate
from app.pipeline.grounding_contract import render_claim_ledger, render_verifier_contract
from app.providers.llm.base import BaseLLMProvider


VerificationStatus = Literal["pass", "fail", "unavailable"]
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class AnswerVerificationDecision:
    status: VerificationStatus
    unsupported_claims: tuple[str, ...] = ()
    reason: str = ""


def parse_answer_verification(value: str) -> AnswerVerificationDecision:
    normalized = (value or "").strip()
    match = _JSON_OBJECT_RE.search(normalized)
    if not match:
        bare_label = re.sub(r"[`*_#\s。.！!]+", "", normalized).lower()
        if bare_label in {"pass", "supported", "通过", "全部支持"}:
            return AnswerVerificationDecision("pass", reason="answer_verifier_pass_bare_label")
        if bare_label in {"fail", "unsupported", "不通过", "存在不支持"}:
            return AnswerVerificationDecision("fail", reason="answer_verifier_fail_bare_label")
        return AnswerVerificationDecision("unavailable", reason="answer_verifier_invalid_json")
    try:
        payload = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return AnswerVerificationDecision("unavailable", reason="answer_verifier_invalid_json")
    if not isinstance(payload, dict):
        return AnswerVerificationDecision("unavailable", reason="answer_verifier_invalid_payload")
    verdict = str(payload.get("verdict") or "").strip().lower()
    verdict = {
        "supported": "pass",
        "valid": "pass",
        "unsupported": "fail",
        "invalid": "fail",
    }.get(verdict, verdict)
    if verdict not in {"pass", "fail"}:
        return AnswerVerificationDecision("unavailable", reason="answer_verifier_invalid_verdict")
    raw_claims = payload.get("unsupported_claims")
    claims: tuple[str, ...] = ()
    if isinstance(raw_claims, list):
        claims = tuple(
            dict.fromkeys(
                text[:200]
                for item in raw_claims[:5]
                if (text := str(item).strip())
            )
        )
    reason = str(payload.get("reason_code") or "").strip()[:80]
    return AnswerVerificationDecision(
        verdict,  # type: ignore[arg-type]
        unsupported_claims=claims,
        reason=reason or f"answer_verifier_{verdict}",
    )


async def verify_grounded_answer(
    query: str,
    answer: str,
    results: list[dict],
    llm_provider: BaseLLMProvider,
    *,
    response_mode: str,
    timeout_seconds: float,
    max_tokens: int,
    max_candidate_chars: int,
    max_retries: int = 0,
    evidence_guidance: dict[str, object] | None = None,
) -> AnswerVerificationDecision:
    rendered: list[str] = []
    for fallback_index, result in enumerate(results, start=1):
        index = result.get("source_index")
        if not isinstance(index, int) or index <= 0:
            index = fallback_index
        rendered.append(
            render_evidence_candidate(
                index,
                result,
                max_candidate_chars=max_candidate_chars,
            )
        )
    claim_ledger = render_claim_ledger(
        evidence_guidance.get("claims") if evidence_guidance else None
    )
    messages = [
        {
            "role": "system",
            "content": render_verifier_contract(response_mode, has_sources=bool(results)),
        },
        {
            "role": "user",
            "content": (
                f"回答类型：{response_mode}\n"
                f"用户问题：{query}\n\n"
                f"候选答案：\n{answer}\n\n"
                "参考资料：\n"
                + ("\n\n".join(rendered) or "（本轮没有可通过的参考资料）")
                + (f"\n\n{claim_ledger}" if claim_ledger else "")
            ),
        },
    ]
    decision = AnswerVerificationDecision("unavailable", reason="answer_verifier_unavailable")
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
            decision = parse_answer_verification(response)
        except Exception as exc:
            response = ""
            decision = AnswerVerificationDecision(
                "unavailable",
                reason=f"answer_verifier_{exc.__class__.__name__.lower()}",
            )
        if decision.status != "unavailable" or attempt + 1 >= attempts:
            break
        attempt_messages = [
            *messages,
            {"role": "assistant", "content": response},
            {
                "role": "user",
                "content": (
                    "上一条输出无法解析。请重新核验，并且只输出系统要求的单个 JSON 对象；"
                    "不要输出解释、Markdown 或其他文本。"
                ),
            },
        ]
    return decision
