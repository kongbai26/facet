"""Stable, safe failures for LLM calls.

Provider SDK exceptions vary between OpenAI, local OpenAI-compatible servers,
and transport libraries.  The application should make retry and UI decisions
from a small vocabulary instead of parsing an arbitrary upstream error at each
call site.
"""

from __future__ import annotations

import asyncio
from enum import Enum

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.utils.runtime_errors import (
    AnswerVerificationFailedError,
    AnswerVerificationUnavailableError,
    ChatTurnDeadlineExceededError,
    ConversationTurnQueueTimeoutError,
    GenerationQueueTimeoutError,
    IndexUnavailableError,
)


class LLMFailureCode(str, Enum):
    """Publicly safe categories for a failed LLM request."""

    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    SERVER = "server"
    CONTENT_FILTERED = "content_filtered"
    REASONING_EXHAUSTED = "reasoning_exhausted"
    EMPTY_RESPONSE = "empty_response"
    UNKNOWN = "unknown"


_USER_MESSAGES = {
    LLMFailureCode.TIMEOUT: "模型服务响应超时，请稍后重试。",
    LLMFailureCode.CONNECTION: "模型服务连接失败，请检查服务地址和网络状态。",
    LLMFailureCode.RATE_LIMIT: "模型服务当前较忙，请稍后重试。",
    LLMFailureCode.AUTHENTICATION: "模型服务认证失败，请检查 API Key 配置后重试。",
    LLMFailureCode.INVALID_REQUEST: "模型服务拒绝了本次请求，请检查模型配置后重试。",
    LLMFailureCode.SERVER: "模型服务暂时不可用，请稍后重试。",
    LLMFailureCode.CONTENT_FILTERED: "模型服务无法处理这段内容，请调整问题后重试。",
    LLMFailureCode.REASONING_EXHAUSTED: "模型在思考阶段耗尽了输出额度，请关闭思考模式或提高输出额度后重试。",
    LLMFailureCode.EMPTY_RESPONSE: "模型服务没有返回可用内容，请稍后重试。",
    LLMFailureCode.UNKNOWN: "生成失败，请稍后重试。",
}


class LLMRequestError(RuntimeError):
    """A typed failure that deliberately does not retain upstream details."""

    def __init__(
        self,
        code: LLMFailureCode,
        *,
        stage: str,
        retryable: bool = False,
        emitted_output: bool = False,
    ) -> None:
        self.code = code
        self.stage = stage
        self.retryable = retryable
        self.emitted_output = emitted_output
        super().__init__(f"llm_failure:{code.value}:{stage}")

    @property
    def user_message(self) -> str:
        return _USER_MESSAGES[self.code]


def classify_llm_exception(exc: BaseException) -> tuple[LLMFailureCode, bool]:
    """Classify known SDK failures without passing their body to the client."""
    if isinstance(exc, LLMRequestError):
        return exc.code, exc.retryable
    if isinstance(exc, (asyncio.TimeoutError, APITimeoutError)):
        return LLMFailureCode.TIMEOUT, True
    if isinstance(exc, APIConnectionError):
        return LLMFailureCode.CONNECTION, True
    if isinstance(exc, RateLimitError):
        return LLMFailureCode.RATE_LIMIT, True
    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        message = str(exc).lower()
        if any(token in message for token in ("content filter", "content_filter", "safety")):
            return LLMFailureCode.CONTENT_FILTERED, False
        if status_code in {401, 403}:
            return LLMFailureCode.AUTHENTICATION, False
        if status_code == 429:
            return LLMFailureCode.RATE_LIMIT, True
        if status_code in {400, 404, 409, 413, 422}:
            return LLMFailureCode.INVALID_REQUEST, False
        if isinstance(status_code, int) and status_code >= 500:
            return LLMFailureCode.SERVER, True

    message = str(exc).lower()
    if any(token in message for token in ("content filter", "content_filter", "safety")):
        return LLMFailureCode.CONTENT_FILTERED, False
    if "模型未返回" in str(exc) or "empty response" in message:
        return LLMFailureCode.EMPTY_RESPONSE, False
    if "timeout" in message or "timed out" in message:
        return LLMFailureCode.TIMEOUT, True
    if any(token in message for token in ("connection", "network", "name resolution", "terminated")):
        return LLMFailureCode.CONNECTION, True
    return LLMFailureCode.UNKNOWN, False


def as_llm_request_error(
    exc: BaseException,
    *,
    stage: str,
    emitted_output: bool = False,
) -> LLMRequestError:
    """Normalize an error once at the provider/pipeline boundary."""
    if isinstance(exc, LLMRequestError):
        return exc
    code, retryable = classify_llm_exception(exc)
    return LLMRequestError(
        code,
        stage=stage,
        retryable=retryable,
        emitted_output=emitted_output,
    )


def failure_event_payload(exc: BaseException, *, fallback: str) -> dict[str, str | bool]:
    """Build an additive SSE error payload.  ``error`` remains compatible."""
    if isinstance(exc, ConversationTurnQueueTimeoutError):
        return {
            "error": "当前会话仍在生成，请等待完成后再试。",
            "code": "conversation_busy",
            "stage": "conversation_queue",
            "retryable": True,
            "partial_output": False,
        }
    if isinstance(exc, GenerationQueueTimeoutError):
        return {
            "error": "模型服务当前较忙，请稍后重试。",
            "code": "queue_timeout",
            "stage": "queue",
            "retryable": True,
            "partial_output": False,
        }
    if isinstance(exc, ChatTurnDeadlineExceededError):
        return {
            "error": "本次生成超过等待上限，请缩短问题或稍后重试。",
            "code": "turn_timeout",
            "stage": "turn",
            "retryable": True,
            "partial_output": False,
        }
    if isinstance(exc, IndexUnavailableError):
        return {
            "error": str(exc),
            "code": "index_unavailable",
            "stage": "retrieval",
            "retryable": True,
            "partial_output": False,
        }
    if isinstance(exc, AnswerVerificationUnavailableError):
        return {
            "error": "本轮答案的证据校验未完成，未输出未经验证的内容，请稍后重试。",
            "code": "verification_unavailable",
            "stage": "answer_verification",
            "retryable": True,
            "partial_output": False,
        }
    if isinstance(exc, AnswerVerificationFailedError):
        return {
            "error": "候选答案未通过证据核验，未输出不可靠内容；请调整问题或使用增强模式重试。",
            "code": "verification_failed",
            "stage": "answer_verification",
            "retryable": True,
            "partial_output": False,
        }
    failure = as_llm_request_error(exc, stage="generation")
    return {
        # Keep endpoint-specific fallback wording for unexpected errors, while
        # known categories retain their actionable user-facing explanation.
        "error": fallback if failure.code is LLMFailureCode.UNKNOWN else failure.user_message,
        "code": failure.code.value,
        "stage": failure.stage,
        "retryable": failure.retryable,
        "partial_output": failure.emitted_output,
    }
