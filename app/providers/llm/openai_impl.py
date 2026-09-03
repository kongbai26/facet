"""OpenAI 兼容 LLM 实现"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import AsyncGenerator, List

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.providers.llm.base import BaseLLMProvider
from app.providers.llm.errors import LLMFailureCode, LLMRequestError, as_llm_request_error
from app.providers.llm.thinking import build_thinking_request_kwargs, normalize_thinking_mode
from app.settings.settings import LLMConfig

logger = logging.getLogger(__name__)

def _is_retryable_status_error(exc: APIStatusError) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code == 429 or (isinstance(status_code, int) and status_code >= 500)


def _is_retryable_error(exc: Exception) -> bool:
    return isinstance(exc, (APIConnectionError, APITimeoutError, TimeoutError)) or (
        isinstance(exc, APIStatusError) and _is_retryable_status_error(exc)
    )


async def _close_stream_quietly(stream: object) -> None:
    """Close an SDK stream deterministically without masking its real error."""
    for method_name in ("aclose", "close"):
        closer = getattr(stream, method_name, None)
        if not callable(closer):
            continue
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("failed to close interrupted LLM stream", exc_info=True)
        return


class OpenAILLMProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key or "not-needed",
            base_url=config.api_base or None,
            timeout=config.request_timeout,
            # Retry policy is implemented by _retry_with_backoff below.
            # Disable the SDK's hidden retries so request limits remain real.
            max_retries=0,
        )
        self._runtime_thinking_mode = normalize_thinking_mode(
            config.thinking.mode,
            default="off",
        )

    def set_runtime_thinking_mode(self, mode: object) -> str:
        """Update the global preference used by subsequent requests."""
        self._runtime_thinking_mode = normalize_thinking_mode(
            mode,
            default=self.config.thinking.mode,
        )
        return self._runtime_thinking_mode

    def _thinking_request_kwargs(self, kwargs: dict) -> dict:
        mode = kwargs.get("thinking_mode", self._runtime_thinking_mode)
        return build_thinking_request_kwargs(
            self.config.thinking,
            str(kwargs.get("model_name") or self.config.model_name),
            mode,
            kwargs.get("thinking_effort"),
        )

    def _attempt_timeouts(self) -> tuple[float, ...]:
        """Return the configured timeout target for each individual attempt."""
        return tuple(self.config.request_timeout * ratio for ratio in self.config.attempt_timeout_ratios)

    def _retry_attempts(self):
        """Yield retry attempts without exceeding one logical-request budget.

        ``request_timeout`` used to be applied independently to every retry.
        A configuration of 30 seconds could therefore wait 15 + 24 + 30
        seconds (plus backoff) for one answer.  Use it as the total budget for
        the logical request; retries consume what remains.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(self.config.request_timeout)
        max_attempts = max(1, int(self.config.max_attempts))
        for attempt, configured_timeout in enumerate(self._attempt_timeouts()[:max_attempts]):
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            yield attempt, max(0.1, min(float(configured_timeout), remaining)), deadline

    async def _retry_with_backoff(self, request, *, stage: str):
        """Run a request with explicit retries inside one timeout budget."""
        attempts = list(self._retry_attempts())
        if not attempts:
            raise LLMRequestError(LLMFailureCode.TIMEOUT, stage=stage, retryable=True)
        base_delay = max(0.0, float(self.config.retry_backoff_seconds))
        for position, (attempt, timeout_seconds, deadline) in enumerate(attempts):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise LLMRequestError(LLMFailureCode.TIMEOUT, stage=stage, retryable=True)
            timeout_seconds = min(timeout_seconds, remaining)
            failure: Exception | None = None
            try:
                # The SDK timeout is forwarded to httpx, but httpx applies it
                # to individual I/O phases.  ``wait_for`` enforces the actual
                # per-attempt wall-clock budget promised by this provider.
                return await asyncio.wait_for(request(timeout_seconds), timeout=timeout_seconds)
            except (APIStatusError, APIConnectionError, APITimeoutError, TimeoutError) as exc:
                failure = exc
                if not _is_retryable_error(exc) or position == len(attempts) - 1:
                    raise as_llm_request_error(exc, stage=stage) from exc

            delay = min(base_delay * (2 ** attempt), max(0.0, deadline - asyncio.get_running_loop().time()))
            if delay <= 0:
                assert failure is not None
                raise as_llm_request_error(failure, stage=stage) from failure
            assert failure is not None
            logger.warning(
                "LLM request failed with %s; retrying in %.1fs (attempt %d/%d, next timeout %.1fs)",
                type(failure).__name__,
                delay,
                attempt + 1,
                len(attempts),
                attempts[position + 1][1],
            )
            await asyncio.sleep(delay)

    async def chat(self, messages: List[dict], **kwargs) -> str:
        model_name = str(kwargs.get("model_name") or self.config.model_name)
        thinking_kwargs = self._thinking_request_kwargs(kwargs)
        resp = await self._retry_with_backoff(
            lambda timeout: self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                timeout=timeout,
                **thinking_kwargs,
            ),
            stage="completion",
        )
        message = resp.choices[0].message
        # ``reasoning_content`` is hidden chain-of-thought metadata on some
        # OpenAI-compatible servers. It is never a user-facing answer, even
        # when a server omits ``content``.
        answer = _coalesce_message_text(message, allow_reasoning=False)
        if not answer.strip():
            failure_code = (
                LLMFailureCode.REASONING_EXHAUSTED
                if _has_reasoning_content(message)
                else LLMFailureCode.EMPTY_RESPONSE
            )
            raise LLMRequestError(failure_code, stage="completion")
        return answer

    async def chat_stream(self, messages: List[dict], **kwargs) -> AsyncGenerator[str, None]:
        """Stream text with progress-based limits and pre-output retries.

        A single ``request_timeout`` is appropriate for a one-shot completion,
        but it is the wrong health signal for a stream: a local model can take
        longer than that to produce a useful long answer while continuously
        sending chunks.  Streaming therefore has three independent contracts:
        time to first provider event, time between subsequent events, and an
        explicit maximum stream duration.  Visible output is never replayed.
        """
        loop = asyncio.get_running_loop()
        first_token_timeout = max(0.1, float(self.config.stream_first_token_timeout_seconds))
        idle_timeout = max(0.1, float(self.config.stream_idle_timeout_seconds))
        total_timeout = max(0.1, float(self.config.stream_total_timeout_seconds))
        deadline = loop.time() + total_timeout
        max_attempts = max(1, int(self.config.max_attempts))
        base_delay = max(0.0, float(self.config.retry_backoff_seconds))
        emitted = False
        model_name = str(kwargs.get("model_name") or self.config.model_name)
        thinking_kwargs = self._thinking_request_kwargs(kwargs)

        for attempt in range(max_attempts):
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise LLMRequestError(
                    LLMFailureCode.TIMEOUT,
                    stage="stream",
                    retryable=True,
                    emitted_output=emitted,
                )

            stream = None
            saw_provider_event = False
            saw_reasoning = False
            try:
                request_timeout = min(first_token_timeout, remaining)
                stream = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=kwargs.get("temperature", self.config.temperature),
                        max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                        stream=True,
                        # The SDK keeps its own read timeout for the stream.
                        # Use the progress budget here rather than the
                        # non-stream logical request budget.
                        timeout=max(first_token_timeout, idle_timeout),
                        **thinking_kwargs,
                    ),
                    timeout=request_timeout,
                )
                iterator = stream.__aiter__()
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise TimeoutError("LLM stream exceeded the configured duration")
                    progress_timeout = idle_timeout if saw_provider_event else first_token_timeout
                    try:
                        chunk = await asyncio.wait_for(
                            iterator.__anext__(),
                            timeout=min(progress_timeout, remaining),
                        )
                    except StopAsyncIteration:
                        if not emitted:
                            failure_code = (
                                LLMFailureCode.REASONING_EXHAUSTED
                                if saw_reasoning
                                else LLMFailureCode.EMPTY_RESPONSE
                            )
                            raise LLMRequestError(failure_code, stage="stream")
                        return

                    # A reasoning delta or a provider heartbeat proves the
                    # request is alive even though hidden reasoning is never
                    # presented as answer text.
                    saw_provider_event = True
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta
                    saw_reasoning = saw_reasoning or _has_reasoning_content(delta)
                    text = _coalesce_message_text(delta, allow_reasoning=False)
                    if text:
                        emitted = True
                        yield text
            except (APIStatusError, APIConnectionError, APITimeoutError, TimeoutError) as failure:
                if emitted or not _is_retryable_error(failure) or attempt == max_attempts - 1:
                    raise as_llm_request_error(
                        failure,
                        stage="stream",
                        emitted_output=emitted,
                    ) from failure
                delay = min(
                    base_delay * (2 ** attempt),
                    max(0.0, deadline - loop.time()),
                )
                if delay <= 0:
                    raise as_llm_request_error(
                        failure,
                        stage="stream",
                        emitted_output=emitted,
                    ) from failure
                logger.warning(
                    "LLM stream failed before visible output with %s; retrying in %.1fs (attempt %d/%d)",
                    type(failure).__name__, delay, attempt + 1, max_attempts,
                )
                await asyncio.sleep(delay)
            finally:
                if stream is not None:
                    await _close_stream_quietly(stream)


def _coalesce_message_text(message, *, allow_reasoning: bool = False) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return content
    if allow_reasoning:
        reasoning = getattr(message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning:
            return reasoning
    return ""


def _has_reasoning_content(message) -> bool:
    for field in ("reasoning_content", "reasoning"):
        value = getattr(message, field, None)
        if isinstance(value, str) and value.strip():
            return True
    return False
