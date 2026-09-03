"""Fast LLM connectivity probes."""

from __future__ import annotations

import asyncio

from app.providers.llm.base import BaseLLMProvider

# A connectivity check may be the first request after the local model has
# been unloaded or restarted.  2.5s caused false failures even though normal
# chat requests were healthy, so leave enough time for a cold model response
# while still bounding a genuinely unavailable endpoint.
LLM_CONNECTIVITY_TIMEOUT_SECONDS = 10.0


async def probe_llm_connectivity(
    llm_provider: BaseLLMProvider,
    *,
    timeout_seconds: float = LLM_CONNECTIVITY_TIMEOUT_SECONDS,
) -> None:
    """Run a short ping against the configured LLM provider.

    The probe intentionally uses a tiny completion and a short timeout so
    health checks can fail fast when the model endpoint is offline.
    """

    try:
        await asyncio.wait_for(
            llm_provider.chat(
                [{"role": "user", "content": "ping"}],
                temperature=0,
                max_tokens=1,
                thinking_mode="off",
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        raise TimeoutError(
            f"LLM connectivity check timed out after {timeout_seconds:.1f}s"
        ) from exc
