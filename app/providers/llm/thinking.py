"""Model-scoped thinking controls for OpenAI-compatible endpoints.

Thinking is a semantic user preference, while the actual request field is an
endpoint dialect.  Keeping those concerns separate prevents model-name guesses
from injecting a vLLM-only field into Ollama (or the reverse).
"""

from __future__ import annotations

import fnmatch
from typing import Any

from app.settings.settings import LLMThinkingConfig, LLMThinkingProfile

THINKING_MODES = {"auto", "on", "off"}
THINKING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
_ROUTING_PREFIXES = {
    "openai", "xai", "grok", "qwen", "deepseek", "siliconflow", "zhipu",
}


def normalize_thinking_mode(value: object, *, default: str = "auto") -> str:
    fallback = str(default or "auto").strip().lower()
    if fallback not in THINKING_MODES:
        fallback = "auto"
    normalized = str(value or "").strip().lower()
    return normalized if normalized in THINKING_MODES else fallback


def normalize_thinking_effort(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in THINKING_EFFORTS else None


def _canonical_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower()
    if "/" in normalized and normalized.split("/", 1)[0] in _ROUTING_PREFIXES:
        return normalized.split("/", 1)[1]
    return normalized


def resolve_thinking_profile(
    config: LLMThinkingConfig,
    model_name: str,
) -> tuple[str | None, LLMThinkingProfile | None]:
    """Resolve exact/canonical names first, then ordered glob patterns."""
    canonical = _canonical_model_name(model_name)
    exact_candidates = {str(model_name or "").strip().lower(), canonical}
    for pattern, profile in config.profiles.items():
        if str(pattern).strip().lower() in exact_candidates:
            return pattern, profile
    for pattern, profile in config.profiles.items():
        normalized_pattern = str(pattern).strip().lower()
        if any(char in normalized_pattern for char in "*?[") and fnmatch.fnmatchcase(
            canonical, normalized_pattern
        ):
            return pattern, profile
    return None, None


def build_thinking_request_kwargs(
    config: LLMThinkingConfig,
    model_name: str,
    mode: object,
    effort_override: object = None,
) -> dict[str, Any]:
    """Translate a semantic mode into the selected server dialect."""
    resolved_mode = normalize_thinking_mode(mode, default=config.mode)
    _pattern, profile = resolve_thinking_profile(config, model_name)
    if profile is None:
        return {}

    requested_effort = normalize_thinking_effort(effort_override)
    # The profile is the source of truth for native gateway values. A stale
    # conversation or direct API caller must not inject an effort level this
    # model profile never advertised.
    if requested_effort not in set(profile.efforts):
        requested_effort = None
    if requested_effort is not None:
        resolved_mode = "off" if requested_effort == "none" else "on"
    if resolved_mode == "auto":
        return {}

    enabled = resolved_mode == "on"
    effort = (requested_effort or profile.effort) if enabled else "none"
    if profile.transport in {"openai", "ollama", "vllm"}:
        return {"reasoning_effort": effort}
    if profile.transport == "vllm_template":
        return {
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": enabled},
            }
        }
    # Qwen's template-driven dialect uses enable_thinking to turn reasoning
    # off and its native effort levels when reasoning is enabled.
    if not enabled:
        return {
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        }
    native_effort = "xhigh" if effort in {"high", "max"} else effort
    return {
        "extra_body": {
            "chat_template_kwargs": {"reasoning_effort": native_effort},
        }
    }


def describe_thinking_configuration(
    config: LLMThinkingConfig,
    model_name: str,
    mode: object,
) -> dict[str, object]:
    resolved_mode = normalize_thinking_mode(mode, default=config.mode)
    pattern, profile = resolve_thinking_profile(config, model_name)
    return {
        "mode": resolved_mode,
        "supported": profile is not None,
        "transport": profile.transport if profile is not None else None,
        "effort": profile.effort if profile is not None else None,
        "efforts": list(profile.efforts) if profile is not None else [],
        "matched_pattern": pattern,
        "model_name": model_name,
    }
