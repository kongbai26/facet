"""Prompt profile resolution for local/cloud LLM modes."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from app.providers.llm.registry import resolve_llm_mode

PROMPT_PROFILE_AUTO = "auto"
PROMPT_PROFILE_LOCAL = "local"
PROMPT_PROFILE_CLOUD = "cloud"
PROMPT_PROFILE_VALUES = {
    PROMPT_PROFILE_AUTO,
    PROMPT_PROFILE_LOCAL,
    PROMPT_PROFILE_CLOUD,
}


def normalize_prompt_profile(profile: str | None) -> str:
    value = (profile or PROMPT_PROFILE_AUTO).strip().lower()
    return value if value in PROMPT_PROFILE_VALUES else PROMPT_PROFILE_AUTO


def is_local_llm_endpoint(api_base: str | None) -> bool:
    if not api_base:
        return False

    host = urlparse(api_base).hostname or ""
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local)


def resolve_prompt_profile(settings, configured_profile: str | None = None) -> str:
    profile = normalize_prompt_profile(configured_profile)
    if profile != PROMPT_PROFILE_AUTO:
        return profile

    if resolve_llm_mode(settings.llm) == "mock":
        return PROMPT_PROFILE_LOCAL
    if is_local_llm_endpoint(settings.llm.api_base):
        return PROMPT_PROFILE_LOCAL
    return PROMPT_PROFILE_CLOUD
