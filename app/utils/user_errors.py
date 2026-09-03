"""Map backend exceptions to safer user-facing messages."""

from __future__ import annotations

import re

from app.providers.llm.errors import LLMRequestError


_URL_RE = re.compile(r"(?i)\b(?:https?|ftp)://[^\s<>\"'`]+")
_WINDOWS_PATH_RE = re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/]|\\\\)[^\s<>\"'`]+")
_POSIX_PATH_RE = re.compile(
    r"(?<![\w])/(?:Users|home|tmp|var|private|opt|app|mnt|workspace)"
    r"(?:/[^\s<>\"'`]+)+"
)
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|authorization|password)"
    r"(\s*[:=]\s*)[^\s,;]+"
)


def sanitize_user_error_message(message: str | BaseException | None, fallback: str) -> str:
    if isinstance(message, LLMRequestError):
        return message.user_message

    normalized = str(message or "").strip()
    if not normalized:
        return fallback

    lowered = normalized.lower()

    if any(token in lowered for token in ("incorrect api key", "invalid_api_key", "authentication", "unauthorized")):
        return "模型服务认证失败，请检查 API Key 配置后重试。"
    if "rate limit" in lowered or "429" in lowered:
        return "模型服务当前较忙，请稍后重试。"
    if "timeout" in lowered:
        return "模型服务响应超时，请稍后重试。"
    if any(token in lowered for token in ("connection", "temporarily unavailable", "name resolution", "network")):
        return "模型服务连接失败，请检查服务地址和网络状态。"

    return sanitize_diagnostic_detail(normalized)


def sanitize_diagnostic_detail(message: str | None, fallback: str = "") -> str:
    """Remove deployment paths, service URLs, and credentials from diagnostics."""
    normalized = str(message or "").strip()
    if not normalized:
        return fallback

    redacted = _SECRET_RE.sub(r"\1\2[已隐藏]", normalized)
    redacted = _URL_RE.sub("[已隐藏地址]", redacted)
    redacted = _WINDOWS_PATH_RE.sub("[已隐藏路径]", redacted)
    redacted = _POSIX_PATH_RE.sub("[已隐藏路径]", redacted)
    redacted = " ".join(redacted.split())
    if len(redacted) > 240:
        redacted = redacted[:237].rstrip() + "..."
    return redacted or fallback
