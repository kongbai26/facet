"""Authentication state helpers."""

from __future__ import annotations

from typing import Any

from app.utils.security import hash_password, is_placeholder_secret

LEGACY_RAG_ACCESS = ["rag:read", "rag:write", "llm:invoke"]
SESSION_DEFAULT_SCOPES = ["rag:read", "rag:write", "llm:invoke"]


def browser_session_scopes() -> list[str]:
    return [
        "admin:*",
        "vectors:admin",
        "vectors:write",
        "vectors:read",
        "rag:admin",
        "rag:write",
        "rag:read",
        "llm:invoke",
    ]


def session_default_scopes() -> list[str]:
    return list(SESSION_DEFAULT_SCOPES)


def principal_summary(principal: dict[str, Any] | None) -> dict[str, Any] | None:
    if not principal:
        return None
    return {
        "principal_id": principal["principal_id"],
        "name": principal["name"],
        "principal_type": principal["principal_type"],
        "is_admin": bool(principal.get("is_admin")),
    }


def get_legacy_plaintext_password(settings) -> str:
    for candidate in (
        settings.auth.password,
        settings.auth.admin_password,
    ):
        if candidate and not is_placeholder_secret(candidate):
            return candidate.strip()
    return ""


def has_legacy_plaintext_password(settings) -> bool:
    return bool(get_legacy_plaintext_password(settings))


def session_secret_ok(settings) -> bool:
    return bool(settings.auth.session_secret) and not is_placeholder_secret(settings.auth.session_secret)


def collect_auth_warnings(settings, *, initialized: bool) -> list[str]:
    warnings: list[str] = []
    if has_legacy_plaintext_password(settings):
        warnings.append("检测到遗留明文密码配置，运行时已忽略，请清理 .env 中相关字段。")
    if not session_secret_ok(settings):
        warnings.append("SESSION_SECRET 未正确配置，浏览器会话不可用。")
    if settings.auth.enabled and not initialized:
        warnings.append("管理员尚未初始化，请先在 Web 完成首次设置。")
    if settings.app.env == "production" and settings.auth.enabled and not settings.auth.cookie_secure:
        warnings.append("生产环境当前未启用 Secure Cookie；请在 HTTPS 部署中将 auth.cookie_secure 设为 true。")
    return warnings


async def ensure_admin_credential_from_legacy(settings, principal: dict, credential_store) -> bool:
    existing = await credential_store.get_active_password_credential(principal["principal_id"])
    if existing:
        return False

    password_hash = (settings.auth.password_hash or "").strip()
    if not password_hash:
        legacy_password = get_legacy_plaintext_password(settings)
        if not legacy_password:
            return False
        password_hash = hash_password(legacy_password)

    await credential_store.upsert_password_credential(principal["principal_id"], password_hash)
    return True
