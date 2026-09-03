"""Business-level authorization helpers built on top of auth primitives."""

from __future__ import annotations

from fastapi import Depends

from app.api.deps import enforce_admin_access, enforce_scopes, verify_auth


def require_rag_read(identity: dict = Depends(verify_auth)) -> dict:
    return enforce_scopes(identity, "rag:read")


def require_rag_write(identity: dict = Depends(verify_auth)) -> dict:
    return enforce_scopes(identity, "rag:write")


def require_llm_invoke(identity: dict = Depends(verify_auth)) -> dict:
    return enforce_scopes(identity, "llm:invoke")


def require_admin_access(identity: dict = Depends(verify_auth)) -> dict:
    return enforce_admin_access(identity)


def enforce_chat_access(identity: dict, *, has_message: bool) -> dict:
    if has_message:
        return enforce_scopes(identity, "rag:read", "rag:write")
    return enforce_scopes(identity, "rag:read")
