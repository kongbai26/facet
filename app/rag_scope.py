"""Helpers for tenant-aware RAG collection naming."""

from __future__ import annotations

import re

from app.store.vector_store import bound_collection_name, get_collection_name

TENANT_COLLECTION_MARKER = "__rag_tenant__"


def _clean_scope_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]", "_", value or "").lower().strip("_")
    return cleaned or "default"


def get_tenant_rag_collection_name(
    prefix: str,
    model_name: str,
    tenant_slug: str | None = None,
    embedding_dimension: int | None = None,
    profile_hash: str | None = None,
    kb_id: str | None = None,
    index_revision: str | None = None,
) -> str:
    """Return the Chroma collection name for a tenant-scoped RAG corpus.

    The legacy default tenant keeps the historical global collection name so
    existing single-tenant data remains readable without migration.
    """
    if not profile_hash:
        if not tenant_slug or tenant_slug == "default":
            return get_collection_name(prefix, model_name, dimension=embedding_dimension)

        clean_model = _clean_scope_part(model_name)
        clean_tenant = _clean_scope_part(tenant_slug)
        if embedding_dimension is None:
            return f"{prefix}_{TENANT_COLLECTION_MARKER}{clean_tenant}__{clean_model}"
        return bound_collection_name(
            f"{prefix}_{TENANT_COLLECTION_MARKER}{clean_tenant}__{clean_model}_d{int(embedding_dimension)}"
        )

    clean_model = _clean_scope_part(model_name)
    clean_tenant = _clean_scope_part(tenant_slug or "default")
    clean_kb = _clean_scope_part(kb_id or "default")
    clean_profile = _clean_scope_part(profile_hash)[:16]
    clean_revision = _clean_scope_part(index_revision)[:16]
    dimension_part = f"_d{int(embedding_dimension)}" if embedding_dimension is not None else ""
    revision_part = f"_r{clean_revision}" if clean_revision else ""
    return bound_collection_name(
        f"{prefix}_{TENANT_COLLECTION_MARKER}{clean_tenant}__{clean_kb}__{clean_model}{dimension_part}_p{clean_profile}{revision_part}"
    )


async def resolve_embedding_dimension(embedding_provider) -> int | None:
    dimension_fn = getattr(embedding_provider, "dimension", None)
    if not callable(dimension_fn):
        return None
    try:
        return await dimension_fn()
    except Exception:
        return None


async def resolve_embedding_profile(embedding_provider) -> dict:
    """读取 provider 运行时能力，兼容只有 dimension() 的旧 provider。"""
    profile_fn = getattr(embedding_provider, "runtime_profile", None)
    if callable(profile_fn):
        try:
            profile = await profile_fn()
            if isinstance(profile, dict):
                return profile
        except Exception:
            pass
    dimension = await resolve_embedding_dimension(embedding_provider)
    return {"dimension": dimension} if dimension is not None else {}


def is_tenant_scoped_rag_collection(user_visible_name: str) -> bool:
    return (user_visible_name or "").startswith(TENANT_COLLECTION_MARKER)
