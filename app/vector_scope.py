"""Tenant-aware naming helpers for the standalone Vector API."""

from __future__ import annotations

import re

VECTOR_TENANT_COLLECTION_MARKER = "__vector_tenant__"
VECTOR_LOGICAL_NAME_METADATA = "vector_api_logical_name"
VECTOR_TENANT_SLUG_METADATA = "vector_api_tenant_slug"


def _clean_scope_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]", "_", value or "").lower().strip("_")
    return cleaned or "default"


def _is_tenant_vector_collection(user_visible_name: str) -> bool:
    return (user_visible_name or "").startswith(VECTOR_TENANT_COLLECTION_MARKER)


def get_tenant_vector_collection_name(
    prefix: str,
    logical_name: str,
    tenant_slug: str | None = None,
) -> str:
    """Return the physical Chroma collection name for the Vector API."""
    if not tenant_slug or tenant_slug == "default":
        raw_name = f"{prefix}_{logical_name}" if prefix else logical_name
        return _bound_collection_name(raw_name)
    clean_tenant = _clean_scope_part(tenant_slug)
    base = f"{prefix}_" if prefix else ""
    raw_name = f"{base}{VECTOR_TENANT_COLLECTION_MARKER}{clean_tenant}__{logical_name}"
    return _bound_collection_name(raw_name)


def _bound_collection_name(value: str) -> str:
    """Apply Chroma's physical-name limit without changing the API name."""
    # Keep the naming dependency one-way: vector_scope is used by the API,
    # while VectorStore remains the persistence implementation.
    from app.store.vector_store import bound_collection_name

    return bound_collection_name(value)


def parse_tenant_vector_collection_name(user_visible_name: str) -> dict | None:
    if not _is_tenant_vector_collection(user_visible_name):
        return None
    remainder = user_visible_name[len(VECTOR_TENANT_COLLECTION_MARKER):]
    tenant_slug, separator, logical_name = remainder.partition("__")
    if not separator or not tenant_slug or not logical_name:
        return None
    return {
        "tenant_slug": tenant_slug,
        "logical_name": logical_name,
    }


def collection_belongs_to_tenant(user_visible_name: str, tenant_slug: str | None) -> bool:
    if not tenant_slug or tenant_slug == "default":
        return not _is_tenant_vector_collection(user_visible_name)
    parsed = parse_tenant_vector_collection_name(user_visible_name)
    return bool(parsed and parsed["tenant_slug"] == _clean_scope_part(tenant_slug))


def visible_vector_collections(collections: list[dict], tenant_slug: str | None) -> list[dict]:
    result = []
    for item in collections:
        name = item.get("name") or ""
        if not name:
            continue
        metadata = item.get("metadata") or {}
        stored_tenant_slug = metadata.get(VECTOR_TENANT_SLUG_METADATA)
        if stored_tenant_slug is not None:
            if _clean_scope_part(str(stored_tenant_slug)) != _clean_scope_part(tenant_slug or "default"):
                continue
            logical_name = metadata.get(VECTOR_LOGICAL_NAME_METADATA)
            result.append({
                **item,
                "name": str(logical_name or name),
            })
            continue
        if not collection_belongs_to_tenant(name, tenant_slug):
            continue
        if _is_tenant_vector_collection(name):
            parsed = parse_tenant_vector_collection_name(name)
            if not parsed:
                continue
            result.append({**item, "name": parsed["logical_name"]})
        else:
            result.append(item)
    return result
