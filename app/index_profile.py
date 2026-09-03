"""Stable, verifiable identities for an embedding/indexing configuration."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


# One version describes the complete indexing contract.  The chunking
# strategy is already part of the profile payload, so it must not be encoded
# a second time in the document status column.
INDEX_PIPELINE_VERSION = "index_v3"


def normalize_endpoint(endpoint: str | None) -> str:
    """Return the endpoint form used in profile comparisons and hashes."""
    return (endpoint or "").strip().rstrip("/")


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize a profile deterministically so equivalent configs share an ID."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def profile_hash(profile: Mapping[str, Any]) -> str:
    """Return a compact stable identifier for an immutable index profile."""
    return hashlib.sha256(canonical_json(profile).encode("utf-8")).hexdigest()[:16]


def source_fingerprint(documents: list[Mapping[str, Any]]) -> str:
    """Fingerprint the exact ready-document source snapshot for an index.

    The canonical representation is shared by candidate construction and the
    atomic activation transaction.  ``source_revision`` captures a deliberate
    reingest even when the source bytes themselves did not change.
    """
    records = [
        {
            "doc_id": str(document.get("doc_id") or ""),
            "content_hash": str(document.get("content_hash") or ""),
            "file_size": int(document.get("file_size") or 0),
            "source_revision": int(document.get("source_revision") or 0),
        }
        for document in documents
    ]
    records.sort(key=lambda item: item["doc_id"])
    encoded = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_index_profile(settings, embedding_runtime_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Build the complete semantic identity of an index.

    This intentionally includes every input that can change vectors, chunk
    boundaries, or lexical records.  A change therefore creates a candidate
    index instead of silently mixing incompatible chunks in one collection.
    """
    embedding = settings.embedding.openai
    chunking = settings.chunking
    configured_target = getattr(chunking, "child_target_tokens", None)
    if configured_target is None:
        configured_target = chunking.child_max_tokens
    max_input_tokens = int(
        embedding_runtime_profile.get("context_window") or embedding.max_tokens
    )
    effective_target_tokens = min(max(1, int(configured_target)), max(1, max_input_tokens))
    configured_continuity = (
        chunking.child_continuity_tokens
        if chunking.child_continuity_tokens is not None
        else chunking.child_overlap_tokens
    )
    effective_continuity_tokens = min(max(0, int(configured_continuity)), effective_target_tokens)
    parent_child_enabled = bool(chunking.parent_child_enabled)
    semantic = bool(chunking.semantic)
    strategy = "parent_child" if parent_child_enabled else ("semantic" if semantic else "recursive")

    return {
        "version": 2,
        "pipeline_version": INDEX_PIPELINE_VERSION,
        "embedding": {
            "provider": settings.embedding.provider,
            "endpoint": normalize_endpoint(embedding.api_base),
            "model": embedding_runtime_profile.get("model_name") or embedding.model_name,
            "dimension": embedding_runtime_profile.get("dimension"),
            "max_input_tokens": max_input_tokens,
            "tokenizer": {
                "id": embedding_runtime_profile.get("tokenizer_id") or "unverified",
                "version": embedding_runtime_profile.get("tokenizer_version") or "",
                "verified": bool(embedding_runtime_profile.get("tokenizer_verified")),
            },
        },
        "chunking": {
            "strategy": strategy,
            # This cap is applied to every final embedding payload, including
            # flat chunking, so it always belongs to the identity.
            "target_tokens": effective_target_tokens,
            "continuity_tokens": effective_continuity_tokens if parent_child_enabled else 0,
            "parent_max_tokens": int(chunking.parent_max_tokens) if parent_child_enabled else 0,
            "chunk_size": 0 if parent_child_enabled else int(chunking.chunk_size),
            "chunk_overlap": 0 if parent_child_enabled else int(chunking.chunk_overlap),
            "separators": [] if parent_child_enabled or semantic else list(chunking.separators),
            "overlap_sentences": int(chunking.overlap_sentences) if semantic and not parent_child_enabled else 0,
        },
        "parser_version": 1,
        "lexical_schema_version": 1,
    }
