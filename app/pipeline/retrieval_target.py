"""Explicit query targets and resolution for immutable KB index profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.rag_scope import get_tenant_rag_collection_name, resolve_embedding_dimension
from app.utils.runtime_errors import IndexUnavailableError


@dataclass(frozen=True, slots=True)
class RetrievalTarget:
    """One physical collection and the embedding provider that can query it.

    A collection must never be queried with a vector from a different
    embedding profile.  ``kb_ids`` is also retained for legacy collections,
    which contain several knowledge bases in one tenant-scoped collection.
    """

    collection_name: str
    embedding_provider: Any
    kb_ids: tuple[str, ...] | None = None
    profile_hash: str = "legacy"
    index_id: str = "legacy"

    def with_kb_ids(self, kb_ids: list[str] | tuple[str, ...] | None) -> "RetrievalTarget":
        return RetrievalTarget(
            collection_name=self.collection_name,
            embedding_provider=self.embedding_provider,
            kb_ids=tuple(kb_ids) if kb_ids else None,
            profile_hash=self.profile_hash,
            index_id=self.index_id,
        )


async def resolve_active_retrieval_targets(
    *,
    settings,
    tenant_id: str,
    tenant_slug: str | None,
    selected_kb_ids: list[str] | None,
    embedding_provider,
    knowledge_base_store,
    index_profile_store,
    profile_embedding_provider_factory: Callable[[str, dict], Any],
) -> list[RetrievalTarget]:
    """Resolve a tenant scope to its active profile collections.

    Every read path must use this resolver.  Once an active immutable index
    exists, its legacy source collection is intentionally no longer queried.
    """
    if selected_kb_ids:
        knowledge_base_ids = list(dict.fromkeys(selected_kb_ids))
    else:
        knowledge_base_ids = [
            item["kb_id"]
            for item in await knowledge_base_store.list_by_tenant(tenant_id)
            if item.get("status") == "active"
        ]

    targets: list[RetrievalTarget] = []
    legacy_kb_ids: list[str] = []
    for kb_id in knowledge_base_ids:
        active_index = await index_profile_store.get_active_index(kb_id)
        if not active_index:
            # A KB that has entered managed-index lifecycle must never fall
            # back to its old compatibility collection while a candidate is
            # building or repairing. That would silently serve stale data
            # after an active collection was lost.
            list_indexes = getattr(index_profile_store, "list_knowledge_base_indexes", None)
            managed_indexes = await list_indexes(kb_id) if callable(list_indexes) else []
            if managed_indexes:
                collection_name = str(managed_indexes[0].get("collection_name") or kb_id)
                raise IndexUnavailableError(collection_name)
            legacy_kb_ids.append(kb_id)
            continue
        profile = await index_profile_store.get_profile(active_index["profile_hash"])
        if not profile:
            raise RuntimeError(f"知识库 {kb_id} 的活动索引画像缺失，无法安全检索")
        targets.append(RetrievalTarget(
            collection_name=active_index["collection_name"],
            embedding_provider=profile_embedding_provider_factory(active_index["profile_hash"], profile),
            kb_ids=(kb_id,),
            profile_hash=active_index["profile_hash"],
            index_id=active_index["index_id"],
        ))

    # A fresh installation may not yet have KB metadata. Keep the historical
    # collection only for that case or for KBs that have no active generation.
    if legacy_kb_ids or not knowledge_base_ids:
        dimension = await resolve_embedding_dimension(embedding_provider)
        targets.append(RetrievalTarget(
            collection_name=get_tenant_rag_collection_name(
                settings.vectorstore.collection_prefix,
                settings.embedding.openai.model_name,
                tenant_slug=tenant_slug,
                embedding_dimension=dimension,
            ),
            embedding_provider=embedding_provider,
            kb_ids=tuple(legacy_kb_ids) if knowledge_base_ids else None,
        ))
    return targets
