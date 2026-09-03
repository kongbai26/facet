"""Shared BM25 lifecycle facts used by recovery and diagnostics."""

from __future__ import annotations

from app.rag_scope import get_tenant_rag_collection_name


async def resolve_bm25_target_collections(
    settings,
    document_store,
    index_profile_store,
    *,
    tenant_id: str | None = None,
) -> set[str]:
    """Return the collections that live retrieval can currently address.

    A knowledge base with an active immutable index reads from that generation;
    legacy knowledge bases read from the document's model-scoped collection.
    Keeping this resolution in one place prevents status and startup recovery
    from checking a collection that retrieval no longer uses.
    """
    documents = await document_store.list_all(tenant_id=tenant_id)
    ready_documents = [document for document in documents if document.get("status") == "ready"]
    active_indexes: dict[str, dict | None] = {}
    collections: set[str] = set()

    for document in ready_documents:
        kb_id = str(document.get("kb_id") or "")
        active_index = None
        if kb_id:
            if kb_id not in active_indexes:
                active_indexes[kb_id] = await index_profile_store.get_active_index(kb_id)
            active_index = active_indexes[kb_id]
        if active_index:
            collections.add(str(active_index["collection_name"]))
            continue

        collections.add(get_tenant_rag_collection_name(
            settings.vectorstore.collection_prefix,
            document.get("embedding_model") or settings.embedding.openai.model_name,
            tenant_slug=document.get("tenant_slug"),
            embedding_dimension=document.get("embedding_dimension"),
        ))

    return collections
