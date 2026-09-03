"""Durable knowledge-base deletion and physical storage reclamation."""

from __future__ import annotations

from app.pipeline.document_runtime_cleanup import (
    cleanup_legacy_document_runtime,
    delete_collection_if_present,
)
from app.store.parent_chunk_store import ParentChunkStore


async def purge_knowledge_base_runtime(
    *,
    kb: dict,
    settings,
    document_store,
    vector_store,
    index_profile_store,
    conversation_store,
    namespace_store,
    knowledge_base_store,
    bm25_store=None,
    job_store=None,
    delete_job_id: str | None = None,
) -> dict:
    """Reclaim every KB-owned representation, then remove metadata.

    The caller must mark the KB ``deleting`` first.  Active index generations
    are safe to remove only because the status gate hides the KB from all new
    retrieval and source mutation requests before this function runs.
    """
    kb_id = str(kb["kb_id"])
    tenant_id = str(kb["tenant_id"])
    documents = await document_store.list_by_knowledge_base(kb_id)
    deleted_documents = 0
    affected_legacy_collections: set[str] = set()

    for document in documents:
        cleanup = await cleanup_legacy_document_runtime(
            document=document,
            settings=settings,
            vector_store=vector_store,
            bm25_store=bm25_store,
            compact_empty_collection=True,
        )
        if cleanup.errors:
            raise RuntimeError("; ".join(cleanup.errors))
        affected_legacy_collections.add(cleanup.collection_name)
        await document_store.delete(document["doc_id"])
        deleted_documents += 1

    indexes = await index_profile_store.list_knowledge_base_indexes(kb_id)
    deleted_indexes = 0
    for index in indexes:
        await delete_collection_if_present(vector_store, index["collection_name"])
        await ParentChunkStore(settings.storage.metadata_db).delete_profile(index["index_id"])
        if bm25_store is not None:
            bm25_store.invalidate_cache(index["collection_name"])
        await index_profile_store.delete_index_for_knowledge_base(kb_id, index["index_id"])
        deleted_indexes += 1

    await conversation_store.detach_knowledge_base(kb_id, tenant_id)
    await knowledge_base_store.delete(kb_id, tenant_id)
    await namespace_store.delete_if_orphaned(kb["namespace_id"], tenant_id)
    if job_store is not None:
        await job_store.delete_terminal_jobs_for_kb(kb_id, except_job_id=delete_job_id)

    return {
        "kb_id": kb_id,
        "deleted_documents": deleted_documents,
        "deleted_indexes": deleted_indexes,
        "affected_legacy_collections": sorted(affected_legacy_collections),
    }
