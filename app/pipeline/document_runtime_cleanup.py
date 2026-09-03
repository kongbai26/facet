"""Shared physical cleanup for a document's legacy runtime representation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.pipeline.reindex import document_upload_directory
from app.rag_scope import get_tenant_rag_collection_name
from app.store.parent_chunk_store import ParentChunkStore
from app.utils.file_ops import remove_dir_strict

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LegacyRuntimeCleanupResult:
    """Result of removing the non-profile representation of one document."""

    collection_name: str
    errors: tuple[str, ...] = ()


async def delete_collection_if_present(vector_store, collection_name: str) -> None:
    """Remove an empty disposable collection without creating it when absent."""
    delete_collection = getattr(vector_store, "delete_collection_by_name", None)
    if not callable(delete_collection):
        return
    try:
        await delete_collection(collection_name, collection_name=collection_name)
    except Exception as exc:
        message = str(exc).lower()
        if "not exist" not in message and "not found" not in message:
            raise


async def cleanup_legacy_document_runtime(
    *,
    document: dict,
    settings,
    vector_store,
    bm25_store=None,
    compact_empty_collection: bool = False,
) -> LegacyRuntimeCleanupResult:
    """Remove vectors, legacy parent chunks and source files for one document.

    Profile collections are deliberately untouched: their immutable lifecycle
    is managed by the KB index coordinator.  Every direct deletion, recovery
    and KB purge uses this result so that partial failures are handled by the
    caller's transaction/state transition rather than silently diverging.
    """
    doc_id = str(document["doc_id"])
    collection_name = get_tenant_rag_collection_name(
        settings.vectorstore.collection_prefix,
        document.get("embedding_model") or settings.embedding.openai.model_name,
        tenant_slug=document.get("tenant_slug"),
        embedding_dimension=document.get("embedding_dimension"),
    )
    errors: list[str] = []

    try:
        safe_delete = getattr(vector_store, "delete_by_doc_id_if_exists", None)
        if callable(safe_delete):
            await safe_delete(doc_id, collection_name=collection_name)
        else:
            await vector_store.delete_by_doc_id(doc_id, collection_name=collection_name)
    except Exception as exc:
        errors.append(f"向量删除失败: {exc}")

    try:
        await ParentChunkStore(settings.storage.metadata_db).delete_document(
            doc_id,
            profile_hash="legacy",
        )
    except Exception as exc:
        errors.append(f"父块删除失败: {exc}")

    try:
        source_dir = document_upload_directory(settings.storage.upload_dir, {"doc_id": doc_id})
        if source_dir is not None:
            remove_dir_strict(source_dir)
    except Exception as exc:
        errors.append(f"文件删除失败: {exc}")

    if bm25_store is not None:
        try:
            bm25_store.invalidate_cache(collection_name)
        except Exception as exc:
            errors.append(f"BM25 缓存失效失败: {exc}")

    if compact_empty_collection and not errors:
        get_collection_info = getattr(vector_store, "get_collection_info", None)
        if callable(get_collection_info):
            try:
                info = await get_collection_info(collection_name=collection_name)
                if int(info.get("count") or 0) == 0:
                    await delete_collection_if_present(vector_store, collection_name)
            except Exception:
                # The document is already logically deleted.  Compaction is
                # retryable and must not turn that successful transition into
                # a failed deletion.
                logger.warning("legacy collection compaction deferred: %s", collection_name, exc_info=True)

    return LegacyRuntimeCleanupResult(collection_name, tuple(errors))
