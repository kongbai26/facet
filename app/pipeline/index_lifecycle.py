"""Safe construction and validation of knowledge-base candidate indexes."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging
import asyncio

from app.index_profile import build_index_profile, source_fingerprint
from app.pipeline.ingest import IngestDestination, ingest_document
from app.pipeline.document_runtime_cleanup import (
    cleanup_legacy_document_runtime,
    delete_collection_if_present,
)
from app.pipeline.reindex import find_document_file
from app.rag_scope import get_tenant_rag_collection_name, resolve_embedding_profile
from app.store.parent_chunk_store import ParentChunkStore

logger = logging.getLogger(__name__)


async def _cleanup_failed_candidate_representation(
    *,
    collection_name: str,
    index_id: str,
    vector_store,
    metadata_db: str,
    bm25_store=None,
) -> None:
    """Discard partial candidate data while retaining only its failure record."""
    try:
        await delete_collection_if_present(vector_store, collection_name)
    finally:
        await ParentChunkStore(metadata_db).delete_profile(index_id)
        if bm25_store is not None:
            bm25_store.invalidate_cache(collection_name)


def knowledge_base_source_fingerprint(documents: list[dict]) -> str:
    """Fingerprint the exact ready-document set represented by a candidate."""
    return source_fingerprint(documents)


async def build_knowledge_base_candidate(
    *,
    kb_id: str,
    tenant_slug: str,
    settings,
    embedding_provider,
    vector_store,
    document_store,
    index_profile_store,
    bm25_store=None,
) -> dict:
    """Build a complete, non-active representation of one knowledge base.

    Source documents keep their current ``ready`` state throughout.  The
    function intentionally does *not* activate the candidate: query routing
    must be switched only after this build and offline retrieval evaluation
    have both passed.
    """
    runtime_profile = await resolve_embedding_profile(embedding_provider)
    profile = build_index_profile(settings, runtime_profile)
    digest = await index_profile_store.ensure_profile(profile)
    documents = await document_store.list_by_knowledge_base(kb_id, status="ready")
    source_fingerprint = knowledge_base_source_fingerprint(documents)
    index_id = f"{digest}-{source_fingerprint[:16]}"
    dimension = profile["embedding"].get("dimension")
    model_name = str(profile["embedding"].get("model") or settings.embedding.openai.model_name)
    collection_name = get_tenant_rag_collection_name(
        settings.vectorstore.collection_prefix,
        model_name,
        tenant_slug=tenant_slug,
        embedding_dimension=dimension,
        profile_hash=digest,
        kb_id=kb_id,
        index_revision=source_fingerprint,
    )
    existing = await index_profile_store.ensure_knowledge_base_index(
        kb_id,
        digest,
        collection_name,
        source_fingerprint=source_fingerprint,
        status="building",
    )
    # The same profile and source revision is immutable. Reuse a complete
    # generation only while its physical collection still exists. A metadata
    # row that points to a deleted Chroma collection is a recoverable outage,
    # not a valid candidate; mark it failed and rebuild the now-absent
    # representation from the original documents.
    if existing["status"] in {"ready", "active"}:
        collection_exists = getattr(vector_store, "collection_exists", None)
        if not callable(collection_exists) or await collection_exists(
            collection_name=collection_name,
        ):
            return existing
        logger.error(
            "active/candidate index collection is missing; rebuilding: kb_id=%s index_id=%s collection=%s",
            kb_id,
            existing["index_id"],
            collection_name,
        )
        await index_profile_store.mark_index(
            kb_id,
            existing["index_id"],
            "failed",
            error_message="活动索引的物理 collection 缺失，正在自动重建。",
        )

    await index_profile_store.mark_index(kb_id, index_id, "building", error_message="")
    # A failed retry may have left a partial collection.  It is safe to remove
    # because this generation has never been made active.
    await delete_collection_if_present(vector_store, collection_name)
    await ParentChunkStore(settings.storage.metadata_db).delete_profile(index_id)
    total_chunks = 0
    try:
        for document in documents:
            source = find_document_file(settings.storage.upload_dir, document)
            if not source:
                message = "上传原文件缺失，无法构建候选索引。"
                await index_profile_store.upsert_document_state(
                    document["doc_id"], index_id, "failed", error_message=message,
                )
                raise RuntimeError(f"{document['doc_id']}: {message}")

            chunks = await ingest_document(
                Path(source),
                document["doc_id"],
                settings,
                embedding_provider,
                vector_store,
                document_store,
                bm25_store,
                defer_bm25_rebuild=True,
                destination=IngestDestination.candidate(collection_name, index_id),
                index_profile_store=index_profile_store,
            )
            vector_count = await vector_store.count_by_doc_id(
                document["doc_id"], collection_name=collection_name, dimension=dimension,
            )
            if vector_count != chunks:
                raise RuntimeError(
                    f"{document['doc_id']}: candidate vector count={vector_count}, expected={chunks}"
                )
            total_chunks += chunks

        if bm25_store and settings.retrieval.hybrid.enabled:
            await bm25_store.rebuild_from_vector_store(
                vector_store,
                collection_name=collection_name,
                document_store=document_store,
            )
        return await index_profile_store.mark_index(
            kb_id,
            index_id,
            "ready",
            chunk_count=total_chunks,
            error_message="",
        )
    except Exception as exc:
        try:
            await _cleanup_failed_candidate_representation(
                collection_name=collection_name,
                index_id=index_id,
                vector_store=vector_store,
                metadata_db=settings.storage.metadata_db,
                bm25_store=bm25_store,
            )
        except Exception:
            logger.exception("failed candidate physical cleanup: index_id=%s", index_id)
        await index_profile_store.mark_index(
            kb_id,
            index_id,
            "failed",
            error_message=str(exc),
        )
        raise


async def activate_knowledge_base_index(
    *,
    kb_id: str,
    index_id: str,
    document_store,
    index_profile_store,
) -> dict:
    """Atomically cut over only when the candidate still represents sources.

    This turns a source change during a long candidate build into a clear
    retry requirement instead of an apparently-successful partial rollout.
    """
    candidate = await index_profile_store.get_knowledge_base_index(kb_id, index_id)
    if not candidate:
        raise KeyError(f"index not found: {kb_id}/{index_id}")

    # The production store performs the source read and pointer cutover in a
    # single BEGIN IMMEDIATE transaction.  Keep the fallback for lightweight
    # test doubles and older integrations, but never use a non-atomic path for
    # the real SQLite store.
    activate_if_current = getattr(index_profile_store, "activate_index_if_source_current", None)
    same_metadata_db = (
        getattr(document_store, "db_path", None)
        and getattr(index_profile_store, "db_path", None)
        and document_store.db_path == index_profile_store.db_path
    )
    if callable(activate_if_current) and same_metadata_db:
        return await activate_if_current(kb_id, index_id)

    documents = await document_store.list_by_knowledge_base(kb_id, status="ready")
    current_fingerprint = knowledge_base_source_fingerprint(documents)
    if candidate["source_fingerprint"] != current_fingerprint:
        raise ValueError("候选索引构建期间文档已变化，请重新构建后再切换")
    return await index_profile_store.activate_index(kb_id, index_id)


async def cleanup_legacy_source_representations(
    *,
    kb_id: str,
    settings,
    vector_store,
    document_store,
    bm25_store=None,
) -> list[str]:
    """Remove compatibility-only source vectors after a verified cutover.

    Profile-routed retrieval never reads the legacy collection once a KB has
    an active generation.  Keeping it would permanently store every source
    chunk and parent twice.  This function deliberately touches *only* the
    legacy representation; profile collections remain immutable.
    """
    documents = await document_store.list_by_knowledge_base(kb_id, status="ready")
    collections: dict[str, list[dict]] = {}
    for document in documents:
        collection_name = get_tenant_rag_collection_name(
            settings.vectorstore.collection_prefix,
            document.get("embedding_model") or settings.embedding.openai.model_name,
            tenant_slug=document.get("tenant_slug"),
            embedding_dimension=document.get("embedding_dimension"),
        )
        collections.setdefault(collection_name, []).append(document)

    cleaned: list[str] = []
    safe_delete = getattr(vector_store, "delete_by_doc_id_if_exists", None)
    for collection_name, collection_documents in collections.items():
        legacy_collection_existed = False
        for document in collection_documents:
            if callable(safe_delete):
                deleted_from_existing = await safe_delete(
                    document["doc_id"], collection_name=collection_name,
                )
                legacy_collection_existed = bool(deleted_from_existing) or legacy_collection_existed
            else:
                # Compatibility with lightweight test doubles.  Production
                # VectorStore always exposes the non-creating variant above.
                await vector_store.delete_by_doc_id(document["doc_id"], collection_name=collection_name)
                legacy_collection_existed = True
            await ParentChunkStore(settings.storage.metadata_db).delete_document(
                document["doc_id"], profile_hash="legacy"
            )
        if bm25_store is not None:
            bm25_store.invalidate_cache(collection_name)
        get_collection_info = getattr(vector_store, "get_collection_info", None)
        if callable(get_collection_info) and legacy_collection_existed:
            try:
                info = await get_collection_info(collection_name=collection_name)
                if int(info.get("count") or 0) == 0:
                    await delete_collection_if_present(vector_store, collection_name)
            except Exception:
                # The source representation is already logically gone.  A
                # failed physical compaction must not roll back a live cutover.
                logger.warning("legacy collection cleanup deferred: %s", collection_name)
        cleaned.append(collection_name)
    return cleaned


async def finalize_knowledge_base_index_activation(
    *,
    kb_id: str,
    index_id: str,
    settings,
    vector_store,
    document_store,
    index_profile_store,
    bm25_store=None,
) -> dict:
    """The one cutover path: activate, drop legacy copies, then reclaim safely."""
    activated = await activate_knowledge_base_index(
        kb_id=kb_id,
        index_id=index_id,
        document_store=document_store,
        index_profile_store=index_profile_store,
    )
    try:
        await cleanup_legacy_source_representations(
            kb_id=kb_id,
            settings=settings,
            vector_store=vector_store,
            document_store=document_store,
            bm25_store=bm25_store,
        )
        await reclaim_stale_ready_knowledge_base_indexes(
            document_store=document_store,
            vector_store=vector_store,
            index_profile_store=index_profile_store,
            bm25_store=bm25_store,
            kb_ids=[kb_id],
        )
        await reclaim_excess_inactive_knowledge_base_indexes(
            kb_id=kb_id,
            retain_ready_generations=getattr(settings.app, "index_generation_retention", 1),
            vector_store=vector_store,
            index_profile_store=index_profile_store,
            bm25_store=bm25_store,
        )
        await reclaim_expired_inactive_knowledge_base_indexes(
            retention_days=getattr(settings.app, "index_generation_retention_days", 7),
            vector_store=vector_store,
            index_profile_store=index_profile_store,
            bm25_store=bm25_store,
        )
        await reclaim_unreferenced_vector_storage(
            settings=settings,
            vector_store=vector_store,
            document_store=document_store,
            index_profile_store=index_profile_store,
        )
    except Exception:
        # Activation has already committed.  Retention is retryable at the
        # next cutover or startup and must not make a successful index job look
        # failed.
        logger.exception("post-cutover cleanup deferred: kb_id=%s index_id=%s", kb_id, index_id)
    return activated


async def reconcile_knowledge_base_index(
    *,
    kb_id: str,
    tenant_slug: str,
    settings,
    embedding_provider,
    vector_store,
    document_store,
    index_profile_store,
    bm25_store=None,
    auto_activate: bool,
    max_snapshot_attempts: int = 3,
) -> dict:
    """The only normal create/rebuild path for a KB index generation.

    Uploads, model or chunking changes, startup repair and explicit rebuilds
    all differ only in *why* a source snapshot changed.  They must therefore
    use the same build/validate/cutover implementation.
    """
    last_snapshot_error: ValueError | None = None
    for _attempt in range(max(1, int(max_snapshot_attempts))):
        candidate = await build_knowledge_base_candidate(
            kb_id=kb_id,
            tenant_slug=tenant_slug,
            settings=settings,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            document_store=document_store,
            index_profile_store=index_profile_store,
            bm25_store=bm25_store,
        )
        if not auto_activate:
            return candidate
        try:
            return await finalize_knowledge_base_index_activation(
                kb_id=kb_id,
                index_id=candidate["index_id"],
                settings=settings,
                vector_store=vector_store,
                document_store=document_store,
                index_profile_store=index_profile_store,
                bm25_store=bm25_store,
            )
        except ValueError as exc:
            if "文档已变化" not in str(exc):
                raise
            last_snapshot_error = exc
            logger.info("candidate snapshot changed during build; retrying: kb_id=%s", kb_id)
    assert last_snapshot_error is not None
    raise last_snapshot_error


async def finalize_pending_document_removals_after_cutover(
    *,
    kb_id: str,
    settings,
    vector_store,
    document_store,
    index_profile_store,
    bm25_store=None,
) -> list[str]:
    """Physically remove deleting documents only after a KB cutover is live.

    This is called by the sole candidate-job worker after it has activated a
    representation built from the current ``ready`` source set.  It makes a
    deletion durable without ever subtracting one document from an active
    collection.
    """
    documents = await document_store.list_by_knowledge_base(kb_id)
    deleted_doc_ids: list[str] = []
    failed_doc_ids: list[str] = []
    for document in documents:
        if document.get("status") not in {"deleting", "delete_failed"}:
            continue
        doc_id = document["doc_id"]
        try:
            representations = await index_profile_store.list_document_collections(doc_id)
            for representation in representations:
                if representation.get("index_status") == "active":
                    raise RuntimeError("活动索引仍包含待删除文档")
                await reclaim_inactive_knowledge_base_index(
                    kb_id=representation["kb_id"],
                    index_id=representation["index_id"],
                    vector_store=vector_store,
                    index_profile_store=index_profile_store,
                    bm25_store=bm25_store,
                )

            cleanup = await cleanup_legacy_document_runtime(
                document=document,
                settings=settings,
                vector_store=vector_store,
                bm25_store=bm25_store,
                compact_empty_collection=True,
            )
            if cleanup.errors:
                raise RuntimeError("; ".join(cleanup.errors))
            await index_profile_store.retire_document_states(doc_id)
            await document_store.delete(doc_id)
            deleted_doc_ids.append(doc_id)
        except Exception as exc:
            await document_store.update_status_if(
                doc_id,
                ["deleting", "delete_failed"],
                "delete_failed",
                error_message=f"索引切换后的物理删除失败: {exc}",
            )
            logger.exception("post-cutover document deletion failed: doc_id=%s", doc_id)
            failed_doc_ids.append(doc_id)
    if failed_doc_ids:
        raise RuntimeError(f"文档索引切换后的物理删除失败: {failed_doc_ids}")
    return deleted_doc_ids


async def schedule_profile_rebuilds_after_configuration_change(
    *,
    settings,
    embedding_provider,
    document_store,
    index_profile_store,
    ingest_job_store,
) -> list[dict]:
    """Queue safe automatic rebuilds when the configured index profile changes."""
    runtime_profile = await resolve_embedding_profile(embedding_provider)
    desired_hash = await index_profile_store.ensure_profile(
        build_index_profile(settings, runtime_profile)
    )
    documents = await document_store.list_all()
    knowledge_bases: dict[tuple[str, str], list[dict]] = {}
    for document in documents:
        if not document.get("kb_id") or not document.get("tenant_id"):
            continue
        knowledge_bases.setdefault(
            (str(document["tenant_id"]), str(document["kb_id"])),
            [],
        ).append(document)

    scheduled: list[dict] = []
    transient_statuses = {"processing", "reindex_queued", "reindexing", "deleting"}
    for (tenant_id, kb_id), kb_documents in knowledge_bases.items():
        if any(document.get("status") in transient_statuses for document in kb_documents):
            logger.info("KB profile rebuild waits for source jobs: kb_id=%s", kb_id)
            continue
        ready_documents = [item for item in kb_documents if item.get("status") == "ready"]
        document = ready_documents[0] if ready_documents else kb_documents[0]
        active = await index_profile_store.get_active_index(kb_id)
        current_fingerprint = knowledge_base_source_fingerprint(ready_documents)
        if active and active.get("profile_hash") == desired_hash:
            # Older metadata rows predate generation fingerprints.  Preserve
            # their historical "profile already current" behavior; every new
            # generation always stores the fingerprint and is rebuilt when its
            # ready-source snapshot changes.
            active_fingerprint = active.get("source_fingerprint")
            if not active_fingerprint or active_fingerprint == current_fingerprint:
                continue
        if not ready_documents and (active is None or not active.get("source_fingerprint")):
            continue
        scheduled.append(await ingest_job_store.get_or_create_active_kb_job(
            tenant_id,
            "index_candidate",
            kb_id=kb_id,
            payload={
                "kb_id": kb_id,
                "tenant_slug": document.get("tenant_slug") or "default",
                "auto_activate": True,
                "reason": "index_profile_changed",
            },
        ))
    return scheduled


async def reclaim_excess_inactive_knowledge_base_indexes(
    *,
    kb_id: str,
    retain_ready_generations: int,
    vector_store,
    index_profile_store,
    bm25_store=None,
) -> list[dict]:
    """Bound storage only after a verified replacement has become active."""
    indexes = await index_profile_store.list_knowledge_base_indexes(kb_id)
    reclaimable = [
        item for item in indexes
        if item.get("status") in {"ready", "failed", "retired"}
    ]
    inactive_ready = [item for item in reclaimable if item.get("status") == "ready"]
    keep_ids = {
        item["index_id"]
        for item in inactive_ready[:max(0, int(retain_ready_generations))]
    }
    reclaimed: list[dict] = []
    for index in reclaimable:
        if index.get("index_id") in keep_ids:
            continue
        reclaimed.append(await reclaim_inactive_knowledge_base_index(
            kb_id=kb_id,
            index_id=index["index_id"],
            vector_store=vector_store,
            index_profile_store=index_profile_store,
            bm25_store=bm25_store,
        ))
    return reclaimed


async def reclaim_stale_ready_knowledge_base_indexes(
    *,
    document_store,
    vector_store,
    index_profile_store,
    bm25_store=None,
    kb_ids: list[str] | None = None,
) -> list[dict]:
    """Remove ready candidates that no longer match their KB source snapshot.

    Generations are immutable: a candidate built before an upload, deletion or
    reingest can never become safe again.  Keeping it as a rollback version
    only confuses users and wastes vector storage, because activation correctly
    rejects its stale source fingerprint.
    """
    list_indexes = getattr(index_profile_store, "list_knowledge_base_indexes", None)
    if not callable(list_indexes):
        return []

    target_kb_ids = list(dict.fromkeys(kb_ids or []))
    if not target_kb_ids:
        list_all_indexes = getattr(index_profile_store, "list_all_indexes", None)
        if not callable(list_all_indexes):
            return []
        target_kb_ids = list(dict.fromkeys(
            str(index["kb_id"])
            for index in await list_all_indexes()
            if index.get("status") == "active"
        ))

    reclaimed: list[dict] = []
    for kb_id in target_kb_ids:
        ready_documents = await document_store.list_by_knowledge_base(kb_id, status="ready")
        current_fingerprint = knowledge_base_source_fingerprint(ready_documents)
        for index in await list_indexes(kb_id):
            if (
                index.get("status") == "ready"
                and index.get("source_fingerprint") != current_fingerprint
            ):
                reclaimed.append(await reclaim_inactive_knowledge_base_index(
                    kb_id=kb_id,
                    index_id=index["index_id"],
                    vector_store=vector_store,
                    index_profile_store=index_profile_store,
                    bm25_store=bm25_store,
                ))
    return reclaimed


async def reclaim_expired_inactive_knowledge_base_indexes(
    *,
    retention_days: int,
    vector_store,
    index_profile_store,
    bm25_store=None,
) -> list[dict]:
    """Delete inactive generations whose rollback retention window elapsed."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, int(retention_days)))
    expired = await index_profile_store.list_expired_inactive_indexes(cutoff.isoformat())
    reclaimed: list[dict] = []
    for index in expired:
        reclaimed.append(await reclaim_inactive_knowledge_base_index(
            kb_id=index["kb_id"],
            index_id=index["index_id"],
            vector_store=vector_store,
            index_profile_store=index_profile_store,
            bm25_store=bm25_store,
        ))
    return reclaimed


async def reclaim_unreferenced_vector_storage(
    *,
    settings,
    vector_store,
    document_store,
    index_profile_store,
) -> dict:
    """Reclaim physical vector data not referenced by the current lifecycle.

    Index metadata is the source of truth for profile generations, while
    documents still reference the compatibility collection.  The VectorStore
    additionally protects user-owned Vector API collections by their metadata.
    Keeping this set construction here makes startup, cutover and periodic
    maintenance use exactly the same safety boundary.
    """
    cleanup = getattr(vector_store, "cleanup_orphaned_storage", None)
    if not callable(cleanup):
        return {}

    protected: set[str] = set()
    list_all_indexes = getattr(index_profile_store, "list_all_indexes", None)
    if callable(list_all_indexes):
        for index in await list_all_indexes():
            if index.get("status") in {"building", "ready", "active"}:
                protected.add(str(index.get("collection_name") or ""))

    list_all_documents = getattr(document_store, "list_all", None)
    if callable(list_all_documents):
        for document in await list_all_documents():
            protected.add(get_tenant_rag_collection_name(
                settings.vectorstore.collection_prefix,
                document.get("embedding_model") or settings.embedding.openai.model_name,
                tenant_slug=document.get("tenant_slug"),
                embedding_dimension=document.get("embedding_dimension"),
            ))

    try:
        result = await cleanup(
            protected_collection_names=protected,
            orphan_grace_seconds=getattr(
                settings.app,
                "vector_storage_orphan_grace_seconds",
                86_400,
            ),
        )
    except Exception:
        # Storage auditing is retryable maintenance.  It must not prevent the
        # application from starting or make a successful cutover look failed.
        logger.exception("孤立向量存储回收失败，将在下一次维护时重试")
        return {"error": "cleanup_failed"}
    if result.get("deleted_collections") or result.get("deleted_orphan_segments"):
        logger.info(
            "孤立向量存储已回收: collections=%d segments=%d bytes=%d",
            result.get("deleted_collections", 0),
            result.get("deleted_orphan_segments", 0),
            result.get("removed_bytes", 0),
        )
    return result


async def run_index_retention_maintenance(
    *,
    settings,
    vector_store,
    document_store,
    index_profile_store,
    bm25_store,
    stop_event: asyncio.Event,
) -> None:
    """Periodically enforce time-based retention without requiring a restart."""
    interval = int(getattr(settings.app, "index_retention_cleanup_interval_seconds", 0))
    if interval <= 0:
        return
    interval = max(60, interval)
    retention_days = max(0, int(getattr(settings.app, "index_generation_retention_days", 7)))
    while not stop_event.is_set():
        try:
            reclaimed = await reclaim_expired_inactive_knowledge_base_indexes(
                retention_days=retention_days,
                vector_store=vector_store,
                index_profile_store=index_profile_store,
                bm25_store=bm25_store,
            )
            if reclaimed:
                logger.info("expired index generations reclaimed: count=%d", len(reclaimed))
            await reclaim_unreferenced_vector_storage(
                settings=settings,
                vector_store=vector_store,
                document_store=document_store,
                index_profile_store=index_profile_store,
            )
        except Exception:
            logger.exception("expired index generation cleanup failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


async def schedule_knowledge_base_reconciliations_after_source_changes(
    *,
    documents: list[dict],
    ingest_job_store,
    reason: str = "source_changed",
) -> list[dict]:
    """Queue one durable, KB-scoped reconciliation job per changed KB.

    Source writes may finish concurrently, but candidate construction must not.
    The database queue's KB-scoped unique claim is the serialization boundary
    shared by uploads, manual reingest and configuration-change recovery.
    """
    representatives: dict[tuple[str, str], dict] = {}
    for document in documents:
        if document.get("status") != "ready":
            continue
        tenant_id = str(document.get("tenant_id") or "")
        kb_id = str(document.get("kb_id") or "")
        if tenant_id and kb_id:
            representatives.setdefault((tenant_id, kb_id), document)

    jobs: list[dict] = []
    for (tenant_id, kb_id), document in representatives.items():
        jobs.append(await ingest_job_store.get_or_create_active_kb_job(
            tenant_id,
            "index_candidate",
            kb_id=kb_id,
            payload={
                "kb_id": kb_id,
                "tenant_slug": document.get("tenant_slug") or "default",
                "auto_activate": True,
                "reason": reason,
            },
        ))
    return jobs


async def reclaim_inactive_knowledge_base_index(
    *,
    kb_id: str,
    index_id: str,
    vector_store,
    index_profile_store,
    bm25_store=None,
) -> dict:
    """Release an obsolete rollback generation without touching live traffic."""
    index = await index_profile_store.get_knowledge_base_index(kb_id, index_id)
    if not index:
        raise KeyError(f"index not found: {kb_id}/{index_id}")
    if index["status"] == "active":
        raise ValueError("不能回收当前活动索引")

    await delete_collection_if_present(vector_store, index["collection_name"])
    await ParentChunkStore(index_profile_store.db_path).delete_profile(index["index_id"])
    deleted = await index_profile_store.delete_inactive_index(kb_id, index_id)
    if bm25_store is not None:
        bm25_store.invalidate_cache(index["collection_name"])
    return deleted
