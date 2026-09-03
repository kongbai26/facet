"""Startup recovery for document metadata and local storage."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from app.pipeline.reindex import (
    auto_reindex_enabled,
    find_document_file,
    queue_batch_reindex,
)
from app.pipeline.document_runtime_cleanup import cleanup_legacy_document_runtime
from app.pipeline.ingest import EMPTY_CONTENT_ERROR_MESSAGE, EMPTY_CONTENT_STATUS_REASON
from app.pipeline.tokenizer_policy import is_recoverable_tokenizer_capability_failure
from app.pipeline.bm25_lifecycle import resolve_bm25_target_collections
from app.settings.settings import AppConfig
from app.rag_scope import get_tenant_rag_collection_name, resolve_embedding_profile
from app.store.bm25_store import BM25Store
from app.store.document_store import DocumentStore
from app.store.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def recover_storage(
    settings: AppConfig,
    document_store: DocumentStore,
    vector_store: VectorStore,
    bm25_store: Optional[BM25Store] = None,
    ingest_job_store=None,
    embedding_provider=None,
) -> None:
    """Repair interrupted lifecycle states and old metadata before serving traffic."""
    touched_models: set[str] = set()
    reindex_candidates: list[dict] = []
    docs = await document_store.list_all()
    from app.store.index_profile_store import IndexProfileStore
    from app.store.knowledge_base_store import KnowledgeBaseStore

    index_profile_store = IndexProfileStore(settings.storage.metadata_db)
    if ingest_job_store is not None:
        knowledge_base_store = KnowledgeBaseStore(settings.storage.metadata_db)
        for knowledge_base in await knowledge_base_store.list_all():
            if knowledge_base.get("status") not in {"deleting", "delete_failed"}:
                continue
            await ingest_job_store.queue_knowledge_base_delete(
                knowledge_base["tenant_id"],
                kb_id=knowledge_base["kb_id"],
                payload={
                    "kb_id": knowledge_base["kb_id"],
                    "knowledge_base_name": knowledge_base.get("name") or "知识库",
                    "tenant_slug": knowledge_base.get("slug") or "default",
                    "reason": "startup_recovery",
                },
            )
    if _needs_repair(settings, docs):
        _backup_metadata_db(settings.storage.metadata_db)
    can_auto_reindex = auto_reindex_enabled(settings) and ingest_job_store is not None
    current_dimension = None
    current_context_window = None
    if embedding_provider is not None:
        try:
            profile = await resolve_embedding_profile(embedding_provider)
            current_dimension = profile.get("dimension")
            if profile.get("context_window") is not None:
                current_context_window = int(profile["context_window"])
        except Exception as exc:
            logger.warning("无法获取当前 embedding 维度，启动恢复将仅按模型名判断: %s", exc)
    current_endpoint = (settings.embedding.openai.api_base or "").rstrip("/")

    # Index metadata is a routing contract. If its active physical collection
    # vanished (for example after an interrupted volume restore), do not let
    # later reads recreate an empty collection or fall back to stale legacy
    # vectors. Mark the generation failed and queue the normal immutable
    # candidate lifecycle once per KB.
    missing_active_index_kb_ids: set[str] = set()
    collection_exists = getattr(vector_store, "collection_exists", None)
    if callable(collection_exists):
        docs_by_kb: dict[str, list[dict]] = {}
        for document in docs:
            kb_id = str(document.get("kb_id") or "")
            if kb_id:
                docs_by_kb.setdefault(kb_id, []).append(document)
        for kb_id, kb_documents in docs_by_kb.items():
            active_index = await index_profile_store.get_active_index(kb_id)
            if not active_index:
                continue
            collection_name = str(active_index.get("collection_name") or "")
            if not collection_name:
                continue
            try:
                exists = await collection_exists(collection_name=collection_name)
            except Exception as exc:
                logger.warning(
                    "无法确认活动索引 collection 是否存在，跳过自动修复: kb_id=%s error=%s",
                    kb_id,
                    exc,
                )
                continue
            if exists:
                continue

            logger.error(
                "活动索引 collection 缺失，已标记并排队重建: kb_id=%s collection=%s",
                kb_id,
                collection_name,
            )
            await index_profile_store.mark_index(
                kb_id,
                active_index["index_id"],
                "failed",
                error_message="活动索引的物理 collection 缺失，已排队自动重建。",
            )
            missing_active_index_kb_ids.add(kb_id)
            if ingest_job_store is not None:
                representative = kb_documents[0]
                await ingest_job_store.get_or_create_active_kb_job(
                    representative.get("tenant_id") or "",
                    "index_candidate",
                    kb_id=kb_id,
                    payload={
                        "kb_id": kb_id,
                        "tenant_slug": representative.get("tenant_slug") or "default",
                        "auto_activate": True,
                        "reason": "recovery_active_collection_missing",
                    },
                )

    for doc in docs:
        model = doc.get("embedding_model") or settings.embedding.openai.model_name
        doc_dimension = doc.get("embedding_dimension")
        collection_name = _get_doc_collection_name(settings, doc, model, doc_dimension)
        content_hash = doc.get("content_hash") or ""
        upload_file = find_document_file(settings.storage.upload_dir, doc)

        if not content_hash and upload_file:
            content_hash = _sha256_file(upload_file)
            await document_store.set_content_hash(doc["doc_id"], content_hash, model, doc_dimension)
        elif doc.get("embedding_model") != model:
            await document_store.set_content_hash(doc["doc_id"], content_hash, model, doc_dimension)

        status = doc.get("status")
        status_reason = doc.get("status_reason") or ""
        if status == "processing":
            chunks_count = doc.get("chunks_count") or 0
            # 智能恢复：有 chunks + 有向量 → 标记 ready，否则标记 failed
            if chunks_count > 0 and upload_file:
                try:
                    vector_count = await _count_doc_vectors(
                        vector_store,
                        doc["doc_id"],
                        collection_name,
                    )
                    if vector_count is None:
                        logger.warning(f"文档 {doc['doc_id']} 无法确认向量数量，跳过自动恢复")
                        continue
                    if vector_count > 0:
                        logger.info(f"文档 {doc['doc_id']} 摄入已完成但状态未更新，自动恢复为 ready")
                        await document_store.update_status_if(
                            doc["doc_id"],
                            ["processing"],
                            "ready",
                            status_reason="",
                        )
                        touched_models.add(collection_name)
                        continue
                except Exception as e:
                    logger.warning(f"检查文档 {doc['doc_id']} 向量时出错: {e}")

            # 无法恢复：清理残留，标记 failed
            await _cleanup_doc(settings, vector_store, doc)
            await document_store.update_status_if(
                doc["doc_id"],
                ["processing"],
                "failed",
                error_message="服务启动时发现上次摄入中断，已清理残留",
                chunks_count=0,
                status_reason="interrupted_ingest",
            )
            touched_models.add(collection_name)
            continue

        profile_active = None
        if doc.get("kb_id"):
            profile_active = await index_profile_store.get_active_index(str(doc["kb_id"]))

        if status == "deleting":
            if profile_active is not None:
                if ingest_job_store is None:
                    await document_store.update_status_if(
                        doc["doc_id"],
                        ["deleting"],
                        "delete_failed",
                        error_message="服务启动时没有可用的索引任务队列，未修改活动索引。",
                    )
                    continue
                await ingest_job_store.get_or_create_active_kb_job(
                    doc.get("tenant_id") or "",
                    "index_candidate",
                    kb_id=doc["kb_id"],
                    payload={
                        "kb_id": doc["kb_id"],
                        "tenant_slug": doc.get("tenant_slug") or "default",
                        "auto_activate": True,
                        "reason": "recovery_document_delete",
                    },
                )
                continue
            errors = await _cleanup_doc(settings, vector_store, doc)
            touched_models.add(collection_name)
            if errors:
                await document_store.update_status_if(
                    doc["doc_id"],
                    ["deleting"],
                    "delete_failed",
                    error_message="; ".join(errors),
                    status_reason="",
                )
            else:
                try:
                    await document_store.delete(doc["doc_id"])
                except KeyError:
                    pass
            continue

        if status == "delete_failed":
            if profile_active is not None:
                if ingest_job_store is None:
                    continue
                await ingest_job_store.get_or_create_active_kb_job(
                    doc.get("tenant_id") or "",
                    "index_candidate",
                    kb_id=doc["kb_id"],
                    payload={
                        "kb_id": doc["kb_id"],
                        "tenant_slug": doc.get("tenant_slug") or "default",
                        "auto_activate": True,
                        "reason": "recovery_document_delete",
                    },
                )
                continue
            errors = await _cleanup_doc(settings, vector_store, doc)
            touched_models.add(collection_name)
            if errors:
                await document_store.update_status_if(
                    doc["doc_id"],
                    ["delete_failed"],
                    "delete_failed",
                    error_message="; ".join(errors),
                    status_reason="",
                )
            else:
                try:
                    await document_store.delete(doc["doc_id"])
                except KeyError:
                    pass
            continue

        if status in {"reindex_queued", "reindexing"}:
            if not upload_file:
                await document_store.update_status(
                    doc["doc_id"],
                    "failed",
                    error_message="上传原文件缺失，无法重新摄入。",
                    chunks_count=0,
                    status_reason="source_missing",
                )
                continue
            if can_auto_reindex:
                reindex_candidates.append(doc)
            elif status_reason == "model_mismatch":
                await document_store.update_status(
                    doc["doc_id"],
                    "failed",
                    error_message="当前无法自动重建，请稍后手动重新摄入。",
                    chunks_count=0,
                    status_reason="model_mismatch",
                )
            continue

        if not upload_file and status == "ready":
            await _delete_doc_vectors(vector_store, doc["doc_id"], collection_name)
            await document_store.update_status_if(
                doc["doc_id"],
                ["ready"],
                "failed",
                error_message="上传文件缺失，已从可检索状态移除",
                chunks_count=0,
                status_reason="source_missing",
            )
            touched_models.add(collection_name)
            continue

        # Once a KB has an active immutable generation, the source document's
        # old embedding metadata and compatibility collection are no longer a
        # retrieval contract.  Model/context changes are handled by the same
        # profile-candidate scheduler as every other index change.
        if status == "ready" and doc.get("kb_id") in missing_active_index_kb_ids:
            # The KB-level candidate job owns this repair. Do not enqueue a
            # separate legacy-document reindex while it is rebuilding the
            # missing immutable generation.
            continue

        if status == "ready" and profile_active is not None:
            continue

        endpoint_mismatch = bool(doc.get("embedding_endpoint")) and (
            doc.get("embedding_endpoint") or "").rstrip("/") != current_endpoint
        context_mismatch = bool(doc.get("embedding_context_window")) and (
            current_context_window is not None
            and int(doc.get("embedding_context_window")) != current_context_window
        )
        if status == "ready" and (
            model != settings.embedding.openai.model_name or endpoint_mismatch or context_mismatch
        ):
            if upload_file and can_auto_reindex:
                reindex_candidates.append(doc)
            else:
                await document_store.update_status_if(
                    doc["doc_id"],
                    ["ready"],
                    "failed",
                    error_message=(
                        f"embedding_model={model} 或 embedding_endpoint={doc.get('embedding_endpoint')} "
                        "与当前 embedding 部署不一致，等待重新建立索引"
                    ),
                    chunks_count=0,
                    status_reason="model_mismatch",
                )
            touched_models.add(collection_name)
            continue

        if status == "ready" and upload_file and (doc.get("chunks_count") or 0) == 0:
            try:
                vector_count = await _count_doc_vectors(
                    vector_store,
                    doc["doc_id"],
                    collection_name,
                )
            except Exception as e:
                logger.warning(f"检查文档 {doc['doc_id']} 零 chunk 向量时出错: {e}")
                vector_count = None

            if vector_count == 0:
                logger.warning(f"文档 {doc['doc_id']} 没有解析出可检索文本，恢复为 failed")
                await document_store.update_status_if(
                    doc["doc_id"],
                    ["ready"],
                    "failed",
                    error_message=EMPTY_CONTENT_ERROR_MESSAGE,
                    chunks_count=0,
                    status_reason=EMPTY_CONTENT_STATUS_REASON,
                )
                touched_models.add(collection_name)
            continue

        if status == "ready" and current_dimension is not None and doc_dimension != current_dimension:
            if upload_file and can_auto_reindex:
                reindex_candidates.append(doc)
            else:
                await document_store.update_status_if(
                    doc["doc_id"],
                    ["ready"],
                    "failed",
                    error_message=(
                        f"embedding_model={model} 或 embedding_dimension={doc_dimension} "
                        "与当前配置不一致，等待重新建立索引"
                    ),
                    chunks_count=0,
                    status_reason="model_mismatch",
                )
            touched_models.add(collection_name)
            continue

        # 模型/维度已切回与当前配置一致，但文档仍残留 model_mismatch 失败状态：尝试恢复。
        if status == "failed" and status_reason == "model_mismatch" and _doc_matches_current_model(
            doc, settings, current_dimension, current_endpoint, current_context_window
        ):
            try:
                vector_count = await _count_doc_vectors(vector_store, doc["doc_id"], collection_name)
            except Exception as e:
                logger.warning(f"检查文档 {doc['doc_id']} 向量时出错: {e}")
                vector_count = None

            if vector_count is None:
                continue
            if vector_count > 0:
                logger.info(f"文档 {doc['doc_id']} 模型/维度已匹配且向量存在，恢复为 ready")
                await document_store.update_status_if(
                    doc["doc_id"],
                    ["failed"],
                    "ready",
                    error_message="",
                    chunks_count=doc.get("chunks_count") or vector_count,
                    status_reason="",
                )
                touched_models.add(collection_name)
            elif not upload_file:
                await document_store.update_status(
                    doc["doc_id"],
                    "failed",
                    error_message="上传原文件缺失，无法重新摄入。",
                    chunks_count=0,
                    status_reason="source_missing",
                )
                touched_models.add(collection_name)
            elif can_auto_reindex:
                reindex_candidates.append(doc)
                touched_models.add(collection_name)
            continue

        # A previous deployment may have treated a missing non-standard
        # /tokenize endpoint as terminal.  Once the policy is relaxed, retain
        # the source file and queue it like every other durable rebuild.  Do
        # this only after startup's embedding probe succeeded; otherwise a
        # model outage would create a job that is known to fail immediately.
        if is_recoverable_tokenizer_capability_failure(doc, settings):
            if not upload_file:
                await document_store.update_status(
                    doc["doc_id"],
                    "failed",
                    error_message="上传原文件缺失，无法重新摄入。",
                    chunks_count=0,
                    status_reason="source_missing",
                )
            elif can_auto_reindex and embedding_provider is not None:
                reindex_candidates.append(doc)
                touched_models.add(collection_name)
            continue

        if status == "failed" and (
            _is_recoverable_model_mismatch_failure(doc, settings)
            or (current_dimension is not None and doc_dimension != current_dimension)
            or endpoint_mismatch
            or context_mismatch
        ):
            if model != settings.embedding.openai.model_name and status_reason != "model_mismatch":
                await document_store.update_status(
                    doc["doc_id"],
                    "failed",
                    error_message=f"embedding_model={model} 与当前模型不一致，等待重新建立索引",
                    chunks_count=0,
                    status_reason="model_mismatch",
                )
                doc = await document_store.get(doc["doc_id"]) or doc
            if current_dimension is not None and doc_dimension != current_dimension and status_reason != "model_mismatch":
                await document_store.update_status(
                    doc["doc_id"],
                    "failed",
                    error_message=(
                        f"embedding_model={model} 或 embedding_dimension={doc_dimension} "
                        "与当前配置不一致，等待重新建立索引"
                    ),
                    chunks_count=0,
                    status_reason="model_mismatch",
                )
                doc = await document_store.get(doc["doc_id"]) or doc
            if not upload_file:
                await document_store.update_status(
                    doc["doc_id"],
                    "failed",
                    error_message="上传原文件缺失，无法重新摄入。",
                    chunks_count=0,
                    status_reason="source_missing",
                )
            elif can_auto_reindex:
                reindex_candidates.append(doc)
            touched_models.add(collection_name)
            continue

        # 向量缺失检测：文档 ready 但向量库中无对应向量
        if status == "ready" and upload_file:
            chunks_count = doc.get("chunks_count") or 0
            if chunks_count > 0:
                try:
                    vector_count = await _count_doc_vectors(
                        vector_store,
                        doc["doc_id"],
                        collection_name,
                    )
                    if vector_count is None:
                        logger.warning(f"文档 {doc['doc_id']} 无法确认向量数量，跳过缺失检测")
                        continue
                    if vector_count == 0:
                        logger.warning(f"文档 {doc['doc_id']} 向量数据丢失（chunks_count={chunks_count}），标记为 failed")
                        await document_store.update_status_if(
                            doc["doc_id"],
                            ["ready"],
                            "failed",
                            error_message="向量数据丢失，需重新摄入",
                            chunks_count=0,
                            status_reason="vector_missing",
                        )
                        touched_models.add(collection_name)
                except Exception as e:
                    logger.warning(f"检查文档 {doc['doc_id']} 向量时出错: {e}")

    if reindex_candidates and can_auto_reindex:
        grouped_candidates: dict[str, list[dict]] = {}
        for doc in reindex_candidates:
            tenant_slug = doc.get("tenant_slug") or "default"
            grouped_candidates.setdefault(tenant_slug, []).append(doc)

        for tenant_slug, docs_for_tenant in grouped_candidates.items():
            batch_result = await queue_batch_reindex(
                settings,
                document_store,
                ingest_job_store,
                docs_for_tenant,
                tenant_id=docs_for_tenant[0].get("tenant_id") or "",
                trigger="model_change",
                embedding_provider=embedding_provider,
                status_reason=(
                    "tokenizer_capability"
                    if all(is_recoverable_tokenizer_capability_failure(item, settings) for item in docs_for_tenant)
                    else "model_mismatch"
                ),
            )
            if batch_result.get("ok"):
                logger.info(
                    "启动恢复创建了批量 reindex job: tenant_slug=%s job_id=%s documents=%d",
                    tenant_slug,
                    batch_result["job"]["job_id"],
                    len(batch_result.get("document_ids", [])),
                )
            else:
                logger.warning(
                    "启动恢复创建批量 reindex job 失败: tenant_slug=%s reason=%s",
                    tenant_slug,
                    batch_result.get("message"),
                )

    # Older chunks predate KB-level metadata. Backfill it without changing
    # embeddings/text before retrieval starts using direct kb_id filters.
    refreshed_docs = await document_store.list_all()
    touched_models.update(
        await _backfill_knowledge_base_metadata(settings, vector_store, refreshed_docs)
    )
    touched_models.update(await _deduplicate_documents(settings, document_store, vector_store))
    await document_store.ensure_unique_content_index()
    # 先失效可能过期的 BM25 缓存，再为实际检索会访问的 collection 预暖，
    # 避免 touched_models 包含当前 collection 时把刚预暖的缓存又清掉。
    _invalidate_bm25(settings, bm25_store, touched_models)
    await _warmup_bm25_for_retrieval_collections(
        settings,
        bm25_store,
        vector_store,
        document_store,
        index_profile_store=index_profile_store,
    )
    _cleanup_metadata_backups(
        settings.storage.metadata_db,
        retention_days=settings.app.metadata_backup_retention_days,
        max_files=settings.app.metadata_backup_max_files,
    )


async def _warmup_bm25_for_retrieval_collections(
    settings: AppConfig,
    bm25_store: Optional[BM25Store],
    vector_store: VectorStore,
    document_store: DocumentStore,
    *,
    index_profile_store=None,
) -> None:
    """启动时预先构建/加载实际检索会访问的全部 BM25 索引。"""
    if not settings.retrieval.hybrid.enabled or not bm25_store:
        return
    if index_profile_store is None:
        return
    collections = await resolve_bm25_target_collections(
        settings,
        document_store,
        index_profile_store,
    )
    for collection_name in sorted(collections):
        try:
            await bm25_store.ensure_ready(
                vector_store,
                collection_name,
                document_store=document_store,
            )
            logger.info("BM25 启动预暖完成: collection=%s", collection_name)
        except Exception as exc:
            logger.warning(
                "BM25 启动预暖失败，将在首次检索时懒加载: collection=%s error=%s",
                collection_name,
                exc,
            )


async def _deduplicate_documents(
    settings: AppConfig,
    document_store: DocumentStore,
    vector_store: VectorStore,
) -> set[str]:
    touched_models: set[str] = set()
    docs = await document_store.list_all()
    groups: dict[tuple[str, str, str, str, str | None], list[dict]] = {}

    for doc in docs:
        content_hash = doc.get("content_hash") or ""
        tenant_id = doc.get("tenant_id") or ""
        model = doc.get("embedding_model") or settings.embedding.openai.model_name
        dimension = doc.get("embedding_dimension")
        if not content_hash:
            continue
        groups.setdefault((tenant_id, doc.get("kb_id") or "", content_hash, model, dimension), []).append(doc)

    for (_tenant_id, _kb_id, content_hash, model, dimension), duplicates in groups.items():
        if len(duplicates) <= 1:
            continue

        keeper = sorted(duplicates, key=_dedupe_sort_key)[0]
        logger.warning(
            "发现重复文档 content_hash=%s model=%s dimension=%s，保留 %s，清理 %d 条",
            content_hash,
            model,
            dimension,
            keeper["doc_id"],
            len(duplicates) - 1,
        )
        for doc in duplicates:
            if doc["doc_id"] == keeper["doc_id"]:
                continue
            collection_name = _get_doc_collection_name(settings, doc, model, doc.get("embedding_dimension"))
            errors = await _cleanup_doc(settings, vector_store, doc)
            if errors:
                logger.warning("重复文档 %s 清理不完整: %s", doc["doc_id"], errors)
                await document_store.set_content_hash(doc["doc_id"], "", model, dimension)
                await document_store.update_status(
                    doc["doc_id"],
                    "delete_failed",
                    error_message="; ".join(errors),
                )
            else:
                try:
                    await document_store.delete(doc["doc_id"])
                except KeyError:
                    pass
            touched_models.add(collection_name)

    return touched_models


async def _cleanup_doc(
    settings: AppConfig,
    vector_store: VectorStore,
    document: dict,
) -> list[str]:
    cleanup = await cleanup_legacy_document_runtime(
        document=document,
        settings=settings,
        vector_store=vector_store,
        compact_empty_collection=True,
    )
    if not cleanup.errors:
        # State retirement is metadata-only; physical profile generations are
        # owned by the lifecycle coordinator and were never touched here.
        from app.store.index_profile_store import IndexProfileStore

        await IndexProfileStore(settings.storage.metadata_db).retire_document_states(document["doc_id"])

    return list(cleanup.errors)


def _needs_repair(settings: AppConfig, docs: list[dict]) -> bool:
    seen: set[tuple[str, str, str, str]] = set()
    for doc in docs:
        model = doc.get("embedding_model") or settings.embedding.openai.model_name
        content_hash = doc.get("content_hash") or ""
        dimension = doc.get("embedding_dimension")
        status = doc.get("status")
        status_reason = doc.get("status_reason") or ""
        if status in {"processing", "deleting", "delete_failed", "reindex_queued", "reindexing"}:
            return True
        if status == "ready" and model != settings.embedding.openai.model_name:
            return True
        if status == "failed" and _is_recoverable_model_mismatch_failure(doc, settings):
            return True
        if status == "ready" and (doc.get("chunks_count") or 0) == 0:
            return True
        if status == "failed" and status_reason == "model_mismatch" and _doc_matches_current_model(
            doc, settings, None
        ):
            return True
        if status == "ready" and not find_document_file(settings.storage.upload_dir, doc):
            return True
        if not content_hash:
            return True
        if not doc.get("tenant_id") or not doc.get("tenant_slug"):
            return True
        key = (doc.get("tenant_id") or "", content_hash, model, str(dimension))
        if key in seen:
            return True
        seen.add(key)
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dedupe_sort_key(doc: dict) -> tuple[int, str]:
    priority = {
        "ready": 0,
        "reindexing": 1,
        "reindex_queued": 2,
        "failed": 3,
        "delete_failed": 4,
        "processing": 5,
        "deleting": 6,
    }.get(doc.get("status"), 7)
    return (priority, str(doc.get("created_at") or ""))


def _is_recoverable_model_mismatch_failure(doc: dict, settings: AppConfig) -> bool:
    model = doc.get("embedding_model") or settings.embedding.openai.model_name
    if (doc.get("status_reason") or "") == "model_mismatch":
        return model != settings.embedding.openai.model_name
    error_message = doc.get("error_message") or ""
    return "embedding_model=" in error_message and "不一致" in error_message


def _doc_matches_current_model(
    doc: dict,
    settings: AppConfig,
    current_dimension: int | None,
    current_endpoint: str | None = None,
    current_context_window: int | None = None,
) -> bool:
    """判断文档记录的模型/维度是否与当前配置一致。"""
    model = doc.get("embedding_model") or settings.embedding.openai.model_name
    if model != settings.embedding.openai.model_name:
        return False
    if doc.get("embedding_endpoint") and current_endpoint is not None:
        if (doc.get("embedding_endpoint") or "").rstrip("/") != current_endpoint.rstrip("/"):
            return False
    if doc.get("embedding_context_window") and current_context_window is not None:
        if int(doc["embedding_context_window"]) != current_context_window:
            return False
    doc_dimension = doc.get("embedding_dimension")
    if current_dimension is None:
        return True
    return doc_dimension == current_dimension


def _backup_metadata_db(metadata_db: str) -> None:
    db_path = Path(metadata_db)
    if not db_path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup_path = db_path.with_suffix(f".{stamp}.bak")
    try:
        # SQLite WAL commits may not yet reside in metadata.db itself.  The
        # online backup API copies a consistent logical snapshot, unlike a
        # plain filesystem copy of only the main database file.
        source = sqlite3.connect(db_path)
        destination = sqlite3.connect(backup_path)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        logger.info("metadata.db 已备份到 %s", backup_path)
    except Exception as e:
        logger.warning("metadata.db 备份失败: %s", e)


def _cleanup_metadata_backups(
    metadata_db: str,
    *,
    retention_days: int,
    max_files: int,
) -> int:
    db_path = Path(metadata_db)
    if not db_path.exists():
        return 0

    backup_paths = []
    for path in db_path.parent.glob(f"{db_path.stem}.*.bak"):
        if path.is_file():
            try:
                backup_paths.append((path.stat().st_mtime, path))
            except OSError as exc:
                logger.warning("读取 metadata backup 元数据失败: %s", exc)

    if not backup_paths:
        return 0

    keep_count = max(1, max_files)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, retention_days))
    backup_paths.sort(key=lambda item: item[0], reverse=True)

    deleted = 0
    for index, (mtime, path) in enumerate(backup_paths):
        if index < keep_count and datetime.fromtimestamp(mtime, tz=timezone.utc) >= cutoff:
            continue
        try:
            path.unlink()
            deleted += 1
        except OSError as exc:
            logger.warning("清理 metadata backup 失败: %s", exc)

    if deleted:
        logger.info("metadata.db 旧备份已清理: deleted=%d kept=%d", deleted, len(backup_paths) - deleted)
    return deleted


def _invalidate_bm25(
    settings: AppConfig,
    bm25_store: Optional[BM25Store],
    models: set[str],
) -> None:
    if not bm25_store or not settings.retrieval.hybrid.enabled:
        return
    for collection_name in models:
        bm25_store.invalidate_cache(collection_name)


def _get_doc_collection_name(settings: AppConfig, doc: dict, model: str, embedding_dimension: int | None = None) -> str:
    return get_tenant_rag_collection_name(
        settings.vectorstore.collection_prefix,
        model,
        tenant_slug=doc.get("tenant_slug"),
        embedding_dimension=embedding_dimension,
    )


async def _backfill_knowledge_base_metadata(
    settings: AppConfig,
    vector_store: VectorStore,
    docs: list[dict],
) -> set[str]:
    """Idempotently add authoritative kb_id metadata to old vector chunks."""
    collections: dict[str, dict[str, str]] = {}
    for doc in docs:
        if doc.get("status") != "ready" or not doc.get("kb_id"):
            continue
        collection_name = _get_doc_collection_name(
            settings,
            doc,
            doc.get("embedding_model") or settings.embedding.openai.model_name,
            doc.get("embedding_dimension"),
        )
        collections.setdefault(collection_name, {})[doc["doc_id"]] = doc["kb_id"]

    touched: set[str] = set()
    for collection_name, kb_by_doc_id in collections.items():
        try:
            records = await vector_store.get_all_chunks(collection_name=collection_name)
        except Exception as exc:
            logger.warning("kb_id metadata backfill skipped: collection=%s error=%s", collection_name, exc)
            continue
        ids = records.get("ids") or []
        metadatas = records.get("metadatas") or []
        update_ids: list[str] = []
        update_metadatas: list[dict] = []
        for index, chunk_id in enumerate(ids):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            expected_kb_id = kb_by_doc_id.get(metadata.get("doc_id") or "")
            if not expected_kb_id or metadata.get("kb_id") == expected_kb_id:
                continue
            metadata["kb_id"] = expected_kb_id
            update_ids.append(chunk_id)
            update_metadatas.append(metadata)
        if not update_ids:
            continue
        try:
            await vector_store.update_metadata(update_ids, update_metadatas, collection_name=collection_name)
            touched.add(collection_name)
            logger.info("kb_id metadata backfill finished: collection=%s chunks=%d", collection_name, len(update_ids))
        except Exception as exc:
            logger.warning("kb_id metadata backfill failed: collection=%s error=%s", collection_name, exc)
    return touched


async def _count_doc_vectors(
    vector_store: VectorStore,
    doc_id: str,
    collection_name: str,
) -> int | None:
    return await vector_store.count_by_doc_id(doc_id, collection_name=collection_name)


async def _delete_doc_vectors(
    vector_store: VectorStore,
    doc_id: str,
    collection_name: str,
) -> None:
    safe_delete = getattr(vector_store, "delete_by_doc_id_if_exists", None)
    if callable(safe_delete):
        await safe_delete(doc_id, collection_name=collection_name)
        return
    await vector_store.delete_by_doc_id(doc_id, collection_name=collection_name)
