"""FastAPI 入口，lifespan 管理"""

import asyncio
import logging
import os
from contextlib import suppress
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.auth_service import (
    LEGACY_RAG_ACCESS,
    ensure_admin_credential_from_legacy,
    has_legacy_plaintext_password,
)
from app.api.auth_router import router as auth_router
from app.api.health_router import router as health_router
from app.api.ingest_router import router as ingest_router
from app.api.documents_router import router as documents_router
from app.api.knowledge_bases_router import router as knowledge_bases_router
from app.api.chat_router import router as chat_router
from app.api.system_router import router as system_router
from app.api.vector_router import router as vector_router
from app.config import get_config
from app.providers.llm.registry import resolve_llm_mode
from app.rag_scope import get_tenant_rag_collection_name
from app.utils.logger import setup_logging

logger = logging.getLogger(__name__)

# 前端构建产物目录
DIST_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"
DIST_DIR_RESOLVED = DIST_DIR.resolve()


async def _probe_embedding_for_startup(embedding_provider, timeout_seconds: int) -> bool:
    """Probe embedding once without allowing an external outage to block startup."""
    if embedding_provider is None:
        return False
    try:
        await asyncio.wait_for(
            embedding_provider.runtime_profile(),
            timeout=max(1, int(timeout_seconds)),
        )
        logger.info("Embedding 启动探测完成")
        return True
    except asyncio.TimeoutError:
        logger.warning(
            "Embedding 启动探测超过 %s 秒，服务将先以降级模式启动",
            timeout_seconds,
        )
    except Exception as exc:
        logger.warning(
            "Embedding 启动探测失败，服务将先以降级模式启动: %s",
            exc,
        )
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭生命周期管理"""
    settings = get_config()
    ingest_worker_task: asyncio.Task | None = None
    ingest_worker_stop: asyncio.Event | None = None
    index_retention_task: asyncio.Task | None = None
    index_retention_stop: asyncio.Event | None = None

    # 启动
    setup_logging(settings.server.log_level)
    logger.info("Facet 服务启动中...")
    logger.info(f"Embedding provider: {settings.embedding.provider}")
    logger.info(f"LLM model: {settings.llm.model_name}")
    logger.info(f"LLM mode: {resolve_llm_mode(settings.llm)}")

    from app.api.deps import (
        get_api_key_store_instance,
        get_auth_credential_store_instance,
        get_bm25_store_instance,
        get_conversation_store_instance,
        get_document_store_instance,
        get_embedding_provider_instance,
        get_ingest_job_store_instance,
        get_index_profile_store_instance,
        get_knowledge_base_store_instance,
        get_namespace_store_instance,
        get_principal_store_instance,
        get_session_store_instance,
        get_tenant_store_instance,
        get_vector_store_instance,
    )
    from app.bootstrap import backfill_missing_tenant_metadata, ensure_default_workspace
    from app.pipeline.recovery import recover_storage
    from app.pipeline.tokenizer_policy import is_recoverable_tokenizer_capability_failure
    from app.pipeline.index_lifecycle import (
        reclaim_expired_inactive_knowledge_base_indexes,
        reclaim_stale_ready_knowledge_base_indexes,
        reclaim_unreferenced_vector_storage,
        run_index_retention_maintenance,
        schedule_profile_rebuilds_after_configuration_change,
    )
    from app.workers.ingest_worker import run_ingest_worker_forever

    if settings.retrieval.reranker.enabled and settings.retrieval.reranker.mode != "off":
        logger.info("Reranker will be probed on demand; it is not a startup dependency")

    tenant_store = get_tenant_store_instance()
    principal_store = get_principal_store_instance()
    namespace_store = get_namespace_store_instance()
    knowledge_base_store = get_knowledge_base_store_instance()
    workspace = await ensure_default_workspace(
        settings,
        tenant_store,
        principal_store,
        namespace_store,
        knowledge_base_store,
    )
    if settings.auth.enabled:
        migrated = await ensure_admin_credential_from_legacy(
            settings,
            workspace["principal"],
            get_auth_credential_store_instance(),
        )
        if migrated:
            logger.info("已将遗留认证凭据迁移到 SQLite")
        else:
            credential = await get_auth_credential_store_instance().get_active_password_credential(
                workspace["principal"]["principal_id"]
            )
            if credential is None:
                logger.info("认证尚未初始化，首次启动将显示 Web 初始化向导")
        if has_legacy_plaintext_password(settings):
            logger.warning("检测到遗留明文密码配置，运行时已忽略，请尽快清理 .env")
    api_key_store = get_api_key_store_instance()
    await api_key_store.cleanup_expired()
    backfilled_keys = await api_key_store.backfill_empty_scope_keys(LEGACY_RAG_ACCESS)
    if backfilled_keys:
        logger.info("已回填历史空 scope API key: %d", backfilled_keys)

    document_store = get_document_store_instance()
    # Ensure profile/index lifecycle migrations are applied before recovery or
    # workers can touch source documents.
    index_profile_store = get_index_profile_store_instance()
    conversation_store = get_conversation_store_instance()
    tenant = workspace["tenant"]
    repair_stats = await backfill_missing_tenant_metadata(
        document_store,
        conversation_store,
        tenant["tenant_id"],
        tenant.get("slug") or "default",
        workspace["knowledge_base"]["kb_id"],
    )
    logger.info(
        "legacy tenant metadata repair finished: documents=%d conversations=%d knowledge_base=%d",
        repair_stats["documents_updated"],
        repair_stats["conversations_updated"],
        repair_stats["knowledge_base_updated"],
    )
    await conversation_store.mark_streaming_messages_stopped()
    session_store = get_session_store_instance()
    await session_store.cleanup_expired()
    vector_store = get_vector_store_instance()
    bm25_store = get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None
    persisted_documents = await document_store.list_all()
    persisted_indexes = await index_profile_store.list_all_indexes()
    # Failed documents are terminal source records and do not need a model
    # probe merely to let the service start. Probe only when recovery or
    # retrieval can actually use Embedding.
    embedding_probe_needed = bool(persisted_indexes) or any(
        document.get("status") in {
            "ready",
            "processing",
            "reindex_queued",
            "reindexing",
        }
        for document in persisted_documents
    ) or any(
        is_recoverable_tokenizer_capability_failure(document, settings)
        for document in persisted_documents
    )
    embedding_provider = None
    try:
        embedding_provider = get_embedding_provider_instance()
    except Exception as exc:
        logger.warning("embedding provider 初始化失败，启动恢复将跳过维度探测: %s", exc)
    if embedding_probe_needed:
        embedding_available_at_startup = await _probe_embedding_for_startup(
            embedding_provider,
            settings.app.startup_dependency_timeout_seconds,
        )
    else:
        embedding_available_at_startup = False
        logger.info("当前没有持久化文档或索引，跳过 Embedding 启动探测")
    if settings.app.enable_startup_recovery:
        await recover_storage(
            settings,
            document_store,
            vector_store,
            bm25_store,
            ingest_job_store=get_ingest_job_store_instance(),
            embedding_provider=embedding_provider if embedding_available_at_startup else None,
        )
    else:
        logger.info("启动恢复已禁用，跳过 recover_storage")

    if (
        settings.queue.backend == "db"
        and settings.app.auto_rebuild_index_on_profile_change
        and embedding_provider is not None
        and embedding_probe_needed
        and embedding_available_at_startup
    ):
        scheduled_profile_rebuilds = await schedule_profile_rebuilds_after_configuration_change(
            settings=settings,
            embedding_provider=embedding_provider,
            document_store=document_store,
            index_profile_store=index_profile_store,
            ingest_job_store=get_ingest_job_store_instance(),
        )
        if scheduled_profile_rebuilds:
            logger.info("已为配置变化自动排队候选索引重建: %d", len(scheduled_profile_rebuilds))

    reclaimed_expired_indexes = await reclaim_expired_inactive_knowledge_base_indexes(
        retention_days=settings.app.index_generation_retention_days,
        vector_store=vector_store,
        index_profile_store=index_profile_store,
        bm25_store=bm25_store,
    )
    if reclaimed_expired_indexes:
        logger.info("启动时已回收过期索引代际: %d", len(reclaimed_expired_indexes))
    reclaimed_stale_indexes = await reclaim_stale_ready_knowledge_base_indexes(
        document_store=document_store,
        vector_store=vector_store,
        index_profile_store=index_profile_store,
        bm25_store=bm25_store,
    )
    if reclaimed_stale_indexes:
        logger.info("启动时已回收过期候选索引: %d", len(reclaimed_stale_indexes))
    await reclaim_unreferenced_vector_storage(
        settings=settings,
        vector_store=vector_store,
        document_store=document_store,
        index_profile_store=index_profile_store,
    )
    if bm25_store is not None:
        valid_bm25_collections: set[str] = set()
        for document in await document_store.list_all():
            valid_bm25_collections.add(get_tenant_rag_collection_name(
                settings.vectorstore.collection_prefix,
                document.get("embedding_model") or settings.embedding.openai.model_name,
                tenant_slug=document.get("tenant_slug"),
                embedding_dimension=document.get("embedding_dimension"),
            ))
        for index in await index_profile_store.list_all_indexes():
            if index.get("status") in {"building", "ready", "active"}:
                valid_bm25_collections.add(index["collection_name"])
        orphaned_bm25_caches = bm25_store.cleanup_orphaned_caches(valid_bm25_collections)
        if orphaned_bm25_caches:
            logger.info("启动时已清理孤立 BM25 缓存: %d", orphaned_bm25_caches)
    pruned_jobs = await get_ingest_job_store_instance().prune_history(
        settings.app.ingest_job_history_retention_days,
    )
    if pruned_jobs:
        logger.info("启动时已清理过期任务历史: %d", pruned_jobs)

    if settings.queue.backend == "db" and settings.queue.autostart_worker:
        ingest_worker_stop = asyncio.Event()
        worker_id = f"ingest-worker-{os.getpid()}"
        ingest_worker_task = asyncio.create_task(
            run_ingest_worker_forever(
                worker_id=worker_id,
                settings=settings,
                stop_event=ingest_worker_stop,
            )
        )
        logger.info("队列 worker 已启动: %s", worker_id)

    if settings.app.index_retention_cleanup_interval_seconds > 0:
        index_retention_stop = asyncio.Event()
        index_retention_task = asyncio.create_task(
            run_index_retention_maintenance(
                settings=settings,
                vector_store=vector_store,
                document_store=document_store,
                index_profile_store=index_profile_store,
                bm25_store=bm25_store,
                stop_event=index_retention_stop,
            )
        )

    if DIST_DIR.exists():
        logger.info(f"前端静态文件: {DIST_DIR}")
    else:
        logger.warning(f"前端构建目录不存在: {DIST_DIR}，请先 cd web && npm run build")

    yield

    # 关闭
    if ingest_worker_stop is not None:
        ingest_worker_stop.set()
    if ingest_worker_task is not None:
        ingest_worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await ingest_worker_task
    if index_retention_stop is not None:
        index_retention_stop.set()
    if index_retention_task is not None:
        index_retention_task.cancel()
        with suppress(asyncio.CancelledError):
            await index_retention_task
    logger.info("Facet 服务关闭")


def create_app() -> FastAPI:
    """工厂函数，延迟加载配置"""
    settings = get_config()

    _app = FastAPI(
        title="Facet",
        description="Facet — a local knowledge base powered by RAG",
        version="0.1.0",
        license_info={"name": "Apache-2.0", "identifier": "Apache-2.0"},
        lifespan=lifespan,
    )

    # CORS（前端同源时不需要，但保留给独立前端开发场景）
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册 API 路由
    _app.include_router(health_router)
    _app.include_router(auth_router)
    _app.include_router(system_router)
    _app.include_router(ingest_router)
    _app.include_router(documents_router)
    _app.include_router(knowledge_bases_router)
    _app.include_router(chat_router)
    _app.include_router(vector_router)

    # 托管前端静态文件
    if DIST_DIR.exists():
        _app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="static-assets")

        @_app.get("/{full_path:path}")
        async def serve_spa(request: Request, full_path: str):
            """SPA catch-all：非 API 路径统一返回 index.html"""
            # API 路径不会到这里（已被上面的 router 接管）
            file_path = DIST_DIR / full_path
            try:
                resolved = file_path.resolve()
            except (OSError, RuntimeError):
                resolved = None
            if resolved and os.path.commonpath([str(resolved), str(DIST_DIR_RESOLVED)]) == str(DIST_DIR_RESOLVED) and resolved.is_file():
                return FileResponse(resolved)
            return FileResponse(DIST_DIR / "index.html")

    return _app


app = create_app()
