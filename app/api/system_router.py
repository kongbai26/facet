"""System status and diagnostics routes."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app.api.authz import require_admin_access
from app.api.failure_responses import public_failure_response
from app.api.deps import (
    _get_settings,
    get_app_settings_store_instance,
    get_bm25_store_instance,
    get_auth_credential_store_instance,
    get_default_workspace,
    get_embedding_provider_instance,
    get_embedding_provider_for_index_profile,
    get_document_store_instance,
    get_ingest_job_store_instance,
    get_index_profile_store_instance,
    get_knowledge_base_store_instance,
    get_llm_provider_instance,
    get_reranker_instance,
    get_vector_store_instance,
    sync_llm_thinking_preference,
    verify_auth,
)
from app.auth_service import (
    collect_auth_warnings,
    ensure_admin_credential_from_legacy,
    has_legacy_plaintext_password,
    session_secret_ok,
)
from app.pipeline.reindex import auto_reindex_enabled
from app.pipeline.bm25_lifecycle import resolve_bm25_target_collections
from app.prompt_profile import is_local_llm_endpoint, resolve_prompt_profile
from app.rag_scope import get_tenant_rag_collection_name
from app.store.vector_store import load_cached_embedding_dimension
from app.providers.llm.registry import resolve_llm_mode
from app.pipeline.chat_flow import prepare_retrieval_only
from app.pipeline.retrieval_target import resolve_active_retrieval_targets
from app.utils.conversation_turns import foreground_generation_budget, generation_slot
from app.utils.llm_probe import probe_llm_connectivity
from app.utils.model_labels import display_model_name
from app.utils.runtime_errors import GenerationQueueTimeoutError
from app.utils.user_errors import sanitize_diagnostic_detail
from app.workers.ingest_worker import get_ingest_worker_status

router = APIRouter(prefix="/api/v1/system", tags=["system"])
logger = logging.getLogger(__name__)

DIST_DIR = Path(__file__).resolve().parent.parent.parent / "web" / "dist"


def _frontend_built() -> bool:
    return (DIST_DIR / "index.html").exists()


def _llm_mode(settings) -> str:
    return resolve_llm_mode(settings.llm)


def _llm_configured(settings) -> bool:
    cfg = settings.llm
    return bool(cfg.api_base and cfg.model_name)


def _embedding_configured(settings) -> bool:
    cfg = settings.embedding.openai
    return bool(cfg.api_base and cfg.model_name)


def _bm25_collection_ready(bm25_store, collection_name: str) -> bool:
    checker = getattr(bm25_store, "is_collection_ready", None)
    if callable(checker):
        return bool(checker(collection_name))
    return bool(getattr(bm25_store, "is_ready", False))


async def _check_reranker_connectivity(settings) -> dict:
    """Check the optional reranker without making it a hard app dependency."""
    config = settings.retrieval.reranker
    if not config.enabled or config.mode == "off":
        return {
            "key": "reranker_connectivity",
            "status": "warn",
            "code": "reranker_disabled",
            "message": "Reranker 未启用，当前使用旧检索链路。",
        }

    reranker = get_reranker_instance()
    if reranker is None:
        return {
            "key": "reranker_connectivity",
            "status": "error",
            "code": "reranker_unavailable",
            "message": "Reranker 服务不可用。",
            "detail": "reranker provider is not initialized",
        }

    try:
        status_reader = getattr(reranker, "status", None)
        status = status_reader() if callable(status_reader) else {}
        if not status.get("active"):
            ensure_ready = getattr(reranker, "ensure_ready", None)
            if callable(ensure_ready):
                await ensure_ready(force=True)
            status = status_reader() if callable(status_reader) else status
        if not status.get("active"):
            raise RuntimeError(status.get("last_error") or "reranker is inactive")

        # Exercise the same scoring path used by retrieval, not only metadata.
        await reranker.rerank(
            "health check",
            ["This document is used only to verify reranker availability."],
        )
        return {
            "key": "reranker_connectivity",
            "status": "ok",
            "code": "ok",
            "message": "Reranker 服务可连接。",
        }
    except Exception as exc:
        return {
            "key": "reranker_connectivity",
            "status": "error",
            "code": "reranker_connection_failed",
            "message": "Reranker 服务连接失败，当前会回退旧检索链路。",
            "detail": sanitize_diagnostic_detail(str(exc) or exc.__class__.__name__),
        }


async def _queue_runtime_status(settings, ingest_job_store) -> dict:
    """Report whether durable document jobs have an embedded worker to consume them."""
    try:
        queued_count = len(await ingest_job_store.list_jobs(status="queued", limit=1000))
        running_count = len(await ingest_job_store.list_jobs(status="running", limit=1000))
    except Exception as exc:
        return {
            "state": "unknown",
            "queued_count": 0,
            "running_count": 0,
            "last_heartbeat_age_seconds": None,
            "detail": sanitize_diagnostic_detail(str(exc), "无法读取队列状态。"),
        }

    if settings.queue.backend != "db":
        return {
            "state": "disabled",
            "queued_count": queued_count,
            "running_count": running_count,
            "last_heartbeat_age_seconds": None,
            "detail": "当前队列后端不使用内置任务执行器。",
        }
    if not settings.queue.autostart_worker:
        return {
            "state": "disabled",
            "queued_count": queued_count,
            "running_count": running_count,
            "last_heartbeat_age_seconds": None,
            "detail": "内置任务执行器未启用，排队任务不会自动执行。",
        }

    heartbeat = get_ingest_worker_status(
        stale_after_seconds=max(10.0, float(settings.queue.worker_poll_interval_seconds) * 3),
    )
    return {
        "state": "running" if heartbeat["running"] else "stopped",
        "queued_count": queued_count,
        "running_count": running_count,
        "last_heartbeat_age_seconds": heartbeat["last_heartbeat_age_seconds"],
        "detail": (
            "任务执行器正在运行。"
            if heartbeat["running"]
            else "内置任务执行器未运行，排队任务无法自动处理。"
        ),
    }


def _queue_runtime_check(queue_runtime: dict) -> dict:
    """Translate queue liveness into the active-diagnostics response format."""
    state = queue_runtime["state"]
    queued_count = int(queue_runtime["queued_count"])
    running_count = int(queue_runtime["running_count"])
    counts = f"排队 {queued_count} 个，执行中 {running_count} 个。"
    if state == "running":
        return {
            "key": "queue_worker",
            "status": "ok",
            "code": "ok",
            "message": "文档任务执行器正在运行。",
            "detail": counts,
        }
    if state == "disabled":
        return {
            "key": "queue_worker",
            "status": "error" if queued_count else "warn",
            "code": "queue_worker_disabled",
            "message": "文档任务执行器未启用。",
            "detail": f"{queue_runtime['detail']} {counts}",
        }
    if state == "stopped":
        return {
            "key": "queue_worker",
            "status": "error",
            "code": "queue_worker_stopped",
            "message": "文档任务执行器未运行。",
            "detail": f"{queue_runtime['detail']} {counts}",
        }
    return {
        "key": "queue_worker",
        "status": "error",
        "code": "queue_status_unavailable",
        "message": "无法确认文档任务执行器状态。",
        "detail": queue_runtime["detail"],
    }


class PromptProfileRequest(BaseModel):
    profile: Literal["auto", "local", "cloud"]


class LLMThinkingModeRequest(BaseModel):
    mode: Literal["auto", "on", "off"]


class RetrievalDiagnoseRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


def _check_sqlite_writable(db_path: str) -> dict:
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.rollback()
        finally:
            conn.close()
        return {
            "key": "sqlite_write",
            "status": "ok",
            "code": "ok",
            "message": "SQLite 可写。",
        }
    except Exception as exc:
        return {
            "key": "sqlite_write",
            "status": "error",
            "code": "sqlite_write_failed",
            "message": "SQLite 不可写。",
            "detail": sanitize_diagnostic_detail(str(exc), "SQLite 写入检查失败。"),
        }


async def _ensure_system_access(request: Request, response: Response) -> None:
    settings = _get_settings()
    workspace = await get_default_workspace()
    credential_store = get_auth_credential_store_instance()
    await ensure_admin_credential_from_legacy(
        settings,
        workspace["principal"],
        credential_store,
    )
    credential = await credential_store.get_active_password_credential(
        workspace["principal"]["principal_id"]
    )
    initialized = credential is not None
    if not settings.auth.enabled or not initialized:
        return
    try:
        identity = await verify_auth(
            request,
            response,
            authorization=request.headers.get("Authorization"),
        )
        require_admin_access(identity)
    except HTTPException as exc:
        raise exc


@router.get("/status")
async def get_system_status(request: Request, response: Response):
    await _ensure_system_access(request, response)
    settings = _get_settings()
    workspace = await get_default_workspace()
    credential = await get_auth_credential_store_instance().get_active_password_credential(
        workspace["principal"]["principal_id"]
    )
    initialized = credential is not None
    stored_prompt_profile = await get_app_settings_store_instance().get_prompt_profile()
    effective_prompt_profile = resolve_prompt_profile(settings, stored_prompt_profile)
    thinking_state = await sync_llm_thinking_preference()
    warnings = collect_auth_warnings(settings, initialized=initialized)
    llm_mode = _llm_mode(settings)
    # ``/status`` is deliberately passive. It backs setup/settings screens
    # and may be polled while BM25 is rebuilding; issuing real model requests
    # here used to compete with foreground chat and ingest on local servers.
    # The explicit ``POST /checks`` endpoint remains the active diagnostic.
    llm_reachable: bool | None = True if llm_mode == "mock" else None
    if llm_mode == "mock":
        warnings.append("LLM 正在使用模拟模式。")
    elif not _llm_configured(settings):
        warnings.append("LLM 配置缺失。")
    embedding_reachable: bool | None = None
    if not _embedding_configured(settings):
        embedding_reachable = False
        warnings.append("Embedding 配置缺失。")
    if not _frontend_built():
        warnings.append("前端尚未构建。")
    document_store = get_document_store_instance()
    ingest_job_store = get_ingest_job_store_instance()
    queue_runtime = await _queue_runtime_status(settings, ingest_job_store)
    if queue_runtime["state"] == "stopped":
        if queue_runtime["queued_count"]:
            warnings.append(
                f"任务执行器未运行，已有 {queue_runtime['queued_count']} 个任务等待处理。"
            )
        else:
            warnings.append("任务执行器未运行，后续文档任务不会自动处理。")
    elif queue_runtime["state"] == "disabled":
        warnings.append(queue_runtime["detail"])
    elif queue_runtime["state"] == "unknown":
        warnings.append("无法读取任务执行器状态。")

    bm25_ready = False
    bm25_rebuilding = False
    bm25_state = "disabled"
    bm25_target_count = 0
    bm25_ready_count = 0
    if settings.retrieval.hybrid.enabled:
        target_collections = await resolve_bm25_target_collections(
            settings,
            document_store,
            get_index_profile_store_instance(),
        )
        # Compatibility fallback for old metadata that reports ready rows but
        # cannot yet resolve their collection from document records.
        if not target_collections:
            has_ready_documents = await document_store.has_ready_documents()
            if has_ready_documents:
                # Status must not trigger a dimension probe either: resolving
                # a missing cache can issue an embedding request. A legacy
                # collection name without a cached dimension is sufficient
                # for this informational BM25 state only.
                embedding_dimension = load_cached_embedding_dimension(
                    settings.vectorstore.persist_dir,
                    settings.embedding.openai.model_name,
                )
                target_collections.add(get_tenant_rag_collection_name(
                    settings.vectorstore.collection_prefix,
                    settings.embedding.openai.model_name,
                    tenant_slug="default",
                    embedding_dimension=embedding_dimension,
                ))

        bm25_target_count = len(target_collections)
        bm25_store = get_bm25_store_instance()
        ready_collections = {
            collection_name
            for collection_name in target_collections
            if _bm25_collection_ready(bm25_store, collection_name)
        }
        bm25_ready_count = len(ready_collections)
        missing_collections = target_collections - ready_collections
        bm25_ready = bool(target_collections) and not missing_collections

        if not target_collections:
            bm25_state = "empty"
            warnings.append("当前暂无可检索文档，BM25 将在文档就绪后初始化。")
        elif bm25_ready:
            bm25_state = "ready"
        else:
            queued_jobs = await ingest_job_store.list_jobs(status="queued")
            running_jobs = await ingest_job_store.list_jobs(status="running")
            active_jobs = queued_jobs + running_jobs

            def _job_rebuilds_missing_collection(job: dict) -> bool:
                payload = job.get("payload") or {}
                return (
                    job.get("job_type") == "bm25_rebuild"
                    and payload.get("collection_name") in missing_collections
                )

            bm25_rebuilding = any(_job_rebuilds_missing_collection(job) for job in active_jobs)
            source_jobs_active = any(
                (job.get("job_type") or "") in {"ingest", "batch_ingest", "reindex", "batch_reindex"}
                for job in active_jobs
            )
            if bm25_rebuilding:
                bm25_state = "building"
                warnings.append("BM25 索引正在建立中。")
            elif source_jobs_active:
                bm25_state = "waiting_documents"
                warnings.append("BM25 索引正在等待文档处理完成。")
            else:
                bm25_state = "unavailable"
                warnings.append("BM25 索引未就绪且当前没有构建任务，检索将暂时使用向量召回。")
    if not settings.retrieval.exact_match.enabled:
        warnings.append("精确召回增强未启用。")
    reranker_status = {
        "configured": False,
        "active": False,
        "available": False,
        "mode": settings.retrieval.reranker.mode,
        "expected_model": settings.retrieval.reranker.expected_model,
        "model_name": "",
        "last_error": "",
    }
    reranker = get_reranker_instance()
    status_reader = getattr(reranker, "status", None) if reranker is not None else None
    if callable(status_reader):
        reranker_status.update(status_reader())
    reranker_status["model_name"] = display_model_name(reranker_status.get("model_name"))
    reranker_status["expected_model"] = display_model_name(reranker_status.get("expected_model"))
    reranker_status["last_error"] = sanitize_diagnostic_detail(reranker_status.get("last_error"))
    if reranker_status["configured"] and not reranker_status["active"]:
        warnings.append("Reranker 不可用，当前使用旧检索链路。")
    pending_count = await document_store.count_by_statuses({"reindex_queued"})
    running_count = await document_store.count_by_statuses({"reindexing"})
    active_reindex_doc_ids = await ingest_job_store.list_active_doc_ids("reindex")
    docs = await document_store.list_all()
    blocked_count = sum(
        1
        for doc in docs
        if doc.get("status") == "failed"
        and (doc.get("status_reason") or "") == "model_mismatch"
        and doc["doc_id"] not in active_reindex_doc_ids
    )
    reindex_enabled = auto_reindex_enabled(settings)
    if not reindex_enabled:
        warnings.append("Embedding 模型切换自动重建未启用。")
    if blocked_count > 0:
        warnings.append(f"有 {blocked_count} 个文档因模型切换等待重建。")

    return {
        "auth": {
            "initialized": initialized,
            "legacy_password_detected": has_legacy_plaintext_password(settings),
            "session_secret_ok": session_secret_ok(settings),
        },
        "llm": {
            "configured": _llm_configured(settings) or llm_mode == "mock",
            "mode": llm_mode,
            "mock": llm_mode == "mock",
            "reachable": llm_reachable,
            "connectivity_check": "on_demand",
            "thinking": thinking_state,
        },
        "embedding": {
            "configured": _embedding_configured(settings),
            "reachable": embedding_reachable,
            "connectivity_check": "on_demand",
            "current_model": display_model_name(settings.embedding.openai.model_name),
        },
        "retrieval": {
            "hybrid_enabled": settings.retrieval.hybrid.enabled,
            "exact_match_enabled": settings.retrieval.exact_match.enabled,
            "bm25_ready": bm25_ready,
            "bm25_rebuilding": bm25_rebuilding,
            "bm25_state": bm25_state,
            "bm25_target_count": bm25_target_count,
            "bm25_ready_count": bm25_ready_count,
            "reranker": reranker_status,
        },
        "vectorstore": {"mode": settings.vectorstore.mode},
        "database": {"backend": settings.database.backend},
        "queue": {
            "backend": settings.queue.backend,
            "autostart_worker": settings.queue.autostart_worker,
            "worker": queue_runtime,
        },
        "reindex": {
            "enabled": reindex_enabled,
            "pending_count": pending_count,
            "running_count": running_count,
            "blocked_count": blocked_count,
        },
        "prompt": {
            "profile": stored_prompt_profile,
            "effective_profile": effective_prompt_profile,
            "source": "manual" if stored_prompt_profile != "auto" else "auto",
            "local_endpoint_detected": is_local_llm_endpoint(settings.llm.api_base),
        },
        "frontend": {"built": _frontend_built()},
        "warnings": warnings,
    }


@router.put("/prompt-profile")
async def update_prompt_profile(req: PromptProfileRequest, request: Request, response: Response):
    await _ensure_system_access(request, response)
    settings = _get_settings()
    stored_profile = await get_app_settings_store_instance().set_prompt_profile(req.profile)
    effective_profile = resolve_prompt_profile(settings, stored_profile)
    return {
        "profile": stored_profile,
        "effective_profile": effective_profile,
        "source": "manual" if stored_profile != "auto" else "auto",
        "local_endpoint_detected": is_local_llm_endpoint(settings.llm.api_base),
    }


@router.put("/llm-thinking")
async def update_llm_thinking_mode(
    req: LLMThinkingModeRequest,
    request: Request,
    response: Response,
):
    await _ensure_system_access(request, response)
    store = get_app_settings_store_instance()
    await store.set_llm_thinking_mode(req.mode)
    state = await sync_llm_thinking_preference()
    state["source"] = "manual"
    return state


@router.post("/retrieval/diagnose")
async def diagnose_retrieval(req: RetrievalDiagnoseRequest, request: Request, response: Response):
    """Run a body-free, admin-only diagnostic retrieval for the default workspace."""
    await _ensure_system_access(request, response)
    settings = _get_settings()
    workspace = await get_default_workspace()
    tenant = workspace["tenant"]
    try:
        async with foreground_generation_budget(
            settings.chat.max_concurrent_streams,
            queue_wait_timeout_seconds=settings.chat.generation_queue_wait_timeout_seconds,
            turn_timeout_seconds=settings.chat.answer_quality.profile(
                settings.chat.answer_quality.default_mode
            ).turn_timeout_seconds,
        ):
            embedding_provider = get_embedding_provider_instance()
            retrieval_targets = await resolve_active_retrieval_targets(
                settings=settings,
                tenant_id=tenant["tenant_id"],
                tenant_slug=tenant.get("slug"),
                selected_kb_ids=None,
                embedding_provider=embedding_provider,
                knowledge_base_store=get_knowledge_base_store_instance(),
                index_profile_store=get_index_profile_store_instance(),
                profile_embedding_provider_factory=get_embedding_provider_for_index_profile,
            )
            turn = await prepare_retrieval_only(
                req.query,
                [],
                settings,
                get_llm_provider_instance(),
                embedding_provider,
                get_vector_store_instance(),
                get_document_store_instance(),
                get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None,
                get_reranker_instance(),
                tenant_id=tenant["tenant_id"],
                tenant_slug=tenant.get("slug"),
                diagnostics=True,
                retrieval_targets=retrieval_targets,
                answer_quality_mode=settings.chat.answer_quality.default_mode,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("retrieval diagnostic failed")
        return public_failure_response(exc, fallback="检索诊断暂时不可用，请稍后重试。")
    return {
        "decision": turn.decision,
        "reason": turn.reason,
        "retrieval_query": turn.retrieval_query,
        "result_count": len(turn.results),
        "route_plan": turn.route_plan,
        "evidence": turn.evidence,
        "answer_policy": turn.answer_policy,
        "trace": turn.trace or {},
    }


@router.post("/checks")
async def run_system_checks(request: Request, response: Response):
    await _ensure_system_access(request, response)
    settings = _get_settings()
    results: list[dict] = []
    llm_mode = _llm_mode(settings)

    if _embedding_configured(settings):
        try:
            provider = get_embedding_provider_instance()
            async with generation_slot(
                settings.chat.max_concurrent_streams,
                wait_timeout_seconds=1.0,
            ):
                await provider.embed_query("health check")
            results.append({
                "key": "embedding_connectivity",
                "status": "ok",
                "code": "ok",
                "message": "Embedding 服务可连接。",
            })
        except GenerationQueueTimeoutError:
            results.append({
                "key": "embedding_connectivity",
                "status": "warn",
                "code": "model_queue_busy",
                "message": "模型服务正忙，未执行 Embedding 主动检查。",
            })
        except Exception as exc:
            results.append({
                "key": "embedding_connectivity",
                "status": "error",
                "code": "embedding_connection_failed",
                "message": "Embedding 服务连接失败。",
                "detail": sanitize_diagnostic_detail(str(exc), "Embedding 服务检查失败。"),
            })
    else:
        results.append({
            "key": "embedding_connectivity",
            "status": "warn",
            "code": "embedding_not_configured",
            "message": "Embedding 尚未配置。",
        })

    if llm_mode == "mock":
        results.append({
            "key": "llm_connectivity",
            "status": "ok",
            "code": "mock_mode",
            "message": "LLM 当前使用模拟模式。",
        })
    else:
        try:
            provider = get_llm_provider_instance()
            async with generation_slot(
                settings.chat.max_concurrent_streams,
                wait_timeout_seconds=1.0,
            ):
                await probe_llm_connectivity(
                    provider,
                    timeout_seconds=settings.llm.connectivity_timeout_seconds,
                )
            results.append({
                "key": "llm_connectivity",
                "status": "ok",
                "code": "ok",
                "message": "LLM 服务可连接。",
            })
        except GenerationQueueTimeoutError:
            results.append({
                "key": "llm_connectivity",
                "status": "warn",
                "code": "model_queue_busy",
                "message": "模型服务正忙，未执行 LLM 主动检查。",
            })
        except Exception as exc:
            results.append({
                "key": "llm_connectivity",
                "status": "error",
                "code": "llm_connection_failed",
                "message": "LLM 连接失败。",
                "detail": sanitize_diagnostic_detail(str(exc), "LLM 服务检查失败。"),
            })

    results.append(await _check_reranker_connectivity(settings))

    try:
        vector_store = get_vector_store_instance()
        # A diagnostic must not create the default RAG collection merely to
        # prove Chroma is reachable. Prefer a catalog read; old adapters keep
        # their compatible count fallback.
        list_collections = getattr(vector_store, "list_physical_collections", None)
        if not callable(list_collections):
            list_collections = getattr(vector_store, "list_collections", None)
        if callable(list_collections):
            await list_collections()
        else:
            await vector_store.count()
        results.append({
            "key": "vectorstore_access",
            "status": "ok",
            "code": "ok",
            "message": "向量库可访问。",
        })
    except Exception as exc:
        results.append({
            "key": "vectorstore_access",
            "status": "error",
            "code": "vectorstore_unavailable",
            "message": "向量库不可访问。",
            "detail": sanitize_diagnostic_detail(str(exc), "向量库访问检查失败。"),
        })

    results.append(_check_sqlite_writable(settings.storage.metadata_db))
    queue_runtime = await _queue_runtime_status(
        settings,
        get_ingest_job_store_instance(),
    )
    results.append(_queue_runtime_check(queue_runtime))
    return {"checks": results}
