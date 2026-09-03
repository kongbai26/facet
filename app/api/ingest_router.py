"""文档上传路由"""

from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api.authz import require_rag_write
from app.api.deps import (
    _get_settings,
    get_bm25_store_instance,
    get_document_store_instance,
    get_embedding_provider_instance,
    get_ingest_job_store_instance,
    get_vector_store_instance,
    resolve_identity_tenant,
)
from app.bootstrap import ensure_tenant_default_knowledge_base
from app.pipeline.bm25_scheduler import rebuild_bm25_for_collection
from app.settings.settings import AppConfig
from app.pipeline.embedding_readiness import EmbeddingReadinessError, ensure_embedding_ready
from app.pipeline.ingest import ingest_document
from app.pipeline.index_lifecycle import schedule_knowledge_base_reconciliations_after_source_changes
from app.rag_scope import get_tenant_rag_collection_name, resolve_embedding_dimension
from app.store.vector_store import load_cached_embedding_dimension
from app.utils.file_ops import remove_dir_strict
from app.utils.user_errors import sanitize_user_error_message

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
ARCHIVE_UPLOAD_EXTENSIONS = {".docx", ".xlsx"}


@lru_cache(maxsize=8)
def _upload_metadata_stores(metadata_db: str):
    """Use stores bound to the request's active settings/database path.

    This keeps upload resolution correct during tests and controlled runtime
    reconfiguration, where the dependency singleton may still point at the
    previous database path.
    """
    from app.store.knowledge_base_store import KnowledgeBaseStore
    from app.store.namespace_store import NamespaceStore

    return NamespaceStore(metadata_db), KnowledgeBaseStore(metadata_db)


def sanitize_filename(filename: str) -> str:
    """去除路径分隔符和特殊字符，防止路径穿越。"""
    filename = Path(filename).name
    filename = re.sub(r"[^\w\u4e00-\u9fff\-_. ]", "_", filename)
    return filename[:200]


def _validate_upload_extension(filename: str, settings: AppConfig) -> None:
    ext = Path(filename or "").suffix.lower()
    if ext not in settings.storage.allowed_extensions:
        raise HTTPException(400, f"不支持的文件格式: {ext}")


def _validate_archive_safety(file_path: Path, filename: str, settings: AppConfig) -> None:
    """Reject office-document archives with unsafe declared expansion sizes."""
    if file_path.suffix.lower() not in ARCHIVE_UPLOAD_EXTENSIONS:
        return
    try:
        with zipfile.ZipFile(file_path) as archive:
            members = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise HTTPException(400, f"文件格式无效: {filename}") from exc

    if len(members) > settings.storage.max_archive_members:
        raise HTTPException(413, f"压缩文档包含过多文件，最多允许 {settings.storage.max_archive_members} 项")

    max_uncompressed_bytes = settings.storage.max_uncompressed_archive_size_mb * 1024 * 1024
    declared_size = sum(max(0, info.file_size) for info in members)
    if declared_size > max_uncompressed_bytes:
        raise HTTPException(
            413,
            f"压缩文档解压后超过 {settings.storage.max_uncompressed_archive_size_mb}MB 限制",
        )

    for info in members:
        member_path = Path(info.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise HTTPException(400, "压缩文档包含不安全的内部路径")


async def _persist_uploaded_file(
    file: UploadFile,
    save_dir: Path,
    settings: AppConfig,
) -> tuple[str, Path, int, str]:
    """保存上传文件到指定目录，返回 (safe_name, save_path, file_size, content_hash)。"""
    safe_name = sanitize_filename(file.filename or "unnamed")
    save_path = save_dir / safe_name
    save_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = settings.storage.max_file_size_mb * 1024 * 1024
    written = 0
    digest = hashlib.sha256()
    async with aiofiles.open(save_path, "wb") as f:
        while chunk := await file.read(8192):
            written += len(chunk)
            if written > max_bytes:
                await f.close()
                save_path.unlink(missing_ok=True)
                remove_dir_strict(save_dir)
                raise HTTPException(413, f"文件超过 {settings.storage.max_file_size_mb}MB 限制")
            digest.update(chunk)
            await f.write(chunk)

    if written == 0:
        save_path.unlink(missing_ok=True)
        remove_dir_strict(save_dir)
        raise HTTPException(400, "文件为空")

    try:
        _validate_archive_safety(save_path, safe_name, settings)
    except Exception:
        save_path.unlink(missing_ok=True)
        remove_dir_strict(save_dir)
        raise

    return safe_name, save_path, written, digest.hexdigest()


def _remove_dir_best_effort(path: Path, context: str) -> None:
    if not path.exists():
        return
    try:
        remove_dir_strict(path)
    except Exception as exc:
        logger.warning("%s 清理失败: %s", context, exc)


async def _resolve_embedding_dimension_for_upload(settings: AppConfig) -> int | None:
    cached_dimension = load_cached_embedding_dimension(
        settings.vectorstore.persist_dir,
        settings.embedding.openai.model_name,
    )
    if cached_dimension is not None:
        return cached_dimension
    return await resolve_embedding_dimension(get_embedding_provider_instance())


async def _resolve_upload_knowledge_base(
    requested_kb_id: str | None,
    tenant: dict,
    settings: AppConfig,
) -> str:
    """Resolve an upload KB strictly inside the caller's tenant."""
    tenant_id = tenant["tenant_id"]
    namespace_store, knowledge_base_store = _upload_metadata_stores(settings.storage.metadata_db)
    if requested_kb_id:
        knowledge_base = await knowledge_base_store.get(requested_kb_id)
        if not knowledge_base or knowledge_base.get("tenant_id") != tenant_id:
            raise HTTPException(404, "Knowledge base not found")
    else:
        defaults = await ensure_tenant_default_knowledge_base(
            settings,
            tenant,
            namespace_store,
            knowledge_base_store,
        )
        knowledge_base = defaults["knowledge_base"]
    if knowledge_base.get("status") != "active":
        raise HTTPException(409, "Knowledge base is not active")
    return knowledge_base["kb_id"]


@router.post("/upload")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    kb_id: str | None = Form(default=None),
    identity: dict = Depends(require_rag_write),
):
    """上传文档并在后台摄入。"""
    settings = _get_settings()
    _validate_upload_extension(file.filename or "", settings)
    try:
        await ensure_embedding_ready(
            settings,
            provider=get_embedding_provider_instance(),
        )
    except EmbeddingReadinessError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "code": "embedding_unavailable",
                "status": "rejected",
                "message": str(exc),
            },
        )

    temp_doc_id = str(uuid4())
    save_dir = Path(settings.storage.upload_dir) / temp_doc_id
    document_store = None
    doc_id = temp_doc_id
    try:
        safe_name, save_path, written, content_hash = await _persist_uploaded_file(
            file, save_dir, settings
        )

        document_store = get_document_store_instance()
        embedding_model = settings.embedding.openai.model_name
        embedding_dimension = await _resolve_embedding_dimension_for_upload(settings)
        tenant = await resolve_identity_tenant(identity)
        tenant_id = tenant["tenant_id"]
        tenant_slug = tenant.get("slug")
        kb_id = await _resolve_upload_knowledge_base(kb_id, tenant, settings)
        claim = await document_store.claim_upload(
            doc_id=temp_doc_id,
            filename=safe_name,
            file_size=written,
            content_hash=content_hash,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            tenant_id=tenant_id,
            tenant_slug=tenant_slug,
            kb_id=kb_id,
        )

        if claim["action"] == "ready":
            _remove_dir_best_effort(save_dir, f"文档 {safe_name} 临时目录")
            existing = claim["document"]
            return {
                "doc_id": existing["doc_id"],
                "filename": existing["filename"],
                "chunks_count": existing["chunks_count"],
                "message": "文档已存在",
            }

        if claim["action"] == "conflict":
            _remove_dir_best_effort(save_dir, f"文档 {safe_name} 临时目录")
            existing = claim["document"]
            return JSONResponse(
                status_code=409,
                content={
                    "doc_id": existing["doc_id"],
                    "status": existing["status"],
                    "message": "相同内容的文档当前不可重复上传",
                },
            )

        doc_id = temp_doc_id

        if settings.app.env == "production":
            try:
                job = await get_ingest_job_store_instance().create_job(
                    tenant_id,
                    "ingest",
                    doc_id=doc_id,
                    payload={"save_path": str(save_path)},
                )
            except Exception as e:
                logger.exception(f"文档 {doc_id} 创建摄入任务失败")
                await document_store.update_status_if(
                    doc_id,
                    ["processing"],
                    "failed",
                    error_message=f"创建摄入任务失败: {e}",
                    chunks_count=0,
                )
                return JSONResponse(
                    status_code=500,
                    content={
                        "code": "ingest_job_create_failed",
                        "doc_id": doc_id,
                        "status": "failed",
                        "message": "创建摄入任务失败",
                    },
                )
            return JSONResponse(
                status_code=202,
                content={
                    "doc_id": doc_id,
                    "filename": safe_name,
                    "status": "processing",
                    "job_id": job["job_id"],
                    "message": "文档已上传，已加入摄入队列",
                },
            )

        background_tasks.add_task(_ingest_in_background, save_path, doc_id)
        return JSONResponse(
            status_code=202,
            content={
                "doc_id": doc_id,
                "filename": safe_name,
                "status": "processing",
                "message": "文档已上传，正在后台摄入",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        _remove_dir_best_effort(save_dir, f"文档 {doc_id} 上传目录")
        if document_store is not None:
            try:
                current = await document_store.get(doc_id)
                if current and current.get("status") == "processing":
                    await document_store.update_status_if(
                        doc_id,
                        ["processing"],
                        "failed",
                        error_message=sanitize_user_error_message(
                            str(exc),
                            "文档上传失败，请检查配置后重试。",
                        ),
                        chunks_count=0,
                    )
            except Exception as rollback_exc:
                logger.warning("文档 %s 上传异常回滚失败: %s", doc_id, rollback_exc)
        logger.exception("文档 %s 上传失败", doc_id)
        safe_detail = sanitize_user_error_message(
            str(exc),
            "文档上传失败，请检查配置后重试。",
        )
        raise HTTPException(500, f"上传文档失败: {safe_detail}")


@router.post("/batch-upload")
async def batch_upload_documents(
    files: list[UploadFile],
    background_tasks: BackgroundTasks,
    kb_id: str | None = Form(default=None),
    identity: dict = Depends(require_rag_write),
):
    """批量上传文档，后台统一摄入并只重建一次 BM25。"""
    settings = _get_settings()
    if not files:
        raise HTTPException(400, "至少需要上传一个文件")
    if len(files) > settings.storage.max_batch_files:
        raise HTTPException(
            413,
            f"批量上传最多支持 {settings.storage.max_batch_files} 个文件",
        )
    # 先判断文件格式；全部文件都非法时无需初始化外部模型服务。
    valid_files = [
        file
        for file in files
        if Path(file.filename or "").suffix.lower() in settings.storage.allowed_extensions
    ]
    if valid_files:
        try:
            await ensure_embedding_ready(
                settings,
                provider=get_embedding_provider_instance(),
            )
        except EmbeddingReadinessError as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "code": "embedding_unavailable",
                    "status": "rejected",
                    "message": str(exc),
                },
            )
    document_store = get_document_store_instance()
    embedding_model = settings.embedding.openai.model_name
    tenant = await resolve_identity_tenant(identity)
    tenant_id = tenant["tenant_id"]
    tenant_slug = tenant.get("slug")
    kb_id = await _resolve_upload_knowledge_base(kb_id, tenant, settings)

    prepared: list[dict] = []
    ingest_doc_ids: list[str] = []
    ingest_save_paths: list[str] = []
    embedding_dimension: int | None = None
    embedding_dimension_resolved = False

    for file in files:
        try:
            _validate_upload_extension(file.filename or "", settings)
            if not embedding_dimension_resolved:
                embedding_dimension = await _resolve_embedding_dimension_for_upload(settings)
                embedding_dimension_resolved = True
            temp_doc_id = str(uuid4())
            save_dir = Path(settings.storage.upload_dir) / temp_doc_id
            safe_name, save_path, written, content_hash = await _persist_uploaded_file(
                file, save_dir, settings
            )
        except HTTPException as exc:
            prepared.append({
                "filename": file.filename or "unnamed",
                "status": "failed",
                "error": exc.detail,
            })
            continue

        claim = await document_store.claim_upload(
            doc_id=temp_doc_id,
            filename=safe_name,
            file_size=written,
            content_hash=content_hash,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            tenant_id=tenant_id,
            tenant_slug=tenant_slug,
            kb_id=kb_id,
        )

        action = claim["action"]
        if action == "ready":
            _remove_dir_best_effort(save_dir, f"文档 {safe_name} 临时目录")
            existing = claim["document"]
            prepared.append({
                "doc_id": existing["doc_id"],
                "filename": existing["filename"],
                "status": "ready",
                "message": "文档已存在",
            })
            continue

        if action == "conflict":
            _remove_dir_best_effort(save_dir, f"文档 {safe_name} 临时目录")
            existing = claim["document"]
            prepared.append({
                "doc_id": existing["doc_id"],
                "filename": existing["filename"],
                "status": "conflict",
                "message": "相同内容的文档当前不可重复上传",
            })
            continue

        doc_id = temp_doc_id

        prepared.append({
            "doc_id": doc_id,
            "filename": safe_name,
            "status": "processing",
            "message": "文档已上传，等待后台摄入",
        })
        ingest_doc_ids.append(doc_id)
        ingest_save_paths.append(str(save_path))

    batch_job_id = None
    if ingest_doc_ids:
        if settings.app.env == "production":
            try:
                job = await get_ingest_job_store_instance().create_job(
                    tenant_id,
                    "batch_ingest",
                    payload={
                        "doc_ids": ingest_doc_ids,
                        "save_paths": ingest_save_paths,
                        "tenant_id": tenant_id,
                        "tenant_slug": tenant_slug,
                        "to_embedding_model": embedding_model,
                        "to_embedding_dimension": embedding_dimension,
                        "collection_name": get_tenant_rag_collection_name(
                            settings.vectorstore.collection_prefix,
                            embedding_model,
                            tenant_slug=tenant_slug,
                            embedding_dimension=embedding_dimension,
                        ),
                    },
                )
                batch_job_id = job["job_id"]
            except Exception as e:
                logger.exception("批量上传创建摄入任务失败")
                for doc_id in ingest_doc_ids:
                    await document_store.update_status_if(
                        doc_id,
                        ["processing"],
                        "failed",
                        error_message=f"创建批量摄入任务失败: {e}",
                        chunks_count=0,
                    )
                for item in prepared:
                    if item.get("status") == "processing":
                        item["status"] = "failed"
                        item["message"] = "创建批量摄入任务失败"
                return JSONResponse(
                    status_code=500,
                    content={
                        "code": "batch_ingest_job_create_failed",
                        "status": "failed",
                        "message": "创建批量摄入任务失败",
                        "prepared": prepared,
                    },
                )
        else:
            background_tasks.add_task(
                _batch_ingest_in_background,
                [Path(p) for p in ingest_save_paths],
                ingest_doc_ids,
                tenant_id,
                tenant_slug,
            )

    return JSONResponse(
        status_code=202,
        content={
            "batch_job_id": batch_job_id,
            "status": "processing" if ingest_doc_ids else "completed",
            "prepared": prepared,
            "message": "文档已上传，正在后台批量摄入" if ingest_doc_ids else "所有文档已就绪或冲突",
        },
    )


async def _ingest_in_background(save_path: Path, doc_id: str) -> None:
    settings = _get_settings()
    document_store = get_document_store_instance()
    try:
        embedding_provider = get_embedding_provider_instance()
        vector_store = get_vector_store_instance()
        bm25_store = get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None
        await ingest_document(
            save_path,
            doc_id,
            settings,
            embedding_provider,
            vector_store,
            document_store,
            bm25_store,
        )
        document = await document_store.get(doc_id)
        if document:
            await schedule_knowledge_base_reconciliations_after_source_changes(
                documents=[document],
                ingest_job_store=get_ingest_job_store_instance(),
                reason="source_changed",
            )
    except Exception as e:
        logger.exception(f"文档 {doc_id} 后台摄入失败")
        current = await document_store.get(doc_id)
        if current and current.get("status") == "processing":
            await document_store.update_status(
                doc_id,
                "failed",
                error_message=sanitize_user_error_message(
                    str(e),
                    "文档摄入失败，请检查模型配置后重试。",
                ),
                chunks_count=0,
            )


async def _batch_ingest_in_background(
    save_paths: list[Path],
    doc_ids: list[str],
    tenant_id: str,
    tenant_slug: str | None,
) -> None:
    """dev 环境下批量摄入后台任务，全部文档摄入结束后统一重建一次 BM25。"""
    settings = _get_settings()
    document_store = get_document_store_instance()
    embedding_provider = get_embedding_provider_instance()
    vector_store = get_vector_store_instance()
    bm25_store = get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None

    for save_path, doc_id in zip(save_paths, doc_ids):
        try:
            await ingest_document(
                save_path,
                doc_id,
                settings,
                embedding_provider,
                vector_store,
                document_store,
                bm25_store,
                defer_bm25_rebuild=True,
            )
        except Exception as e:
            logger.exception(f"批量摄入文档 {doc_id} 失败")
            current = await document_store.get(doc_id)
            if current and current.get("status") == "processing":
                await document_store.update_status(
                    doc_id,
                    "failed",
                    error_message=sanitize_user_error_message(
                        str(e),
                        "文档摄入失败，请检查模型配置后重试。",
                    ),
                    chunks_count=0,
                )

    if bm25_store and settings.retrieval.hybrid.enabled:
        embedding_dimension = await _resolve_embedding_dimension_for_upload(settings)
        collection_name = get_tenant_rag_collection_name(
            settings.vectorstore.collection_prefix,
            settings.embedding.openai.model_name,
            tenant_slug=tenant_slug,
            embedding_dimension=embedding_dimension,
        )
        await rebuild_bm25_for_collection(collection_name, settings)

    refreshed_documents: list[dict] = []
    for doc_id in doc_ids:
        document = await document_store.get(doc_id)
        if document and document.get("status") == "ready" and document.get("kb_id"):
            refreshed_documents.append(document)
    await schedule_knowledge_base_reconciliations_after_source_changes(
        documents=refreshed_documents,
        ingest_job_store=get_ingest_job_store_instance(),
        reason="source_changed",
    )
