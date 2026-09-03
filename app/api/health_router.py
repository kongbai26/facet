"""健康检查路由"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import get_config

router = APIRouter(tags=["system"])


@router.get("/health")
async def health_check():
    """基础健康检查"""
    return {"status": "healthy"}


@router.get("/ready")
async def readiness_check():
    """检查本地持久化目录是否已准备好接收请求。

    这是无网络依赖的 readiness 检查：不主动探测 LLM/embedding/reranker，
    避免健康探针把外部模型服务抖动放大成应用重启风暴。
    """
    settings = get_config()
    checks = {
        "metadata_db_parent": Path(settings.storage.metadata_db).parent.exists(),
        "upload_dir": Path(settings.storage.upload_dir).exists(),
    }
    if settings.vectorstore.mode == "persistent":
        checks["vectorstore_dir"] = Path(settings.vectorstore.persist_dir).exists()
    if settings.retrieval.hybrid.enabled:
        checks["bm25_cache_dir"] = Path(settings.retrieval.hybrid.bm25_cache_dir).exists()

    ready = all(checks.values())
    payload = {"status": "ready" if ready else "not_ready", "checks": checks}
    return JSONResponse(status_code=200 if ready else 503, content=payload)
