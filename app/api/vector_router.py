"""Standalone Vector API routes — Pinecone / Qdrant 风格的向量接口"""

from __future__ import annotations

import logging
import json
import math
import re
from datetime import datetime, timezone
from typing import Annotated, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.api.deps import (
    _get_settings,
    get_api_key_store_instance,
    get_embedding_provider_instance,
    get_principal_store_instance,
    get_vector_store_instance,
    require_scopes,
    resolve_identity_tenant,
    get_rag_collection_name,
)
from app.rag_scope import is_tenant_scoped_rag_collection
from app.vector_scope import parse_tenant_vector_collection_name, get_tenant_vector_collection_name, visible_vector_collections
from app.utils.user_errors import sanitize_user_error_message

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/vectors", tags=["vectors"])

ALLOWED_SERVICE_SCOPES = frozenset({
    "vectors:read",
    "vectors:write",
    "vectors:admin",
    "rag:read",
    "rag:write",
    "llm:invoke",
    "admin:*",
})
VECTOR_TEXT_MAX_CHARS = 16_000
VECTOR_BATCH_TEXT_MAX_CHARS = 100_000
VECTOR_METADATA_MAX_CHARS = 16_384
COLLECTION_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,127}$")


def _validate_json_object_size(value: dict | None, field_name: str) -> dict | None:
    if value is None:
        return value
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain JSON-compatible values") from exc
    if len(encoded) > VECTOR_METADATA_MAX_CHARS:
        raise ValueError(f"{field_name} exceeds the {VECTOR_METADATA_MAX_CHARS}-character limit")
    return value


def _validate_total_text_size(items: list, field_name: str) -> list:
    total_chars = sum(len(item.text or "") for item in items)
    if total_chars > VECTOR_BATCH_TEXT_MAX_CHARS:
        raise ValueError(
            f"{field_name} text exceeds the {VECTOR_BATCH_TEXT_MAX_CHARS}-character limit"
        )
    return items


# ==============================
# Collection 管理 —— Pydantic Models
# ==============================

class CreateCollectionRequest(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,127}$",
        description="Collection 名称，字母开头，仅含字母数字下划线连字符",
    )
    metric: Literal["cosine", "l2", "ip"] = Field(
        "cosine",
        description="距离度量：cosine（余弦）、l2（欧几里得）、ip（内积）",
    )


class CollectionInfo(BaseModel):
    name: str
    count: int
    metric: str


class ListCollectionsResponse(BaseModel):
    collections: List[CollectionInfo]


class DescribeCollectionResponse(BaseModel):
    name: str
    count: int
    metric: str
    is_protected: bool = False


class DeleteCollectionResponse(BaseModel):
    name: str
    status: str = "deleted"


# ==============================
# 向量操作 —— Pydantic Models
# ==============================

class VectorItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=512)
    text: Optional[str] = Field(
        None,
        min_length=1,
        max_length=VECTOR_TEXT_MAX_CHARS,
        description="文本内容（自动向量化，与 vector 二选一）",
    )
    vector: Optional[List[float]] = Field(
        None, min_length=1, max_length=16384, description="预计算向量（与 text 二选一）",
    )
    metadata: Optional[dict] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_metadata_size(self):
        _validate_json_object_size(self.metadata, "metadata")
        return self


class UpsertRequest(BaseModel):
    vectors: List[VectorItem] = Field(..., min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_total_text_size(self):
        _validate_total_text_size(self.vectors, "vectors")
        return self


class UpsertResponse(BaseModel):
    upserted_count: int


class SearchRequest(BaseModel):
    vector: Optional[List[float]] = Field(
        None, min_length=1, max_length=16384, description="查询向量（与 text 二选一）",
    )
    text: Optional[str] = Field(
        None,
        min_length=1,
        max_length=VECTOR_TEXT_MAX_CHARS,
        description="查询文本（自动向量化，与 vector 二选一）",
    )
    top_k: int = Field(10, ge=1, le=1000)
    filter: Optional[dict] = Field(None, description="元数据过滤（ChromaDB where 语法）")
    score_threshold: Optional[float] = Field(
        None, ge=0.0,
        description="最大距离；仅返回 distance <= 此值（0=完全相同，越小越相近）",
    )
    include_vector: bool = Field(False, description="是否在结果中包含向量")

    @model_validator(mode="after")
    def validate_filter_size(self):
        _validate_json_object_size(self.filter, "filter")
        return self


class SearchResultItem(BaseModel):
    id: str
    score: float = Field(description="Chroma 原始距离（兼容字段名；越小越相近）")
    metadata: Optional[dict] = None
    text: Optional[str] = None
    vector: Optional[List[float]] = None


class SearchResponse(BaseModel):
    results: List[SearchResultItem]


class FetchRequest(BaseModel):
    ids: List[Annotated[str, Field(min_length=1, max_length=512)]] = Field(
        ..., min_length=1, max_length=1000,
    )
    include_vector: bool = False


class FetchResultItem(BaseModel):
    id: str
    vector: Optional[List[float]] = None
    metadata: Optional[dict] = None
    text: Optional[str] = None


class FetchResponse(BaseModel):
    vectors: List[FetchResultItem]


class DeleteRequest(BaseModel):
    ids: Optional[List[Annotated[str, Field(min_length=1, max_length=512)]]] = Field(
        None, max_length=1000,
    )
    filter: Optional[dict] = None

    @model_validator(mode="after")
    def validate_filter_size(self):
        _validate_json_object_size(self.filter, "filter")
        return self


class DeleteResponse(BaseModel):
    deleted_count: int


class UpdateMetadataRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=512)
    metadata: dict = Field(...)

    @model_validator(mode="after")
    def validate_metadata_size(self):
        _validate_json_object_size(self.metadata, "metadata")
        return self


class UpdateMetadataResponse(BaseModel):
    id: str
    status: str = "updated"


# ==============================
# 工具 —— Pydantic Models
# ==============================

class EmbedRequest(BaseModel):
    texts: List[Annotated[str, Field(min_length=1, max_length=VECTOR_TEXT_MAX_CHARS)]] = Field(
        ..., min_length=1, max_length=100,
    )

    @model_validator(mode="after")
    def validate_total_text_size(self):
        total_chars = sum(len(text) for text in self.texts)
        if total_chars > VECTOR_BATCH_TEXT_MAX_CHARS:
            raise ValueError(
                f"texts exceeds the {VECTOR_BATCH_TEXT_MAX_CHARS}-character limit"
            )
        return self


class EmbedResponse(BaseModel):
    vectors: List[List[float]]
    dimension: int


class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    principal_name: Optional[str] = Field(None, min_length=1, max_length=128)
    scopes: List[str] = Field(default_factory=list, max_length=20)
    is_admin: bool = False
    expires_at: Optional[str] = None
    requests_per_minute: Optional[int] = Field(None, ge=1, le=100000)
    daily_quota: Optional[int] = Field(None, ge=1, le=100000000)


class ApiKeyInfo(BaseModel):
    key_id: str
    tenant_id: str
    principal_id: str
    name: str
    scopes: List[str]
    is_admin: bool
    is_active: bool
    requests_per_minute: Optional[int] = None
    daily_quota: Optional[int] = None
    expires_at: Optional[str] = None
    last_used_at: Optional[str] = None
    created_at: str
    updated_at: str


class CreateApiKeyResponse(ApiKeyInfo):
    raw_key: str
    warning: str


class ListApiKeysResponse(BaseModel):
    api_keys: List[ApiKeyInfo]


class UpdateApiKeyRequest(BaseModel):
    scopes: List[str] = Field(default_factory=list, max_length=20)


class DeleteRevokedApiKeysRequest(BaseModel):
    key_ids: Optional[List[str]] = Field(None, max_length=1000)

    @model_validator(mode="after")
    def validate_key_ids(self):
        if self.key_ids is not None and not self.key_ids:
            raise ValueError("key_ids must not be empty when provided")
        if self.key_ids and len(set(self.key_ids)) != len(self.key_ids):
            raise ValueError("key_ids must be unique")
        return self


class UpdateApiKeyResponse(ApiKeyInfo):
    pass


class DeleteApiKeysResponse(BaseModel):
    deleted_count: int


class RevokeApiKeyResponse(BaseModel):
    key_id: str
    status: str = "revoked"


# ==============================
# 辅助函数
# ==============================

def _validate_collection_name(collection_name: str) -> None:
    """Keep path-based collection operations identical to create semantics."""
    # Internal RAG names are intentionally handled by the protection checks
    # that call this helper.  Let those checks return their 403/404 contract
    # instead of exposing a generic path-format error.
    if (collection_name or "").startswith(("__rag_tenant__", "__vector_tenant__")):
        return
    if not COLLECTION_NAME_PATTERN.fullmatch(collection_name or ""):
        raise HTTPException(
            status_code=422,
            detail="Collection name must start with a letter and contain only letters, numbers, '_' or '-'.",
        )

def _check_not_rag_collection(collection_name: str) -> None:
    """禁止对 RAG 系统 collection 执行写/删操作"""
    _validate_collection_name(collection_name)
    if _is_rag_collection(collection_name):
        raise HTTPException(
            status_code=403,
            detail=f"Collection '{collection_name}' is managed by the RAG system and cannot be modified",
        )


def _reject_rag_collection_access(collection_name: str) -> None:
    """Keep managed RAG generations outside the standalone Vector API."""
    _validate_collection_name(collection_name)
    if _is_rag_collection(collection_name):
        # Hiding physical RAG collections prevents an Agent vector key from
        # bypassing tenant-scoped retrieval through the raw vector surface.
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")


def _default_rag_user_collection_name() -> str:
    full_name = get_rag_collection_name()
    prefix = _get_settings().vectorstore.collection_prefix + "_"
    if full_name.startswith(prefix):
        return full_name[len(prefix):]
    return full_name


def _is_rag_collection(collection_name: str) -> bool:
    """判断是否为 RAG 系统 collection"""
    return (
        collection_name == _default_rag_user_collection_name()
        or is_tenant_scoped_rag_collection(collection_name)
    )


async def _current_tenant(identity: dict) -> dict:
    return await resolve_identity_tenant(identity, strict_when_present=True)


async def _tenant_collection_info(collection_name: str, identity: dict) -> dict:
    _validate_collection_name(collection_name)
    if parse_tenant_vector_collection_name(collection_name):
        raise HTTPException(status_code=403, detail="Collection is not available for this tenant")
    tenant = await _current_tenant(identity)
    tenant_slug = tenant.get("slug")
    physical_name = get_tenant_vector_collection_name(
        _get_settings().vectorstore.collection_prefix,
        collection_name,
        tenant_slug=tenant_slug,
    )
    return {
        "tenant": tenant,
        "tenant_slug": tenant_slug,
        "physical_name": physical_name,
        "logical_name": collection_name,
    }


def _is_visible_rag_collection(collection_name: str) -> bool:
    return _is_rag_collection(collection_name)


def _is_rag_managed_collection_info(info: dict) -> bool:
    metadata = info.get("metadata") or {}
    value = metadata.get("rag_managed")
    return value is True or str(value).lower() == "true"


def _api_key_info_payload(record: dict) -> dict:
    return {field_name: record.get(field_name) for field_name in ApiKeyInfo.model_fields}


def _raise_vector_operation_error(exc: Exception, fallback: str) -> None:
    """Turn backend/provider failures into stable Agent-facing errors."""
    logger.exception("standalone vector operation failed")
    status_code = 400 if isinstance(exc, (TypeError, ValueError)) else 502
    detail = sanitize_user_error_message(str(exc), fallback) if status_code == 400 else fallback
    raise HTTPException(
        status_code=status_code,
        detail=detail,
    ) from exc


def _is_collection_not_found_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "not found",
            "does not exist",
            "no such collection",
            "could not find collection",
        )
    )


def _raise_collection_lookup_error(exc: Exception, collection_name: str) -> None:
    if _is_collection_not_found_error(exc):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found") from exc
    _raise_vector_operation_error(exc, "向量数据库暂时不可用，请稍后重试。")


def _validate_service_key_request(scopes: list[str], expires_at: str | None) -> tuple[list[str], str | None]:
    normalized_scopes = list(dict.fromkeys(scope.strip() for scope in scopes if scope.strip()))
    invalid_scopes = sorted(set(normalized_scopes) - ALLOWED_SERVICE_SCOPES)
    if invalid_scopes:
        raise HTTPException(status_code=422, detail=f"Unsupported API key scopes: {', '.join(invalid_scopes)}")
    if not expires_at:
        return normalized_scopes, None
    try:
        parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="expires_at must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise HTTPException(status_code=422, detail="expires_at must include a timezone")
    if parsed <= datetime.now(timezone.utc):
        raise HTTPException(status_code=422, detail="expires_at must be in the future")
    return normalized_scopes, parsed.isoformat()


async def _resolve_vectors(
    items: List[VectorItem],
    embedding_provider,
) -> tuple:
    """
    解析 VectorItem 列表：有 vector 直接用，有 text 批量嵌入，两者皆无报 400。
    返回 (ids, vectors, texts, metadatas)
    """
    ids: List[str] = []
    vectors: List[List[float]] = []
    texts: List[Optional[str]] = []
    metadatas: List[Optional[dict]] = []
    texts_to_embed: List[str] = []
    text_indices: List[int] = []

    for item in items:
        if item.vector is not None and item.text is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Item '{item.id}': provide either 'text' or 'vector', not both",
            )
        ids.append(item.id)
        texts.append(item.text)
        metadatas.append(item.metadata)
        if item.vector is not None:
            vectors.append(item.vector)
        elif item.text is not None:
            vectors.append([])  # 占位，稍后填充
            texts_to_embed.append(item.text)
            text_indices.append(len(vectors) - 1)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Item '{item.id}': must provide either 'vector' or 'text'",
            )

    if texts_to_embed:
        embedded = await embedding_provider.embed_texts(texts_to_embed)
        for idx, vec in zip(text_indices, embedded):
            vectors[idx] = vec

    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=400, detail="Vector ids must be unique within one request")
    expected_dimension = await embedding_provider.dimension()
    for item_id, vector in zip(ids, vectors):
        _validate_vector(item_id, vector, expected_dimension)

    return ids, vectors, texts, metadatas


def _validate_vector(item_id: str, vector: List[float], expected_dimension: int | None) -> None:
    if expected_dimension is not None and len(vector) != int(expected_dimension):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Vector '{item_id}' has dimension {len(vector)}; "
                f"expected {int(expected_dimension)}"
            ),
        )
    if not all(math.isfinite(value) for value in vector):
        raise HTTPException(status_code=400, detail=f"Vector '{item_id}' contains a non-finite value")


def _format_search_results(raw: dict, include_vector: bool) -> List[SearchResultItem]:
    """将 ChromaDB query 原始 dict 转换为 SearchResultItem 列表"""
    results: List[SearchResultItem] = []
    ids_list = raw.get("ids", [[]])
    distances_list = raw.get("distances", [[]])
    metadatas_list = raw.get("metadatas", [[]])
    documents_list = raw.get("documents", [[]])
    embeddings_list = raw.get("embeddings", [[]])

    if not ids_list or not ids_list[0]:
        return results

    for i in range(len(ids_list[0])):
        metadata = None
        if metadatas_list and metadatas_list[0] and i < len(metadatas_list[0]):
            metadata = metadatas_list[0][i]

        text = None
        if documents_list and documents_list[0] and i < len(documents_list[0]):
            text = documents_list[0][i]

        vector = None
        if include_vector and embeddings_list and embeddings_list[0] and i < len(embeddings_list[0]):
            vector = embeddings_list[0][i]

        score = distances_list[0][i] if distances_list and distances_list[0] else 0.0

        results.append(SearchResultItem(
            id=ids_list[0][i],
            score=score,
            metadata=metadata,
            text=text,
            vector=vector,
        ))

    return results


async def _ensure_collection_exists(collection_name: str) -> None:
    """Ensure a standalone collection exists without exposing RAG storage."""
    try:
        store = get_vector_store_instance()
        info = await store.get_collection_info(collection_name=collection_name)
    except Exception as exc:
        _raise_collection_lookup_error(exc, collection_name)
    if _is_rag_managed_collection_info(info):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")


# ==============================
# Collection 管理 Endpoints
# ==============================

@router.post(
    "/collections",
    response_model=dict,
    status_code=201,
)
async def create_collection(
    req: CreateCollectionRequest,
    identity: dict = Depends(require_scopes("vectors:write")),
):
    """创建新的向量 Collection（命名空间）"""
    if _is_rag_collection(req.name):
        raise HTTPException(
            status_code=403,
            detail=f"Collection '{req.name}' is reserved for the RAG system",
        )
    tenant = await _current_tenant(identity)
    tenant_slug = tenant.get("slug")
    settings = _get_settings()
    store = get_vector_store_instance()
    if settings.quota.enabled:
        collections = await store.list_collections()
        visible = visible_vector_collections(collections, tenant_slug)
        standalone_count = sum(
            1 for collection in visible
            if not _is_visible_rag_collection(collection["name"])
            and not _is_rag_managed_collection_info(collection)
        )
        if standalone_count >= settings.quota.default_max_collections:
            raise HTTPException(status_code=429, detail="Collection quota exceeded")
    physical_name = get_tenant_vector_collection_name(
        settings.vectorstore.collection_prefix,
        req.name,
        tenant_slug=tenant_slug,
    )
    try:
        await store.get_collection_info(collection_name=physical_name)
        raise HTTPException(status_code=409, detail=f"Collection '{req.name}' already exists")
    except HTTPException:
        raise
    except Exception as exc:
        if not _is_collection_not_found_error(exc):
            _raise_vector_operation_error(exc, "向量数据库暂时不可用，请稍后重试。")
    try:
        await store.create_collection(
            req.name,
            metric=req.metric,
            collection_name=physical_name,
            logical_name=req.name,
            tenant_slug=tenant_slug or "default",
        )
    except Exception as exc:
        # The preflight check is intentionally followed by a second check in
        # the error path because two Agents may create the same collection at
        # the same time.
        try:
            await store.get_collection_info(collection_name=physical_name)
        except Exception as lookup_exc:
            if not _is_collection_not_found_error(lookup_exc):
                _raise_vector_operation_error(lookup_exc, "向量数据库暂时不可用，请稍后重试。")
            _raise_vector_operation_error(exc, "向量集合创建失败，请稍后重试。")
        raise HTTPException(status_code=409, detail=f"Collection '{req.name}' already exists") from exc
    return {"name": req.name, "metric": req.metric, "status": "created"}


async def _list_collections_impl(identity: dict):
    """列出当前 tenant 可见的 Collection。"""
    try:
        store = get_vector_store_instance()
        collections = await store.list_collections()
    except Exception as exc:
        _raise_vector_operation_error(exc, "向量数据库暂时不可用，请稍后重试。")
    tenant = await _current_tenant(identity)
    tenant_slug = tenant.get("slug")
    visible = visible_vector_collections(collections, tenant_slug)
    filtered = [
        collection
        for collection in visible
        if not _is_visible_rag_collection(collection["name"])
        and not _is_rag_managed_collection_info(collection)
    ]
    return ListCollectionsResponse(collections=[CollectionInfo(**c) for c in filtered])


@router.get(
    "/collections",
    response_model=ListCollectionsResponse,
)
async def list_collections(identity: dict = Depends(require_scopes("vectors:read"))):
    return await _list_collections_impl(identity)


@router.get(
    "/collections/{collection_name}",
    response_model=DescribeCollectionResponse,
)
async def describe_collection(
    collection_name: str,
    identity: dict = Depends(require_scopes("vectors:read")),
):
    """获取指定 Collection 详情"""
    _reject_rag_collection_access(collection_name)
    store = get_vector_store_instance()
    resolved = await _tenant_collection_info(collection_name, identity)
    physical_name = resolved["physical_name"]
    logical_name = resolved["logical_name"]
    try:
        info = await store.get_collection_info(collection_name=physical_name)
    except Exception as exc:
        _raise_collection_lookup_error(exc, collection_name)
    if _is_rag_managed_collection_info(info):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    return DescribeCollectionResponse(
        name=logical_name,
        count=info["count"],
        metric=info["metric"],
        is_protected=False,
    )


@router.delete(
    "/collections/{collection_name}",
    response_model=DeleteCollectionResponse,
)
async def delete_collection(
    collection_name: str,
    identity: dict = Depends(require_scopes("vectors:write")),
):
    """删除整个 Collection 及其中全部向量"""
    _check_not_rag_collection(collection_name)
    resolved = await _tenant_collection_info(collection_name, identity)
    store = get_vector_store_instance()
    try:
        info = await store.get_collection_info(collection_name=resolved["physical_name"])
    except Exception as exc:
        _raise_collection_lookup_error(exc, collection_name)
    # A retired RAG generation may no longer match today's name-based pattern.
    # Metadata remains the durable protection for that historical case.
    if _is_rag_managed_collection_info(info):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    try:
        await store.delete_collection_by_name(collection_name, collection_name=resolved["physical_name"])
    except Exception as exc:
        _raise_vector_operation_error(exc, "向量集合删除失败，请稍后重试。")
    return DeleteCollectionResponse(name=collection_name, status="deleted")


# ==============================
# 向量操作 Endpoints
# ==============================

@router.post(
    "/collections/{collection_name}/upsert",
    response_model=UpsertResponse,
)
async def upsert_vectors(
    collection_name: str,
    req: UpsertRequest,
    identity: dict = Depends(require_scopes("vectors:write")),
):
    """批量 Upsert 向量。每项需提供 text（自动嵌入）或 vector（预计算向量）"""
    _check_not_rag_collection(collection_name)
    resolved = await _tenant_collection_info(collection_name, identity)
    await _ensure_collection_exists(resolved["physical_name"])
    store = get_vector_store_instance()
    embedding_provider = get_embedding_provider_instance()

    try:
        ids, vectors, texts, metadatas = await _resolve_vectors(req.vectors, embedding_provider)
        await store.upsert(
            ids=ids,
            vectors=vectors,
            texts=texts,
            metadatas=metadatas,
            collection_name=resolved["physical_name"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_vector_operation_error(exc, "向量写入失败，请稍后重试。")
    return UpsertResponse(upserted_count=len(ids))


@router.post(
    "/collections/{collection_name}/search",
    response_model=SearchResponse,
)
async def search_vectors(
    collection_name: str,
    req: SearchRequest,
    identity: dict = Depends(require_scopes("vectors:read")),
):
    """搜索相似向量。支持 vector 或 text 查询、metadata 过滤、分数阈值"""
    if req.vector is None and req.text is None:
        raise HTTPException(status_code=400, detail="Must provide either 'vector' or 'text'")
    if req.vector is not None and req.text is not None:
        raise HTTPException(status_code=400, detail="Provide only one of 'vector' or 'text', not both")

    _reject_rag_collection_access(collection_name)
    resolved = await _tenant_collection_info(collection_name, identity)
    await _ensure_collection_exists(resolved["physical_name"])

    store = get_vector_store_instance()
    embedding_provider = get_embedding_provider_instance()

    try:
        query_vector = req.vector
        if req.text is not None:
            query_vector = await embedding_provider.embed_query(req.text)
        _validate_vector("query", query_vector or [], await embedding_provider.dimension())

        raw = await store.query(
            vector=query_vector,
            top_k=req.top_k,
            where=req.filter,
            include_vectors=req.include_vector,
            include_documents=True,
            include_metadatas=True,
            collection_name=resolved["physical_name"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_vector_operation_error(exc, "向量搜索失败，请稍后重试。")

    results = _format_search_results(raw, req.include_vector)

    # ``score`` is the raw Chroma distance for the collection's metric.
    # Chroma represents cosine, l2 and inner-product spaces as distances, so
    # lower values are more similar for all three supported metrics.
    if req.score_threshold is not None:
        results = [r for r in results if r.score <= req.score_threshold]

    return SearchResponse(results=results)


@router.post(
    "/collections/{collection_name}/fetch",
    response_model=FetchResponse,
)
async def fetch_vectors(
    collection_name: str,
    req: FetchRequest,
    identity: dict = Depends(require_scopes("vectors:read")),
):
    """按 ID 列表获取向量"""
    _reject_rag_collection_access(collection_name)
    resolved = await _tenant_collection_info(collection_name, identity)
    await _ensure_collection_exists(resolved["physical_name"])
    store = get_vector_store_instance()

    try:
        raw = await store.get_by_ids(
            ids=req.ids,
            include_vectors=req.include_vector,
            include_documents=True,
            include_metadatas=True,
            collection_name=resolved["physical_name"],
        )
    except Exception as exc:
        _raise_vector_operation_error(exc, "向量读取失败，请稍后重试。")

    vectors: List[FetchResultItem] = []
    id_to_idx = {vid: i for i, vid in enumerate(raw.get("ids", []))}
    for vid in req.ids:
        if vid not in id_to_idx:
            continue
        i = id_to_idx[vid]
        metadata = None
        if raw.get("metadatas") and i < len(raw["metadatas"]):
            metadata = raw["metadatas"][i]
        text = None
        if raw.get("documents") and i < len(raw["documents"]):
            text = raw["documents"][i]
        vector = None
        if req.include_vector and raw.get("embeddings") and i < len(raw.get("embeddings", [])):
            vector = raw["embeddings"][i]
        vectors.append(FetchResultItem(
            id=vid,
            vector=vector,
            metadata=metadata,
            text=text,
        ))

    return FetchResponse(vectors=vectors)


@router.post(
    "/collections/{collection_name}/delete",
    response_model=DeleteResponse,
)
async def delete_vectors(
    collection_name: str,
    req: DeleteRequest,
    identity: dict = Depends(require_scopes("vectors:write")),
):
    """删除向量：按 ID 列表 和/或 metadata filter"""
    if not req.ids and not req.filter:
        raise HTTPException(status_code=400, detail="Must provide 'ids' or 'filter'")
    _check_not_rag_collection(collection_name)
    resolved = await _tenant_collection_info(collection_name, identity)
    await _ensure_collection_exists(resolved["physical_name"])

    store = get_vector_store_instance()
    try:
        before = await store.count(collection_name=resolved["physical_name"])

        if req.ids:
            await store.delete_by_ids(ids=req.ids, collection_name=resolved["physical_name"])

        if req.filter:
            await store.delete(where=req.filter, collection_name=resolved["physical_name"])

        after = await store.count(collection_name=resolved["physical_name"])
    except Exception as exc:
        _raise_vector_operation_error(exc, "向量删除失败，请稍后重试。")
    return DeleteResponse(deleted_count=max(0, before - after))


@router.patch(
    "/collections/{collection_name}/metadata",
    response_model=UpdateMetadataResponse,
)
async def update_vector_metadata(
    collection_name: str,
    req: UpdateMetadataRequest,
    identity: dict = Depends(require_scopes("vectors:write")),
):
    """更新单个向量的元数据"""
    _check_not_rag_collection(collection_name)
    resolved = await _tenant_collection_info(collection_name, identity)
    await _ensure_collection_exists(resolved["physical_name"])
    store = get_vector_store_instance()
    try:
        existing = await store.get_by_ids(
            ids=[req.id],
            include_vectors=False,
            include_documents=False,
            include_metadatas=False,
            collection_name=resolved["physical_name"],
        )
    except Exception as exc:
        _raise_vector_operation_error(exc, "向量读取失败，请稍后重试。")
    if req.id not in set(existing.get("ids") or []):
        raise HTTPException(status_code=404, detail=f"Vector '{req.id}' not found")
    try:
        await store.update_metadata(
            ids=[req.id],
            metadatas=[req.metadata],
            collection_name=resolved["physical_name"],
        )
    except Exception as exc:
        _raise_vector_operation_error(exc, "向量元数据更新失败，请稍后重试。")
    return UpdateMetadataResponse(id=req.id, status="updated")


# ==============================
# 工具 Endpoints
# ==============================

@router.post(
    "/embed",
    response_model=EmbedResponse,
)
async def embed_texts(
    req: EmbedRequest,
    _: dict = Depends(require_scopes("vectors:read")),
):
    """纯文本向量化（不存储），使用已配置的 Embedding Provider"""
    embedding_provider = get_embedding_provider_instance()
    try:
        vectors = await embedding_provider.embed_texts(req.texts)
        dim = await embedding_provider.dimension()
    except Exception as exc:
        _raise_vector_operation_error(exc, "文本向量化失败，请稍后重试。")
    return EmbedResponse(vectors=vectors, dimension=dim)


@router.post(
    "/api-keys",
    response_model=CreateApiKeyResponse,
    status_code=201,
)
async def create_api_key(
    req: CreateApiKeyRequest,
    identity: dict = Depends(require_scopes("vectors:admin")),
):
    tenant = await resolve_identity_tenant(identity, strict_when_present=True)
    scopes, expires_at = _validate_service_key_request(req.scopes, req.expires_at)
    identity_scopes = set(identity.get("scopes") or [])
    identity_is_admin = bool(
        identity.get("method") == "disabled"
        or identity.get("is_admin")
        or "admin:*" in identity_scopes
    )
    if not identity_is_admin and (
        req.is_admin
        or any(scope not in {"vectors:read", "vectors:write", "vectors:admin"} for scope in scopes)
    ):
        raise HTTPException(
            status_code=403,
            detail="Only an administrator can grant privileged API key scopes",
        )
    if not scopes and not req.is_admin:
        scopes = ["vectors:read", "vectors:write"]
    principal_store = get_principal_store_instance()
    principal_name = (req.principal_name or req.name).strip()
    principal = await principal_store.get_by_name(tenant["tenant_id"], "service", principal_name)
    if not principal:
        principal = await principal_store.create(
            tenant["tenant_id"],
            principal_name,
            principal_type="service",
        )
    record = await get_api_key_store_instance().create_key(
        tenant["tenant_id"],
        principal["principal_id"],
        req.name.strip(),
        scopes=scopes,
        is_admin=req.is_admin,
        expires_at=expires_at,
        requests_per_minute=req.requests_per_minute,
        daily_quota=req.daily_quota,
    )
    return CreateApiKeyResponse(
        **_api_key_info_payload(record),
        raw_key=record["raw_key"],
        warning=(
            "Store this key securely. It will not be shown again. "
            "Default non-admin keys only receive vector scopes; add rag scopes explicitly if needed."
        ),
    )


@router.get(
    "/api-keys",
    response_model=ListApiKeysResponse,
)
async def list_api_keys(
    identity: dict = Depends(require_scopes("vectors:admin")),
):
    tenant = await resolve_identity_tenant(identity, strict_when_present=True)
    keys = await get_api_key_store_instance().list_keys(tenant["tenant_id"])
    return ListApiKeysResponse(
        api_keys=[ApiKeyInfo(**_api_key_info_payload(item)) for item in keys]
    )


@router.patch(
    "/api-keys/{key_id}",
    response_model=UpdateApiKeyResponse,
)
async def update_api_key(
    key_id: str,
    req: UpdateApiKeyRequest,
    identity: dict = Depends(require_scopes("vectors:admin")),
):
    tenant = await resolve_identity_tenant(identity, strict_when_present=True)
    store = get_api_key_store_instance()
    key_record = await store.get(key_id)
    if not key_record or key_record["tenant_id"] != tenant["tenant_id"]:
        raise HTTPException(status_code=404, detail="API key not found")
    if key_record.get("is_admin"):
        raise HTTPException(
            status_code=400,
            detail="Administrator keys cannot be narrowed by scopes; revoke and recreate the key instead",
        )

    scopes, _ = _validate_service_key_request(req.scopes, None)
    identity_scopes = set(identity.get("scopes") or [])
    identity_is_admin = bool(
        identity.get("method") == "disabled"
        or identity.get("is_admin")
        or "admin:*" in identity_scopes
    )
    if not identity_is_admin and any(
        scope not in {"vectors:read", "vectors:write", "vectors:admin"}
        for scope in scopes
    ):
        raise HTTPException(
            status_code=403,
            detail="Only an administrator can grant privileged API key scopes",
        )
    if not scopes:
        raise HTTPException(status_code=422, detail="At least one scope is required; revoke the key to disable it")

    updated = await store.update_scopes(key_id, scopes)
    if not updated:
        raise HTTPException(status_code=404, detail="API key not found")
    return UpdateApiKeyResponse(**_api_key_info_payload(updated))


@router.delete(
    "/api-keys/revoked",
    response_model=DeleteApiKeysResponse,
)
async def delete_revoked_api_keys(
    req: DeleteRevokedApiKeysRequest | None = None,
    identity: dict = Depends(require_scopes("vectors:admin")),
):
    tenant = await resolve_identity_tenant(identity, strict_when_present=True)
    key_ids = req.key_ids if req else None
    deleted_count = await get_api_key_store_instance().delete_revoked_keys(
        tenant["tenant_id"],
        key_ids,
    )
    return DeleteApiKeysResponse(deleted_count=deleted_count)


@router.delete(
    "/api-keys/{key_id}/permanent",
    response_model=DeleteApiKeysResponse,
)
async def delete_revoked_api_key(
    key_id: str,
    identity: dict = Depends(require_scopes("vectors:admin")),
):
    tenant = await resolve_identity_tenant(identity, strict_when_present=True)
    key_record = await get_api_key_store_instance().get(key_id)
    if not key_record or key_record["tenant_id"] != tenant["tenant_id"]:
        raise HTTPException(status_code=404, detail="API key not found")
    if key_record.get("is_active"):
        raise HTTPException(status_code=409, detail="Active API keys must be revoked before deletion")
    deleted_count = await get_api_key_store_instance().delete_revoked_keys(
        tenant["tenant_id"],
        [key_id],
    )
    return DeleteApiKeysResponse(deleted_count=deleted_count)


@router.delete(
    "/api-keys/{key_id}",
    response_model=RevokeApiKeyResponse,
)
async def revoke_api_key(
    key_id: str,
    identity: dict = Depends(require_scopes("vectors:admin")),
):
    tenant = await resolve_identity_tenant(identity, strict_when_present=True)
    key_record = await get_api_key_store_instance().get(key_id)
    if not key_record or key_record["tenant_id"] != tenant["tenant_id"]:
        raise HTTPException(status_code=404, detail="API key not found")
    await get_api_key_store_instance().revoke_key(key_id)
    return RevokeApiKeyResponse(key_id=key_id)
