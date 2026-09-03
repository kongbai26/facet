"""依赖注入（provider + 认证）"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any, Optional

from fastapi import Depends, Header, HTTPException, Request, Response

from app.auth_service import LEGACY_RAG_ACCESS, browser_session_scopes, session_default_scopes
from app.bootstrap import ensure_default_workspace
from app.config import get_config
from app.providers.embedding.registry import get_embedding_provider
from app.providers.llm.registry import get_llm_provider
from app.providers.llm.thinking import (
    describe_thinking_configuration,
    normalize_thinking_mode,
)
from app.providers.reranker.registry import get_reranker
from app.store.vector_store import get_collection_name, load_cached_embedding_dimension
from app.utils.security import resolve_session_secret

logger = logging.getLogger(__name__)

# 延迟初始化的 provider 单例
_embedding_provider = None
_profile_embedding_providers: dict[str, object] = {}
_llm_provider = None
_vector_store = None
_document_store = None
_bm25_store = None
_conversation_store = None
_session_store = None
_tenant_store = None
_principal_store = None
_namespace_store = None
_knowledge_base_store = None
_api_key_store = None
_ingest_job_store = None
_auth_credential_store = None
_app_settings_store = None
_index_profile_store = None
_reranker = None


def _get_settings():
    return get_config()


def set_session_cookie(settings, response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.auth.session_cookie_name,
        value=token,
        max_age=settings.auth.session_ttl_seconds,
        expires=settings.auth.session_ttl_seconds,
        httponly=True,
        secure=settings.auth.cookie_secure,
        samesite=settings.auth.cookie_samesite,
        path="/",
    )


def get_session_refresh_ttl(settings) -> int | None:
    if settings.auth.session_sliding_expiration_enabled:
        return settings.auth.session_ttl_seconds
    return None


def get_session_refresh_threshold_seconds(settings) -> int | None:
    if not settings.auth.session_sliding_expiration_enabled:
        return None
    ttl_seconds = max(int(settings.auth.session_ttl_seconds), 1)
    refresh_window = max(1, int(ttl_seconds * 0.1))
    refresh_window = min(refresh_window, 3600)
    if ttl_seconds > 1:
        refresh_window = min(refresh_window, ttl_seconds - 1)
    return refresh_window


def get_embedding_provider_instance():
    global _embedding_provider
    if _embedding_provider is None:
        settings = _get_settings()
        _embedding_provider = get_embedding_provider(
            settings.embedding, settings.vectorstore
        )
    return _embedding_provider


def get_embedding_provider_for_index_profile(profile_hash: str, profile: dict[str, Any]):
    """Create a provider pinned to an immutable index profile.

    The application-wide provider follows the current configuration.  It is
    deliberately not reused for an older active profile: vectors from another
    embedding model or endpoint are mathematically incompatible even when
    their dimensions happen to match.
    """
    cached = _profile_embedding_providers.get(profile_hash)
    if cached is not None:
        return cached

    embedding_profile = profile.get("embedding") or {}
    settings = _get_settings()
    embedding_settings = settings.embedding.model_copy(deep=True)
    provider_name = str(embedding_profile.get("provider") or embedding_settings.provider)
    if provider_name != "openai":
        raise ValueError(f"unsupported index embedding provider: {provider_name}")
    embedding_settings.provider = provider_name
    provider_config = embedding_settings.openai
    model_name = str(embedding_profile.get("model") or "").strip()
    endpoint = str(embedding_profile.get("endpoint") or "").strip()
    if not model_name or not endpoint:
        raise ValueError("index profile is missing embedding model or endpoint")
    provider_config.model_name = model_name
    provider_config.api_base = endpoint
    provider_config.max_tokens = int(
        embedding_profile.get("max_input_tokens") or provider_config.max_tokens
    )
    # Querying a historical profile must use the model identity persisted with
    # that profile; discovery is only appropriate while building a new one.
    provider_config.auto_detect_model_name = False
    provider = get_embedding_provider(embedding_settings, settings.vectorstore)
    _profile_embedding_providers[profile_hash] = provider
    return provider


def get_llm_provider_instance():
    global _llm_provider
    if _llm_provider is None:
        settings = _get_settings()
        _llm_provider = get_llm_provider(settings.llm)
    return _llm_provider


def get_reranker_instance():
    global _reranker
    if _reranker is None:
        _reranker = get_reranker(_get_settings().retrieval.reranker)
    return _reranker


def get_vector_store_instance():
    """VectorStore 单例"""
    global _vector_store
    if _vector_store is None:
        from app.store.vector_store import VectorStore
        settings = _get_settings()
        _vector_store = VectorStore(settings.vectorstore, settings.embedding.openai.model_name)
    return _vector_store


def get_document_store_instance():
    """DocumentStore 单例"""
    global _document_store
    if _document_store is None:
        from app.store.document_store import DocumentStore
        settings = _get_settings()
        _document_store = DocumentStore(settings.storage.metadata_db)
    return _document_store


def get_conversation_store_instance():
    """ConversationStore 单例"""
    global _conversation_store
    if _conversation_store is None:
        from app.store.conversation_store import ConversationStore
        settings = _get_settings()
        _conversation_store = ConversationStore(settings.storage.metadata_db)
    return _conversation_store


def get_session_store_instance():
    """SessionStore 单例"""
    global _session_store
    if _session_store is None:
        from app.store.session_store import SessionStore
        settings = _get_settings()
        _session_store = SessionStore(settings.storage.metadata_db)
    return _session_store


def get_bm25_store_instance():
    """BM25Store 单例"""
    global _bm25_store
    if _bm25_store is None:
        from app.store.bm25_store import BM25Store
        settings = _get_settings()
        _bm25_store = BM25Store(
            cache_dir=settings.retrieval.hybrid.bm25_cache_dir,
            lexical_metadata_fields=settings.retrieval.exact_match.lexical_metadata_fields,
        )
    return _bm25_store


def get_tenant_store_instance():
    global _tenant_store
    if _tenant_store is None:
        from app.store.tenant_store import TenantStore

        settings = _get_settings()
        _tenant_store = TenantStore(settings.storage.metadata_db)
    return _tenant_store


def get_principal_store_instance():
    global _principal_store
    if _principal_store is None:
        from app.store.principal_store import PrincipalStore

        settings = _get_settings()
        _principal_store = PrincipalStore(settings.storage.metadata_db)
    return _principal_store


def get_namespace_store_instance():
    global _namespace_store
    if _namespace_store is None:
        from app.store.namespace_store import NamespaceStore

        settings = _get_settings()
        _namespace_store = NamespaceStore(settings.storage.metadata_db)
    return _namespace_store


def get_knowledge_base_store_instance():
    global _knowledge_base_store
    if _knowledge_base_store is None:
        from app.store.knowledge_base_store import KnowledgeBaseStore

        settings = _get_settings()
        _knowledge_base_store = KnowledgeBaseStore(settings.storage.metadata_db)
    return _knowledge_base_store


def get_api_key_store_instance():
    global _api_key_store
    if _api_key_store is None:
        from app.store.api_key_store import ApiKeyStore

        settings = _get_settings()
        _api_key_store = ApiKeyStore(settings.storage.metadata_db)
    return _api_key_store


def get_ingest_job_store_instance():
    global _ingest_job_store
    if _ingest_job_store is None:
        from app.store.ingest_job_store import IngestJobStore

        settings = _get_settings()
        _ingest_job_store = IngestJobStore(settings.storage.metadata_db)
    return _ingest_job_store


def get_auth_credential_store_instance():
    global _auth_credential_store
    if _auth_credential_store is None:
        from app.store.auth_credential_store import AuthCredentialStore

        settings = _get_settings()
        _auth_credential_store = AuthCredentialStore(settings.storage.metadata_db)
    return _auth_credential_store


def get_app_settings_store_instance():
    global _app_settings_store
    if _app_settings_store is None:
        from app.store.app_settings_store import AppSettingsStore

        settings = _get_settings()
        _app_settings_store = AppSettingsStore(settings.storage.metadata_db)
    return _app_settings_store


async def sync_llm_thinking_preference(llm_provider=None) -> dict[str, object]:
    """Apply the persisted global thinking choice to the provider singleton."""
    settings = _get_settings()
    store = get_app_settings_store_instance()
    stored_value = await store.get_value("llm_thinking_mode")
    mode = normalize_thinking_mode(stored_value, default=settings.llm.thinking.mode)
    provider = llm_provider or get_llm_provider_instance()
    setter = getattr(provider, "set_runtime_thinking_mode", None)
    if callable(setter):
        setter(mode)
    state = describe_thinking_configuration(
        settings.llm.thinking,
        settings.llm.model_name,
        mode,
    )
    state["source"] = "manual" if stored_value is not None else "config"
    return state


def get_index_profile_store_instance():
    global _index_profile_store
    if _index_profile_store is None:
        from app.store.index_profile_store import IndexProfileStore

        _index_profile_store = IndexProfileStore(_get_settings().storage.metadata_db)
    return _index_profile_store


def get_rag_collection_name() -> str:
    """返回 RAG 系统使用的 ChromaDB collection 名称（用于 Vector API 保护 RAG 数据）"""
    settings = _get_settings()
    dimension = load_cached_embedding_dimension(
        settings.vectorstore.persist_dir,
        settings.embedding.openai.model_name,
    )
    return get_collection_name(
        settings.vectorstore.collection_prefix,
        settings.embedding.openai.model_name,
        dimension=dimension,
    )


def reset_dependency_cache():
    global _embedding_provider, _llm_provider, _vector_store, _document_store, _bm25_store
    global _conversation_store, _session_store, _tenant_store, _principal_store
    global _namespace_store, _knowledge_base_store, _api_key_store, _ingest_job_store
    global _auth_credential_store, _app_settings_store, _index_profile_store, _reranker
    _embedding_provider = None
    _profile_embedding_providers.clear()
    _llm_provider = None
    _vector_store = None
    _document_store = None
    _bm25_store = None
    _conversation_store = None
    _session_store = None
    _tenant_store = None
    _principal_store = None
    _namespace_store = None
    _knowledge_base_store = None
    _api_key_store = None
    _ingest_job_store = None
    _auth_credential_store = None
    _app_settings_store = None
    _index_profile_store = None
    _reranker = None


async def get_default_workspace() -> dict:
    settings = _get_settings()
    return await ensure_default_workspace(
        settings,
        get_tenant_store_instance(),
        get_principal_store_instance(),
        get_namespace_store_instance(),
        get_knowledge_base_store_instance(),
    )


async def resolve_identity_tenant(
    identity: dict | None = None,
    *,
    strict_when_present: bool = False,
) -> dict:
    tenant_id = (identity or {}).get("tenant_id")
    if tenant_id:
        tenant = await get_tenant_store_instance().get(tenant_id)
        if tenant and tenant.get("status") == "active":
            return tenant
        # An authenticated identity with a tenant binding must never fall
        # back to the default workspace.  That fallback would turn a deleted
        # or disabled API-key/session tenant into accidental default-tenant
        # access on routes that do not need ``strict_when_present``.
        raise HTTPException(status_code=404, detail="Tenant not found")
    workspace = await get_default_workspace()
    if workspace["tenant"].get("status") != "active":
        raise HTTPException(status_code=404, detail="Tenant not found")
    return workspace["tenant"]


def _valid_secret(value: Optional[str], expected: str) -> bool:
    return bool(value) and hmac.compare_digest(value, expected)


def _auth_identity(
    method: str,
    subject_type: str = "workspace",
    subject_id: str = "default",
    *,
    tenant_id: Optional[str] = None,
    principal_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
    key_name: Optional[str] = None,
    scopes: Optional[list[str]] = None,
    is_admin: bool = False,
) -> dict:
    return {
        "method": method,
        "subject_type": subject_type,
        "subject_id": subject_id,
        "tenant_id": tenant_id,
        "principal_id": principal_id or subject_id,
        "api_key_id": api_key_id,
        "key_name": key_name,
        "scopes": scopes or [],
        "is_admin": is_admin,
    }


def _store_auth_identity(request: Request, identity: dict) -> dict:
    request.state.auth_identity = identity
    request.scope["auth_identity"] = identity
    return identity


async def _default_admin_identity(settings, method: str) -> dict:
    workspace = await get_default_workspace()
    principal = workspace["principal"]
    return _auth_identity(
        method,
        "principal",
        principal["principal_id"],
        tenant_id=workspace["tenant"]["tenant_id"],
        principal_id=principal["principal_id"],
        scopes=browser_session_scopes() if principal.get("is_admin") else [],
        is_admin=bool(principal.get("is_admin")),
    )


def _has_scope(identity: dict, scope: str) -> bool:
    scopes = set(identity.get("scopes") or [])
    if identity.get("is_admin"):
        return True
    if "admin:*" in scopes:
        return True
    return scope in scopes


def enforce_scopes(identity: dict, *required_scopes: str) -> dict:
    if identity.get("method") == "disabled":
        return identity
    missing = [scope for scope in required_scopes if not _has_scope(identity, scope)]
    if missing:
        raise HTTPException(status_code=403, detail=f"Missing required scopes: {', '.join(missing)}")
    return identity


def enforce_admin_access(identity: dict) -> dict:
    if identity.get("method") == "disabled":
        return identity
    if identity.get("is_admin"):
        return identity
    if "admin:*" in set(identity.get("scopes") or []):
        return identity
    raise HTTPException(status_code=403, detail="Admin access required")


def require_scopes(*required_scopes: str):
    async def _dependency(identity: dict = Depends(verify_auth)):
        return enforce_scopes(identity, *required_scopes)

    return _dependency


async def verify_auth(
    request: Request,
    response: Response,
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """Verify browser sessions or Authorization bearer tokens."""
    settings = _get_settings()
    if not settings.auth.enabled:
        workspace = await get_default_workspace()
        return _store_auth_identity(
            request,
            _auth_identity(
                "disabled",
                tenant_id=workspace["tenant"]["tenant_id"],
            ),
        )

    bearer_token = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:].strip()
    if bearer_token:
        if settings.auth.bootstrap_admin_token and _valid_secret(bearer_token, settings.auth.bootstrap_admin_token):
            identity = await _default_admin_identity(settings, "bootstrap_admin")
            return _store_auth_identity(request, identity)
        if _valid_secret(bearer_token, settings.auth.bearer_token):
            identity = await _default_admin_identity(settings, "bearer")
            return _store_auth_identity(request, identity)
        key_record = await get_api_key_store_instance().get_key_by_hash(
            hashlib.sha256(bearer_token.encode("utf-8")).hexdigest()
        )
        if key_record:
            principal = await get_principal_store_instance().get(key_record["principal_id"])
            tenant = await get_tenant_store_instance().get(key_record["tenant_id"])
            if not principal or principal.get("status") != "active":
                raise HTTPException(status_code=401, detail="Unauthorized")
            if not tenant or tenant.get("status") != "active":
                raise HTTPException(status_code=401, detail="Unauthorized")
            requests_per_minute = key_record.get("requests_per_minute")
            if requests_per_minute is None and settings.rate_limit.enabled:
                requests_per_minute = settings.rate_limit.default_requests_per_minute
            daily_quota = key_record.get("daily_quota")
            if daily_quota is None and settings.quota.enabled:
                daily_quota = settings.quota.default_daily_quota
            usage = await get_api_key_store_instance().consume_usage(
                key_record["key_id"],
                requests_per_minute=requests_per_minute,
                daily_quota=daily_quota,
            )
            if not usage["allowed"]:
                reason = "rate_limit_exceeded" if usage["reason"] == "requests_per_minute" else "daily_quota_exceeded"
                raise HTTPException(
                    status_code=429,
                    detail={
                        "code": reason,
                        "message": "API key request limit exceeded; retry later.",
                    },
                    headers={"Retry-After": str(usage["retry_after"])},
                )
            await get_api_key_store_instance().touch_last_used(key_record["key_id"])
            identity = _auth_identity(
                "api_key",
                "principal",
                principal["principal_id"],
                tenant_id=tenant["tenant_id"],
                principal_id=principal["principal_id"],
                api_key_id=key_record["key_id"],
                key_name=key_record["name"],
                scopes=(
                    LEGACY_RAG_ACCESS
                    if not (key_record.get("scopes") or []) and not key_record.get("is_admin")
                    else key_record.get("scopes") or []
                ),
                is_admin=bool(key_record.get("is_admin")),
            )
            return _store_auth_identity(request, identity)
        raise HTTPException(
            status_code=401,
            detail={"code": "auth_required", "message": "请先登录。"},
        )

    session_token = request.cookies.get(settings.auth.session_cookie_name)
    if session_token:
        session = await get_session_store_instance().get_session(
            session_token,
            resolve_session_secret(settings.auth.session_secret),
            ttl_seconds=get_session_refresh_ttl(settings),
            refresh_threshold_seconds=get_session_refresh_threshold_seconds(settings),
        )
        if session:
            subject_id = session.get("subject_id", "default")
            principal = await get_principal_store_instance().get(subject_id)
            if not principal or principal.get("status") != "active":
                await get_session_store_instance().delete_session(
                    session_token,
                    resolve_session_secret(settings.auth.session_secret),
                )
                response.delete_cookie(settings.auth.session_cookie_name, path="/")
                raise HTTPException(
                    status_code=401,
                    detail={"code": "auth_required", "message": "登录已失效，请重新登录。"},
                )
            if session.get("refreshed"):
                set_session_cookie(settings, response, session_token)
            return _store_auth_identity(
                request,
                _auth_identity(
                    "session",
                    "principal",
                    subject_id,
                    tenant_id=principal.get("tenant_id"),
                    principal_id=subject_id,
                    scopes=browser_session_scopes() if principal.get("is_admin") else session_default_scopes(),
                    is_admin=bool(principal.get("is_admin")),
                ),
            )

    raise HTTPException(
        status_code=401,
        detail={"code": "auth_required", "message": "请先登录。"},
    )
