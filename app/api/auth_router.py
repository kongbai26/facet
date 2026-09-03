"""Browser session authentication routes."""

from __future__ import annotations

import hmac
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from app.api.deps import (
    _get_settings,
    get_auth_credential_store_instance,
    get_knowledge_base_store_instance,
    get_namespace_store_instance,
    get_principal_store_instance,
    get_session_refresh_threshold_seconds,
    get_session_refresh_ttl,
    get_session_store_instance,
    get_tenant_store_instance,
    reset_dependency_cache,
    set_session_cookie,
    verify_auth,
)
from app.auth_service import (
    collect_auth_warnings,
    ensure_admin_credential_from_legacy,
    principal_summary,
)
from app.bootstrap import ensure_default_workspace
from app.config import reset_config
from app.settings.loader import CONFIG_DIR
from app.utils.login_rate_limit import get_login_rate_limiter
from app.utils.security import hash_password, resolve_session_secret, update_env_file, verify_password

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
PASSWORD_MIN_LENGTH = 6


def _error(status_code: int, code: str, message: str, *, headers: dict[str, str] | None = None, **extra):
    detail = {"code": code, "message": message}
    if extra:
        detail.update(extra)
    raise HTTPException(status_code=status_code, detail=detail, headers=headers)


async def _ensure_workspace() -> dict:
    settings = _get_settings()
    return await ensure_default_workspace(
        settings,
        get_tenant_store_instance(),
        get_principal_store_instance(),
        get_namespace_store_instance(),
        get_knowledge_base_store_instance(),
    )


async def _get_admin_principal() -> dict:
    workspace = await _ensure_workspace()
    return workspace["principal"]


async def _get_admin_credential() -> Optional[dict]:
    principal = await _get_admin_principal()
    return await get_auth_credential_store_instance().get_active_password_credential(
        principal["principal_id"]
    )


async def _materialize_legacy_admin_credential() -> Optional[dict]:
    settings = _get_settings()
    principal = await _get_admin_principal()
    await ensure_admin_credential_from_legacy(
        settings,
        principal,
        get_auth_credential_store_instance(),
    )
    return await get_auth_credential_store_instance().get_active_password_credential(
        principal["principal_id"]
    )


async def _setup_required() -> bool:
    return await _materialize_legacy_admin_credential() is None


async def _create_authenticated_session(response: Response, principal: dict, *, method: str) -> dict:
    settings = _get_settings()
    workspace = await _ensure_workspace()
    session = await get_session_store_instance().create_session(
        secret=resolve_session_secret(settings.auth.session_secret),
        ttl_seconds=settings.auth.session_ttl_seconds,
        subject_type="principal",
        subject_id=principal["principal_id"],
    )
    set_session_cookie(settings, response, session["token"])
    return {
        "authenticated": True,
        "mode": method,
        "subject_type": "principal",
        "subject_id": principal["principal_id"],
        "tenant_id": workspace["tenant"]["tenant_id"],
        "principal": principal_summary(principal),
        "expires_at": session["expires_at"],
    }


async def _read_session_principal(request: Request, response: Response) -> Optional[dict]:
    settings = _get_settings()
    token = request.cookies.get(settings.auth.session_cookie_name)
    if not token:
        return None
    session = await get_session_store_instance().get_session(
        token,
        resolve_session_secret(settings.auth.session_secret),
        ttl_seconds=get_session_refresh_ttl(settings),
        refresh_threshold_seconds=get_session_refresh_threshold_seconds(settings),
    )
    if not session:
        return None
    principal = await get_principal_store_instance().get(session["subject_id"])
    if not principal or principal.get("status") != "active":
        await get_session_store_instance().delete_session(
            token,
            resolve_session_secret(settings.auth.session_secret),
        )
        response.delete_cookie(settings.auth.session_cookie_name, path="/")
        return None
    if session.get("refreshed"):
        set_session_cookie(settings, response, token)
    return principal


class SessionRequest(BaseModel):
    password: str = Field(min_length=1, max_length=128)


class InitialProviderConfig(BaseModel):
    """Provider values collected only while the first administrator is created."""

    llm_api_base: str = Field(min_length=1, max_length=2048)
    llm_api_key: str = Field(default="", max_length=4096)
    llm_model_name: str = Field(min_length=1, max_length=256)
    embedding_api_base: str = Field(min_length=1, max_length=2048)
    embedding_api_key: str = Field(default="", max_length=4096)
    embedding_model_name: str = Field(min_length=1, max_length=256)
    reranker_api_base: str = Field(default="", max_length=2048)
    reranker_expected_model: str = Field(default="", max_length=256)

    @field_validator(
        "llm_api_base",
        "llm_api_key",
        "llm_model_name",
        "embedding_api_base",
        "embedding_api_key",
        "embedding_model_name",
        "reranker_api_base",
        "reranker_expected_model",
    )
    @classmethod
    def reject_env_file_control_characters(cls, value: str) -> str:
        normalized = value.strip()
        if any(character in normalized for character in ("\r", "\n", "\x00")):
            raise ValueError("配置值不能包含换行符或控制字符")
        return normalized

    @field_validator("llm_api_base", "embedding_api_base", "reranker_api_base")
    @classmethod
    def validate_api_base(cls, value: str) -> str:
        if not value:
            return value
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("API 地址必须是完整的 http:// 或 https:// URL")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_reranker_pair(self):
        if self.reranker_expected_model and not self.reranker_api_base:
            raise ValueError("填写重排模型名称时，也需要填写重排 API 地址")
        return self


class PasswordSetupRequest(BaseModel):
    password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=128)
    confirm_password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=128)
    provider_config: InitialProviderConfig | None = None

    @model_validator(mode="after")
    def validate_passwords(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


def _persist_initial_provider_config(provider_config: InitialProviderConfig) -> None:
    """Write first-run provider settings to the ignored local environment file."""
    update_env_file(
        CONFIG_DIR / ".env",
        {
            "LLM_API_BASE": provider_config.llm_api_base,
            "LLM_API_KEY": provider_config.llm_api_key,
            "LLM_MODEL_NAME": provider_config.llm_model_name,
            "EMBEDDING_API_BASE": provider_config.embedding_api_base,
            "EMBEDDING_API_KEY": provider_config.embedding_api_key,
            "EMBEDDING_MODEL_NAME": provider_config.embedding_model_name,
            "RERANKER_API_BASE": provider_config.reranker_api_base,
            "RERANKER_EXPECTED_MODEL": provider_config.reranker_expected_model,
        },
    )


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=128)
    confirm_password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=128)

    @model_validator(mode="after")
    def validate_passwords(self):
        if self.new_password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


@router.get("/bootstrap")
async def get_bootstrap(request: Request, response: Response):
    settings = _get_settings()
    workspace = await _ensure_workspace()
    setup_required = settings.auth.enabled and await _setup_required()
    principal = None
    authenticated = False

    if not settings.auth.enabled:
        principal = workspace["principal"]
        authenticated = True
    elif not setup_required:
        principal = await _read_session_principal(request, response)
        authenticated = principal is not None

    return {
        "auth_enabled": settings.auth.enabled,
        "setup_required": setup_required,
        "bootstrap_token_required": bool(setup_required and settings.app.env == "production"),
        "authenticated": authenticated,
        "principal": principal_summary(principal),
        "warnings": collect_auth_warnings(settings, initialized=not setup_required),
    }


@router.post("/setup")
async def setup_password(req: PasswordSetupRequest, request: Request, response: Response):
    settings = _get_settings()
    if not settings.auth.enabled:
        _error(409, "auth_disabled", "当前未启用认证。")
    if not await _setup_required():
        _error(409, "auth_already_initialized", "管理员已初始化。")
    if settings.app.env == "production":
        expected_token = settings.auth.bootstrap_admin_token
        authorization = request.headers.get("Authorization", "")
        provided_token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        if not expected_token:
            _error(
                503,
                "bootstrap_token_required",
                "生产环境首次初始化需要配置 AUTH_BOOTSTRAP_ADMIN_TOKEN。",
            )
        if not provided_token or not hmac.compare_digest(provided_token, expected_token):
            _error(401, "invalid_bootstrap_token", "首次初始化令牌无效。")

    if req.provider_config is not None:
        _persist_initial_provider_config(req.provider_config)
        # The setup request is the only unauthenticated route that may change
        # provider settings. Reload before creating the first session so later
        # requests use the values just saved to the ignored local .env file.
        reset_config()
        reset_dependency_cache()

    principal = await _get_admin_principal()
    await get_auth_credential_store_instance().upsert_password_credential(
        principal["principal_id"],
        hash_password(req.password),
    )
    return await _create_authenticated_session(response, principal, method="setup")


@router.post("/session")
async def create_session(req: SessionRequest, request: Request, response: Response):
    settings = _get_settings()
    if not settings.auth.enabled:
        principal = (await _ensure_workspace())["principal"]
        return await _create_authenticated_session(response, principal, method="disabled")

    if await _setup_required():
        _error(409, "setup_required", "系统尚未初始化，请先设置管理员密码。")

    client_host = request.client.host if request.client else "unknown"
    limiter_key = f"{settings.storage.metadata_db}:{client_host}"
    limiter = get_login_rate_limiter()
    retry_after = await limiter.retry_after_seconds(
        limiter_key,
        limit=settings.auth.login_rate_limit_attempts,
        window_seconds=settings.auth.login_rate_limit_window_seconds,
    )
    if retry_after:
        _error(
            429,
            "login_rate_limited",
            "登录尝试过于频繁，请稍后重试。",
            headers={"Retry-After": str(retry_after)},
            retry_after=retry_after,
        )

    principal = await _get_admin_principal()
    credential = await _get_admin_credential()
    if not credential or not verify_password(req.password, credential.get("password_hash")):
        await limiter.record_failure(
            limiter_key,
            window_seconds=settings.auth.login_rate_limit_window_seconds,
        )
        _error(401, "invalid_password", "密码不正确。")

    await limiter.reset(limiter_key)
    await get_auth_credential_store_instance().touch_last_used(credential["credential_id"])
    return await _create_authenticated_session(response, principal, method="login")


@router.get("/me")
async def get_me(identity: dict = Depends(verify_auth)):
    principal = None
    if identity.get("principal_id"):
        principal = await get_principal_store_instance().get(identity["principal_id"])
    return {
        "authenticated": True,
        **identity,
        "principal": principal_summary(principal),
    }


@router.post("/password")
async def update_password(
    req: PasswordChangeRequest,
    request: Request,
    identity: dict = Depends(verify_auth),
):
    settings = _get_settings()
    if not settings.auth.enabled:
        _error(409, "auth_disabled", "当前未启用认证。")
    if identity.get("method") == "disabled":
        _error(409, "auth_disabled", "当前未启用认证。")

    principal_id = identity.get("principal_id")
    if not principal_id:
        _error(401, "auth_required", "请先登录。")

    credential = await get_auth_credential_store_instance().get_active_password_credential(principal_id)
    if not credential:
        _error(409, "setup_required", "系统尚未初始化，请先设置管理员密码。")
    if not verify_password(req.current_password, credential.get("password_hash")):
        _error(401, "invalid_current_password", "当前密码不正确。")

    await get_auth_credential_store_instance().upsert_password_credential(
        principal_id,
        hash_password(req.new_password),
    )

    token = request.cookies.get(settings.auth.session_cookie_name)
    if token:
        await get_session_store_instance().delete_other_sessions(
            principal_id,
            secret=resolve_session_secret(settings.auth.session_secret),
            keep_token=token,
            subject_type="principal",
        )

    return {
        "updated": True,
        "authenticated": True,
        "subject_type": "principal",
        "subject_id": principal_id,
    }


@router.post("/logout")
async def logout(request: Request, response: Response):
    settings = _get_settings()
    token = request.cookies.get(settings.auth.session_cookie_name)
    if token:
        await get_session_store_instance().delete_session(
            token,
            resolve_session_secret(settings.auth.session_secret),
        )
    response.delete_cookie(settings.auth.session_cookie_name, path="/")
    return {"authenticated": False}
