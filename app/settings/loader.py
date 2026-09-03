"""配置加载：yaml + .env 合并"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import dotenv
import yaml

from app.settings.settings import AppConfig
from app.utils.security import (
    generate_session_secret,
    is_password_hash,
    is_placeholder_secret,
    update_env_file,
)

# 项目根目录（loader.py 所在目录的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
logger = logging.getLogger(__name__)
OPENAI_ENDPOINT_SUFFIXES = (
    "/embeddings",
    "/chat",
    "/chat/completions",
    "/completions",
    "/responses",
)
DEFAULT_LLM_MODEL_NAME = AppConfig().llm.model_name
DEFAULT_EMBEDDING_MODEL_NAME = AppConfig().embedding.openai.model_name
ENV_OVERRIDE_KEYS = (
    "APP_ENV",
    "ENABLE_STARTUP_RECOVERY",
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_API_BASE",
    "LLM_MODEL_NAME",
    "EMBEDDING_API_KEY",
    "EMBEDDING_API_BASE",
    "EMBEDDING_MODEL_NAME",
    "RERANKER_API_BASE",
    "RERANKER_EXPECTED_MODEL",
    "AUTH_PASSWORD",
    "AUTH_API_KEY",
    "ADMIN_PASSWORD",
    "AUTH_BEARER_TOKEN",
    "AUTH_BOOTSTRAP_ADMIN_TOKEN",
    "SESSION_SECRET",
    "SESSION_SLIDING_EXPIRATION_ENABLED",
    "DATABASE_BACKEND",
    "DATABASE_SQLITE_PATH",
    "VECTORSTORE_MODE",
)


def _ensure_env_file() -> Path:
    """确保 .env 文件存在，不存在则从 .env.example 复制"""
    env_path = CONFIG_DIR / ".env"
    example_path = CONFIG_DIR / ".env.example"

    if not env_path.exists():
        if example_path.exists():
            shutil.copy(example_path, env_path)
        else:
            env_path.write_text(
                "# 请填入你的模型配置和访问密码\n"
                "LLM_API_KEY=ollama\n"
                "LLM_API_BASE=http://localhost:11434/v1\n"
                "EMBEDDING_API_KEY=not-needed\n"
                "# 只填写 API 根地址，不要带 /embeddings\n"
                "EMBEDDING_API_BASE=https://api.openai.com/v1\n"
                "# 首次启动后，访问 Web 初始化向导设置密码\n"
                "SESSION_SECRET=change-me-to-another-secret\n"
            )

        print()
        print("=" * 60)
        print("  首次启动：请在 Web 初始化向导中连接模型服务并设置管理员密码:")
        print(f"  {env_path}")
        print()
        print("  向导会保存 LLM、Embedding 和可选 Reranker 的本地配置。")
        print("  SESSION_SECRET 会在首次启动时自动生成。")
        print()
        print("  支持的服务:")
        print("    - Ollama 本地: API_BASE=http://localhost:11434/v1, API_KEY=ollama")
        print("    - OpenAI:     API_BASE=https://api.openai.com/v1, API_KEY=sk-xxx")
        print("    - DeepSeek:   API_BASE=https://api.deepseek.com/v1, API_KEY=sk-xxx")
        print("=" * 60)
        print()

    return env_path


def _update_legacy_auth_env(
    env_path: Path,
    *,
    session_secret: str | None = None,
) -> None:
    updates = {
        "SESSION_SECRET": session_secret,
    }
    update_env_file(env_path, updates)


def ensure_session_secret(config: AppConfig) -> bool:
    """Ensure SESSION_SECRET exists and is persisted for browser sessions."""
    env_path = CONFIG_DIR / ".env"
    if not is_placeholder_secret(config.auth.session_secret):
        config.auth.session_secret = config.auth.session_secret.strip()
        return False

    config.auth.session_secret = generate_session_secret()
    _update_legacy_auth_env(env_path, session_secret=config.auth.session_secret)
    return True


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_openai_api_base(value: str | None, field_name: str) -> str:
    """Normalize OpenAI-compatible API base URLs without validating user intent."""
    normalized = (value or "").strip()
    if not normalized:
        return ""
    stripped_suffix = None
    for suffix in OPENAI_ENDPOINT_SUFFIXES:
        if normalized == suffix or normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip("/")
            stripped_suffix = suffix
            break

    if stripped_suffix:
        logger.warning("%s 包含 endpoint %s，已自动规范化为 %s", field_name, stripped_suffix, normalized)
    return normalized


def _fetch_openai_models(api_base: str, timeout_seconds: float = 2.0) -> list[dict]:
    endpoint = api_base.rstrip("/") + "/models"
    request = Request(endpoint, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Auto model detection failed for %s: %s", endpoint, exc)
        return []

    if isinstance(payload, dict):
        models = payload.get("data")
        if isinstance(models, list):
            return [item for item in models if isinstance(item, dict)]
        models = payload.get("models")
        if isinstance(models, list):
            return [item for item in models if isinstance(item, dict)]
    return []


def _model_identifier(model: dict) -> str:
    for key in ("id", "key", "name", "display_name"):
        value = model.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _should_auto_detect_model_name(current_name: str, default_name: str, auto_detect_enabled: bool) -> bool:
    normalized = current_name.strip()
    return auto_detect_enabled and (not normalized or normalized == default_name)


def _auto_detect_openai_model_name(
    api_base: str,
    model_type: str,
    preferred_name: str = "",
) -> str | None:
    models = _fetch_openai_models(api_base)
    if not models:
        return None

    candidates = [model for model in models if model.get("type") == model_type]
    if not candidates:
        # llama.cpp/Ollama 兼容接口通常不返回 type，按模型 id 做保守筛选。
        candidates = models
        if model_type == "embedding":
            embedding_candidates = [
                model for model in models
                if "embed" in _model_identifier(model).lower()
            ]
            if embedding_candidates:
                candidates = embedding_candidates

    if preferred_name:
        preferred = next(
            (model for model in candidates if _model_identifier(model) == preferred_name),
            None,
        )
        if preferred is not None:
            return preferred_name

    loaded = [model for model in candidates if model.get("loaded_instances")]
    if loaded:
        scored = sorted(loaded, key=lambda model: len(model.get("loaded_instances") or []), reverse=True)
        top = scored[0]
        if len(scored) == 1:
            identifier = _model_identifier(top)
            return identifier or None
        next_best = len(scored[1].get("loaded_instances") or [])
        if len(top.get("loaded_instances") or []) > next_best:
            identifier = _model_identifier(top)
            return identifier or None

    if len(candidates) == 1:
        identifier = _model_identifier(candidates[0])
        return identifier or None

    return None


def load_config() -> AppConfig:
    # 1. 读 yaml
    yaml_path = CONFIG_DIR / "config.yaml"
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    raw = raw or {}
    raw.setdefault("app", {})
    raw.setdefault("auth", {})
    raw.setdefault("embedding", {})
    raw["embedding"].setdefault("provider", "openai")
    raw["embedding"].setdefault(raw["embedding"]["provider"], {})
    raw.setdefault("llm", {})
    raw.setdefault("storage", {})
    raw.setdefault("database", {})
    raw.setdefault("vectorstore", {})
    raw.setdefault("rate_limit", {})
    raw.setdefault("queue", {})
    raw.setdefault("quota", {})
    raw.setdefault("ingest", {})
    raw.setdefault("observability", {})
    if "sqlite_path" not in raw["database"] and raw["storage"].get("metadata_db"):
        raw["database"]["sqlite_path"] = raw["storage"]["metadata_db"]

    # 2. 读 .env（不存在则自动创建）
    env_path = _ensure_env_file()
    dotenv_values = dotenv.dotenv_values(env_path)
    # Process environment variables and secret managers may inject settings
    # instead of providing a readable .env file. They intentionally take
    # precedence over local development configuration.
    for key in ENV_OVERRIDE_KEYS:
        if key in os.environ:
            dotenv_values[key] = os.environ[key]

    # 3. .env 覆盖 yaml 对应字段
    if "APP_ENV" in dotenv_values:
        raw["app"]["env"] = dotenv_values["APP_ENV"]
    if "ENABLE_STARTUP_RECOVERY" in dotenv_values:
        raw["app"]["enable_startup_recovery"] = _parse_bool(
            dotenv_values["ENABLE_STARTUP_RECOVERY"],
            True,
        )
    if "LLM_PROVIDER" in dotenv_values:
        raw["llm"]["provider"] = dotenv_values["LLM_PROVIDER"]
    if "LLM_API_KEY" in dotenv_values:
        raw["llm"]["api_key"] = dotenv_values["LLM_API_KEY"]
    if "LLM_API_BASE" in dotenv_values:
        raw["llm"]["api_base"] = dotenv_values["LLM_API_BASE"]
    if "LLM_MODEL_NAME" in dotenv_values:
        raw["llm"]["model_name"] = dotenv_values["LLM_MODEL_NAME"]
    if "EMBEDDING_API_KEY" in dotenv_values:
        provider = raw["embedding"]["provider"]
        raw["embedding"][provider]["api_key"] = dotenv_values["EMBEDDING_API_KEY"]
    if "EMBEDDING_API_BASE" in dotenv_values:
        provider = raw["embedding"]["provider"]
        raw["embedding"][provider]["api_base"] = dotenv_values["EMBEDDING_API_BASE"]
    if "EMBEDDING_MODEL_NAME" in dotenv_values:
        provider = raw["embedding"]["provider"]
        raw["embedding"][provider]["model_name"] = dotenv_values["EMBEDDING_MODEL_NAME"]
    if "RERANKER_API_BASE" in dotenv_values:
        raw.setdefault("retrieval", {}).setdefault("reranker", {})["api_base"] = dotenv_values["RERANKER_API_BASE"]
    if "RERANKER_EXPECTED_MODEL" in dotenv_values:
        raw.setdefault("retrieval", {}).setdefault("reranker", {})["expected_model"] = dotenv_values["RERANKER_EXPECTED_MODEL"]
    auth_password = dotenv_values.get("AUTH_PASSWORD") or dotenv_values.get("AUTH_API_KEY")
    if auth_password is not None:
        raw["auth"]["password"] = auth_password
    if "ADMIN_PASSWORD" in dotenv_values:
        raw["auth"]["admin_password"] = dotenv_values["ADMIN_PASSWORD"]
    if "AUTH_BEARER_TOKEN" in dotenv_values:
        raw["auth"]["bearer_token"] = dotenv_values["AUTH_BEARER_TOKEN"]
    if "AUTH_BOOTSTRAP_ADMIN_TOKEN" in dotenv_values:
        raw["auth"]["bootstrap_admin_token"] = dotenv_values["AUTH_BOOTSTRAP_ADMIN_TOKEN"]
    if "SESSION_SECRET" in dotenv_values:
        raw["auth"]["session_secret"] = dotenv_values["SESSION_SECRET"]
    if "SESSION_SLIDING_EXPIRATION_ENABLED" in dotenv_values:
        raw["auth"]["session_sliding_expiration_enabled"] = _parse_bool(
            dotenv_values["SESSION_SLIDING_EXPIRATION_ENABLED"],
            True,
        )
    if "DATABASE_BACKEND" in dotenv_values:
        raw["database"]["backend"] = dotenv_values["DATABASE_BACKEND"]
    if "DATABASE_SQLITE_PATH" in dotenv_values:
        raw["database"]["sqlite_path"] = dotenv_values["DATABASE_SQLITE_PATH"]
    if "VECTORSTORE_MODE" in dotenv_values:
        raw["vectorstore"]["mode"] = dotenv_values["VECTORSTORE_MODE"]
    # 4. 构建 Pydantic 模型
    config = AppConfig(**raw)
    unsupported = []
    if config.app.env not in {"development", "production"}:
        unsupported.append("app.env 只能是 development 或 production")
    if config.app.env == "production" and "*" in config.server.cors_origins:
        unsupported.append("生产环境不能将 server.cors_origins 配置为通配符 '*'")
    if config.database.backend != "sqlite":
        unsupported.append(
            f"database.backend={config.database.backend!r}（当前只实现 sqlite）"
        )
    if config.vectorstore.backend != "chroma" or config.vectorstore.mode != "persistent":
        unsupported.append(
            "vectorstore 当前只实现 backend=chroma、mode=persistent"
        )
    if config.queue.backend != "db":
        unsupported.append("queue.backend 当前只实现 db")
    if config.rate_limit.enabled and config.rate_limit.backend != "sqlite":
        unsupported.append("rate_limit.enabled=true 时当前只实现 backend=sqlite")
    if config.llm.max_tokens > config.llm.context_window:
        unsupported.append("llm.max_tokens 不能大于 llm.context_window")
    if unsupported:
        raise ValueError("配置包含未实现的运行后端：" + "；".join(unsupported))
    config.llm.api_base = _normalize_openai_api_base(config.llm.api_base, "LLM_API_BASE")
    if _should_auto_detect_model_name(config.llm.model_name, DEFAULT_LLM_MODEL_NAME, config.llm.auto_detect_model_name):
        detected = _auto_detect_openai_model_name(config.llm.api_base, "llm")
        if detected:
            logger.info("Auto-detected LLM model name: %s", detected)
            config.llm.model_name = detected
        else:
            logger.warning("LLM 模型名称自动识别失败，请手动配置 llm.model_name。")
    if config.embedding.provider == "openai":
        config.embedding.openai.api_base = _normalize_openai_api_base(
            config.embedding.openai.api_base,
            "EMBEDDING_API_BASE",
        )
        # A configured embedding model is an explicit deployment contract.
        # Do not synchronously contact /models merely to re-confirm it during
        # config loading: this path runs before request handling and an
        # unreachable model server used to stall unrelated pages and startup.
        # Discovery remains available when the name is empty or still the
        # package default, matching the LLM behavior above.
        if _should_auto_detect_model_name(
            config.embedding.openai.model_name,
            DEFAULT_EMBEDDING_MODEL_NAME,
            config.embedding.openai.auto_detect_model_name,
        ):
            detected = _auto_detect_openai_model_name(
                config.embedding.openai.api_base,
                "embedding",
                preferred_name=config.embedding.openai.model_name.strip(),
            )
            if detected:
                if detected != config.embedding.openai.model_name:
                    logger.info(
                        "Embedding 服务未提供配置模型，自动切换为实际模型: %s",
                        detected,
                    )
                config.embedding.openai.model_name = detected
            else:
                logger.warning("Embedding 模型名称自动识别失败，请手动配置 embedding.openai.model_name。")

    if not config.auth.password_hash and is_password_hash(config.auth.password):
        config.auth.password_hash = config.auth.password.strip()
        config.auth.password = ""
    if not config.auth.password_hash and is_password_hash(config.auth.admin_password):
        config.auth.password_hash = config.auth.admin_password.strip()
        config.auth.admin_password = ""
    ensure_session_secret(config)

    # 现有代码仍读取 storage.metadata_db，先与 database.sqlite_path 对齐，避免大面积改动。
    if config.database.backend == "sqlite":
        if config.database.sqlite_path:
            config.storage.metadata_db = config.database.sqlite_path
        else:
            config.database.sqlite_path = config.storage.metadata_db
    elif not config.database.sqlite_path:
        config.database.sqlite_path = config.storage.metadata_db

    # 5. 确保数据目录存在
    Path(config.storage.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(config.vectorstore.persist_dir).mkdir(parents=True, exist_ok=True)
    Path(config.storage.metadata_db).parent.mkdir(parents=True, exist_ok=True)

    return config
