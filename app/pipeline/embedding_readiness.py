"""Embedding 服务可用性校验。

上传和重建索引都依赖 Embedding。入口在创建 processing 文档或排队任务前
完成一次有界探测，避免把一个根本无法执行的任务伪装成“处理中”。
"""

from __future__ import annotations

import asyncio

from app.utils.user_errors import sanitize_user_error_message


class EmbeddingReadinessError(RuntimeError):
    """Embedding 未达到当前索引策略要求。"""


async def ensure_embedding_ready(settings, *, provider=None) -> dict:
    """在写入文档状态前确认 Embedding 服务可用。"""
    config = settings.embedding.openai
    if provider is None:
        from app.api.deps import get_embedding_provider_instance

        try:
            provider = get_embedding_provider_instance()
        except Exception as exc:
            raise EmbeddingReadinessError(
                sanitize_user_error_message(
                    str(exc),
                    "Embedding 服务不可用，请检查服务地址和模型配置。",
                )
            ) from exc
    profile_fn = getattr(provider, "runtime_profile", None)
    if not callable(profile_fn):
        # Keep compatibility with small test doubles and older integrations.
        return {}
    if not str(config.api_base or "").strip():
        raise EmbeddingReadinessError("尚未配置 Embedding 服务地址，暂时无法处理文档。")
    if not str(config.model_name or "").strip():
        raise EmbeddingReadinessError("尚未配置 Embedding 模型，暂时无法处理文档。")

    timeout_seconds = max(
        1.0,
        float(getattr(config, "preflight_timeout_seconds", 15)),
    )
    try:
        profile = await asyncio.wait_for(profile_fn(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise EmbeddingReadinessError(
            f"Embedding 服务在 {int(timeout_seconds)} 秒内没有响应，请检查服务是否启动。"
        ) from exc
    except Exception as exc:
        raise EmbeddingReadinessError(
            sanitize_user_error_message(
                str(exc),
                "Embedding 服务不可用，请检查服务地址和模型配置。",
            )
        ) from exc

    if not isinstance(profile, dict):
        raise EmbeddingReadinessError("Embedding 服务返回的模型能力信息无效。")
    dimension = profile.get("dimension")
    if dimension is not None:
        try:
            if int(dimension) <= 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise EmbeddingReadinessError("Embedding 服务返回了无效的向量维度。") from exc

    if (
        bool(getattr(settings.chunking, "require_exact_tokenizer", False))
        and not bool(profile.get("tokenizer_verified"))
    ):
        raise EmbeddingReadinessError(
            "当前 Embedding 服务没有提供已验证的 tokenizer；请配置 tokenizer 接口，"
            "或关闭“必须精确 tokenizer”后再上传。"
        )
    return profile
