"""Reranker provider factory."""

from __future__ import annotations

from app.providers.reranker.http_impl import HttpReranker
from app.providers.reranker.runtime import RerankerRuntime
from app.settings.settings import RerankerConfig


def get_reranker(config: RerankerConfig):
    if not config.enabled or config.mode == "off":
        return None
    if not config.api_base:
        return RerankerRuntime(
            config,
            None,
            configuration_error="reranker is enabled but reranker.api_base is empty",
        )
    if config.provider != "http":
        return RerankerRuntime(
            config,
            None,
            configuration_error=f"unsupported reranker provider: {config.provider}",
        )
    return RerankerRuntime(config, HttpReranker(config))
