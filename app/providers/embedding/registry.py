"""Embedding Provider 注册表"""

from __future__ import annotations

from typing import Dict, Type

from app.providers.embedding.base import BaseEmbeddingProvider
from app.providers.embedding.openai_impl import OpenAIEmbeddingProvider
from app.settings.settings import EmbeddingConfig, VectorStoreConfig

_PROVIDERS: Dict[str, Type[BaseEmbeddingProvider]] = {
    "openai": OpenAIEmbeddingProvider,
}


def get_embedding_provider(
    embedding_config: EmbeddingConfig,
    vectorstore_config: VectorStoreConfig,
) -> BaseEmbeddingProvider:
    provider_name = embedding_config.provider
    if provider_name not in _PROVIDERS:
        raise ValueError(
            f"未知 embedding provider: {provider_name!r}，可选: {list(_PROVIDERS.keys())}"
        )
    cls = _PROVIDERS[provider_name]
    # 目前只有 openai 实现，取对应配置
    config = getattr(embedding_config, provider_name)
    return cls(config=config, vectorstore_config=vectorstore_config)
