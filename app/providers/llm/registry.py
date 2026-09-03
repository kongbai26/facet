"""LLM Provider 注册表"""

from __future__ import annotations

from typing import Dict, Type

from app.providers.llm.base import BaseLLMProvider
from app.providers.llm.mock_impl import MockLLMProvider
from app.providers.llm.openai_impl import OpenAILLMProvider
from app.settings.settings import LLMConfig

_PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAILLMProvider,
    "mock": MockLLMProvider,
}


def resolve_llm_mode(llm_config: LLMConfig) -> str:
    provider_name = (llm_config.provider or "openai").strip().lower()
    if provider_name == "mock":
        return "mock"
    if provider_name != "openai":
        raise ValueError(
            f"未知 llm provider: {provider_name!r}，可选: {list(_PROVIDERS.keys())}"
        )
    if not (llm_config.api_base.strip() and llm_config.model_name.strip()):
        return "mock"
    return "openai"


def get_llm_provider(llm_config: LLMConfig) -> BaseLLMProvider:
    provider_name = resolve_llm_mode(llm_config)
    cls = _PROVIDERS[provider_name]
    return cls(config=llm_config)
