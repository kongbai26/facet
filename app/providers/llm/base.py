"""LLM Provider 抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List


class BaseLLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: List[dict], **kwargs) -> str:
        """非流式，返回完整结果"""

    @abstractmethod
    async def chat_stream(self, messages: List[dict], **kwargs) -> AsyncGenerator[str, None]:
        """流式，逐块返回"""


class ModelBoundLLMProvider(BaseLLMProvider):
    """Bind one configured model to every call in a chat orchestration run."""

    def __init__(self, provider: BaseLLMProvider, model_name: str):
        self.provider = provider
        self.model_name = model_name

    async def chat(self, messages: List[dict], **kwargs) -> str:
        return await self.provider.chat(messages, **{**kwargs, "model_name": self.model_name})

    async def chat_stream(self, messages: List[dict], **kwargs) -> AsyncGenerator[str, None]:
        async for chunk in self.provider.chat_stream(
            messages,
            **{**kwargs, "model_name": self.model_name},
        ):
            yield chunk


def bind_llm_model(provider: BaseLLMProvider, model_name: str) -> BaseLLMProvider:
    """Return a lightweight request-scoped model view of a shared provider."""
    if isinstance(provider, ModelBoundLLMProvider) and provider.model_name == model_name:
        return provider
    return ModelBoundLLMProvider(provider, model_name)
