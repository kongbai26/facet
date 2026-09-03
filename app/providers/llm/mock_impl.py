"""Deterministic mock LLM used for local demos."""

from __future__ import annotations

from typing import AsyncGenerator, List

from app.providers.llm.base import BaseLLMProvider
from app.settings.settings import LLMConfig


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config

    def _summarize_prompt(self, messages: List[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = str(message.get("content") or "").strip()
                if content:
                    return content[:120]
        return "你的问题"

    async def chat(self, messages: List[dict], **kwargs) -> str:
        prompt = self._summarize_prompt(messages)
        return (
            "这是一个模拟回答："
            f"我现在没有连接真实 LLM，所以先用占位响应帮你确认流程。"
            f"你刚才问的是「{prompt}」。"
        )

    async def chat_stream(self, messages: List[dict], **kwargs) -> AsyncGenerator[str, None]:
        answer = await self.chat(messages, **kwargs)
        step = 24
        for index in range(0, len(answer), step):
            yield answer[index : index + step]
