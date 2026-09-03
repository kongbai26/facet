"""Embedding Provider 抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量 embedding，返回向量列表"""

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """单条查询 embedding"""

    @abstractmethod
    async def dimension(self) -> int:
        """运行时自动检测维度，结果缓存"""

    async def runtime_profile(self) -> dict:
        """返回运行时模型能力；旧 provider 可只提供维度。"""
        return {"dimension": await self.dimension()}

    async def count_tokens(self, text: str) -> int:
        """Count tokens with the embedding model's tokenizer when supported.

        Providers that cannot guarantee this must raise ``NotImplementedError``
        instead of returning a heuristic count as if it were exact.
        """
        raise NotImplementedError("embedding provider does not expose a verified tokenizer")

    async def count_batch_tokens(self, texts: List[str]) -> int:
        """Return the exact token total for one embedding request when known."""
        return sum([await self.count_tokens(text) for text in texts])
