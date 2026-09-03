"""Reranker provider contract."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score for each input document."""
