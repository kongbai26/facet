"""切片器抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """将文本切分为 chunks"""
