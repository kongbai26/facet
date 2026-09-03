"""切片器注册表"""

from __future__ import annotations

from typing import Optional

from app.chunkers.base import BaseChunker
from app.chunkers.recursive import RecursiveChunker
from app.chunkers.semantic import SemanticChunker
from app.settings.settings import ChunkingConfig


def get_chunker(
    chunking_config: Optional[ChunkingConfig] = None,
    max_tokens: Optional[int] = None,
) -> BaseChunker:
    if chunking_config is None:
        return RecursiveChunker(max_tokens=max_tokens)

    # 语义切片
    if chunking_config.semantic:
        return SemanticChunker(
            chunk_size=chunking_config.chunk_size,
            chunk_overlap=chunking_config.chunk_overlap,
            max_tokens=max_tokens,
            overlap_sentences=chunking_config.overlap_sentences,
        )

    # 递归切片（默认）
    return RecursiveChunker(
        chunk_size=chunking_config.chunk_size,
        chunk_overlap=chunking_config.chunk_overlap,
        separators=chunking_config.separators,
        max_tokens=max_tokens,
    )
