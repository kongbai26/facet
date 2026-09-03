"""递归字符切片器（默认）"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.chunkers.base import BaseChunker, Chunk

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约 1.5 token/字，英文约 1.2 token/word）"""
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en_words = len(text.split()) - cn_chars
    return int(cn_chars * 1.5 + max(en_words, 0) * 1.2)


def conservative_token_upper_bound(text: str, *, reserve_tokens: int = 0) -> int:
    """Return a safe token budget when the model tokenizer is unavailable.

    A model tokeniser cannot produce more ordinary text tokens than the UTF-8
    bytes it consumes: unknown characters are represented by byte fallbacks
    and merged vocabulary entries only reduce that count.  ``reserve_tokens``
    covers model-added special tokens such as BOS/EOS.  The result is a safety
    budget, not a claim to know the model's exact count.
    """
    return len((text or "").encode("utf-8", errors="replace")) + max(0, int(reserve_tokens))


class RecursiveChunker(BaseChunker):
    _merge_separator = "\n\n"

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " "]
        self.max_tokens = max_tokens  # Embedding 模型的 token 限制

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        raw_chunks = self._split_recursive(text, self.separators)
        merged = self._merge_small(raw_chunks)

        # Token 限制检查：超限则二次切片
        if self.max_tokens:
            merged = self._enforce_token_limit(merged)

        return [
            Chunk(text=c, index=i, metadata={})
            for i, c in enumerate(merged)
        ]

    def _enforce_token_limit(self, chunks: List[str]) -> List[str]:
        """确保每个 chunk 不超过 max_tokens，超限则二次切片"""
        result = []
        for chunk in chunks:
            tokens = estimate_tokens(chunk)
            if tokens <= self.max_tokens:
                result.append(chunk)
            else:
                # 二次切片：按比例缩小 chunk_size
                ratio = self.max_tokens / tokens
                new_size = max(int(len(chunk) * ratio * 0.9), 50)  # 留 10% 余量
                sub_chunker = RecursiveChunker(
                    chunk_size=new_size,
                    chunk_overlap=max(self.chunk_overlap // 2, 10),
                    separators=self.separators[1:] if len(self.separators) > 1 else [" "],
                    max_tokens=self.max_tokens,
                )
                sub_chunks = sub_chunker.chunk(chunk)
                result.extend([c.text for c in sub_chunks])
                logger.debug(f"二次切片: {tokens} tokens → {len(sub_chunks)} chunks")
        return result

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        if not sep:
            # 无分隔符，硬切
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size and remaining_seps:
                    chunks.extend(self._split_recursive(part, remaining_seps))
                else:
                    current = part
        if current:
            chunks.append(current)

        return chunks

    def _merge_small(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return []

        merged = [chunks[0]]
        for c in chunks[1:]:
            left = merged[-1].rstrip()
            right = c.lstrip()
            merged_len = len(left) + len(self._merge_separator) + len(right)
            if merged_len <= self.chunk_size:
                merged[-1] = f"{left}{self._merge_separator}{right}"
            else:
                merged.append(c)

        # 添加 overlap（确保不超过 chunk_size）
        if self.chunk_overlap > 0 and len(merged) > 1:
            result = [merged[0]]
            for i in range(1, len(merged)):
                prev = merged[i - 1]
                overlap_text = prev[-self.chunk_overlap:]
                # 如果加上 overlap 会超长，截断 overlap
                max_overlap = self.chunk_size - len(merged[i])
                if max_overlap > 0:
                    overlap_text = overlap_text[-max_overlap:]
                else:
                    overlap_text = ""
                result.append(overlap_text + merged[i])
            return result

        return merged
