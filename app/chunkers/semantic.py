"""语义切片器（按句子边界切片）"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from app.chunkers.base import BaseChunker, Chunk
from app.chunkers.recursive import estimate_tokens

logger = logging.getLogger(__name__)

# 中英文句子终结符
SENTENCE_ENDINGS = re.compile(r'(?<=[。！？.!?；;])\s*')


def split_sentences(text: str) -> List[str]:
    """按句子边界分割文本"""
    # 按句子终结符分割
    parts = SENTENCE_ENDINGS.split(text)
    # 去除空白句子
    return [p.strip() for p in parts if p.strip()]


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        max_tokens: Optional[int] = None,
        overlap_sentences: int = 2,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        # 1. 按句子分割
        sentences = split_sentences(text)
        if not sentences:
            return []

        # 2. 合并句子到 chunk_size
        chunks = self._merge_sentences(sentences)

        # 3. 添加句子级 overlap
        if self.overlap_sentences > 0 and len(chunks) > 1:
            chunks = self._add_sentence_overlap(chunks, sentences)

        # 4. Token 限制检查
        if self.max_tokens:
            chunks = self._enforce_token_limit(chunks)

        return [
            Chunk(text=c, index=i, metadata={})
            for i, c in enumerate(chunks)
        ]

    def _merge_sentences(self, sentences: List[str]) -> List[str]:
        """合并句子到 chunk_size"""
        chunks = []
        current = ""

        for sentence in sentences:
            # 单句超长：回退到字符级切片
            if len(sentence) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                # 对超长句子做字符级切片
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
                continue

            candidate = current + sentence if current else sentence
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """对超长句子做字符级切片"""
        # 按逗号、分号等子句边界切分
        sub_endings = re.compile(r'(?<=[，,；;、])\s*')
        parts = sub_endings.split(sentence)

        chunks = []
        current = ""
        for part in parts:
            if len(part) > self.chunk_size:
                # 子句仍然超长，硬切
                if current:
                    chunks.append(current)
                    current = ""
                for i in range(0, len(part), self.chunk_size):
                    chunks.append(part[i:i + self.chunk_size])
                continue

            candidate = current + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part

        if current:
            chunks.append(current)

        return chunks

    def _add_sentence_overlap(self, chunks: List[str], all_sentences: List[str]) -> List[str]:
        """添加句子级 overlap（取前一个 chunk 的最后 N 个句子）"""
        result = [chunks[0]]

        for i in range(1, len(chunks)):
            # 找到当前 chunk 的第一个句子在 all_sentences 中的位置
            current_start = self._find_sentence_start(chunks[i], all_sentences)
            if current_start is None:
                result.append(chunks[i])
                continue

            # 取前 N 个句子作为 overlap
            overlap_start = max(0, current_start - self.overlap_sentences)
            overlap_text = "".join(all_sentences[overlap_start:current_start])

            # 如果 overlap 太长，截断
            if len(overlap_text) > self.chunk_overlap:
                overlap_text = overlap_text[-self.chunk_overlap:]

            result.append(overlap_text + chunks[i])

        return result

    def _find_sentence_start(self, chunk: str, all_sentences: List[str]) -> Optional[int]:
        """找到 chunk 的第一个句子在 all_sentences 中的位置"""
        chunk_start = chunk[:20]  # 取前 20 个字符作为匹配
        for i, sentence in enumerate(all_sentences):
            if sentence.startswith(chunk_start) or chunk_start in sentence:
                return i
        return None

    def _enforce_token_limit(self, chunks: List[str]) -> List[str]:
        """确保每个 chunk 不超过 max_tokens"""
        result = []
        for chunk in chunks:
            tokens = estimate_tokens(chunk)
            if tokens <= self.max_tokens:
                result.append(chunk)
            else:
                # 二次切片
                ratio = self.max_tokens / tokens
                new_size = max(int(len(chunk) * ratio * 0.9), 50)
                sub_chunker = SemanticChunker(
                    chunk_size=new_size,
                    chunk_overlap=max(self.chunk_overlap // 2, 10),
                    max_tokens=self.max_tokens,
                    overlap_sentences=1,
                )
                sub_chunks = sub_chunker.chunk(chunk)
                result.extend([c.text for c in sub_chunks])
        return result
