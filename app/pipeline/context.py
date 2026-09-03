"""上下文管理：token 预算控制"""

from __future__ import annotations

import logging
from typing import List

from app.chunkers.recursive import estimate_tokens

logger = logging.getLogger(__name__)


def select_chunks_by_budget(
    chunks: List[dict],
    max_context_tokens: int,
    reserved_tokens: int = 512,
) -> List[dict]:
    """
    按 token 预算选择 chunk

    Args:
        chunks: 检索结果列表，每项含 text, score, metadata
        max_context_tokens: 模型最大上下文 token 数
        reserved_tokens: 预留给 prompt + 生成的空间

    Returns:
        筛选后的 chunk 列表
    """
    budget = max_context_tokens - reserved_tokens
    if budget <= 0:
        logger.warning(f"token 预算不足: max={max_context_tokens}, reserved={reserved_tokens}")
        return []

    selected = []
    used_tokens = 0

    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk["text"])
        if used_tokens + chunk_tokens > budget:
            # 尝试截断当前 chunk
            remaining = budget - used_tokens
            if remaining > 100:  # 至少保留 100 token
                # 粗略截断
                ratio = remaining / chunk_tokens
                truncated_text = chunk["text"][:int(len(chunk["text"]) * ratio)]
                selected.append({**chunk, "text": truncated_text + "..."})
                used_tokens += remaining
            break
        selected.append(chunk)
        used_tokens += chunk_tokens

    logger.info(
        f"上下文选择: {len(chunks)} → {len(selected)} chunks, "
        f"使用 {used_tokens}/{budget} tokens"
    )
    return selected
