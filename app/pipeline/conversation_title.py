"""Conversation title helpers."""

from __future__ import annotations

import asyncio
import re

_TITLE_WRAP_CHARS = "\"'`[](){}<>“”‘’「」『』《》"

_TITLE_SYSTEM_PROMPT = (
    "你是一个会话标题助手。"
    "请根据用户的第一句话，生成一个简短自然的中文会话标题。"
    "只输出标题本身，不要解释，不要引号，不要序号，不要换行。"
    "标题长度控制在 {max_length} 个字符以内。"
)

_COMPACT_REPLACEMENTS = (
    (r"^(帮我|帮忙|请问|请|麻烦|我想|想请教一下|想请教|想了解一下|想了解)\s*", ""),
    (r"^给这个", ""),
    (r"^给我", ""),
    (r"上线前需要检查哪些", "上线检查"),
    (r"上线前要检查什么", "上线检查"),
    (r"前需要检查哪些", "检查"),
    (r"前要检查什么", "检查"),
    (r"需要检查哪些", "检查"),
    (r"要检查什么", "检查"),
    (r"(如何|怎么|为什么|是否|可否|能否)", ""),
    (r"起一个标题$", "标题"),
)


def build_fallback_title(content: str, max_length: int = 32) -> str:
    title = " ".join(content.strip().split())
    return title[:max_length] or "新对话"


def clean_generated_title(title: str, max_length: int = 32) -> str:
    cleaned = " ".join((title or "").strip().split())
    cleaned = cleaned.strip(_TITLE_WRAP_CHARS)
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_length]


def build_compact_title(content: str, max_length: int = 32) -> str:
    title = " ".join((content or "").strip().split())
    if not title:
        return "新对话"

    for pattern, replacement in _COMPACT_REPLACEMENTS:
        title = re.sub(pattern, replacement, title)

    title = title.replace("配置、备份和回滚步骤", "配置、备份与回滚")
    title = title.replace("配置、备份和回滚", "配置、备份与回滚")
    title = title.replace("检查哪些", "检查")
    title = title.replace("检查配置", "检查：配置")
    title = title.replace("需要提前准备哪些", "准备")
    title = title.replace("哪些人、", "人员、")
    title = title.replace("准备人、", "准备：人员、")
    title = title.replace("脚本和通知流程", "脚本与通知流程")
    title = title.replace("准备人员", "准备：人员")
    title = title.strip("，。！？,.!?:：；; ")
    title = clean_generated_title(title, max_length * 2)

    if len(title) <= max_length:
        return title or "新对话"

    if "检查" in title and "：" not in title:
        prefix, suffix = title.split("检查", 1)
        compact = f"{prefix[:14]}检查"
        suffix = suffix.strip("：:，, ")
        if suffix:
            compact = f"{compact}：{suffix}"
        title = compact

    title = title.replace("步骤", "")
    title = clean_generated_title(title, max_length)
    return title or "新对话"


async def generate_llm_title(content: str, llm_provider, settings) -> str | None:
    raw_content = (content or "").strip()
    if not raw_content or not settings.chat.auto_title_enabled:
        return None

    truncated = raw_content[: settings.chat.auto_title_max_input_length]
    messages = [
        {
            "role": "system",
            "content": _TITLE_SYSTEM_PROMPT.format(max_length=settings.chat.title_max_length),
        },
        {"role": "user", "content": truncated},
    ]
    result = await asyncio.wait_for(
        llm_provider.chat(
            messages,
            temperature=0.2,
            max_tokens=min(64, max(16, settings.chat.title_max_length * 2)),
            allow_reasoning=False,
            thinking_mode="off",
        ),
        timeout=settings.chat.auto_title_timeout_seconds,
    )
    title = clean_generated_title(result, settings.chat.title_max_length)
    if title:
        return title
    compact_title = build_compact_title(raw_content, settings.chat.title_max_length)
    return compact_title or None
