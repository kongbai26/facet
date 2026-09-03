"""中文分词 + 文本清洗工具"""

from __future__ import annotations

import logging
# 停用词（常见中文停用词）
STOP_WORDS = {
    "的", "了", "是", "在", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看",
    "好", "自己", "这", "他", "她", "它", "们", "我们", "你们", "他们",
    "那", "那些", "这些", "这个", "那个", "什么", "怎么", "为什么",
    "吗", "吧", "呢", "啊", "呀", "哦", "嗯",
    "is", "the", "a", "an", "and", "or", "of", "to", "in", "for",
    "on", "at", "by", "with", "from", "as", "into", "this", "that",
}
_jieba = None


def _get_jieba():
    global _jieba
    if _jieba is None:
        import jieba as jieba_module

        jieba_module.setLogLevel(logging.WARNING)
        _jieba = jieba_module
    return _jieba


def tokenize_mixed(text: str) -> list[str]:
    """中英文混合分词"""
    # 英文转小写
    text = text.lower()
    # jieba 分词
    words = _get_jieba().cut_for_search(text)
    tokens = []
    for w in words:
        w = w.strip()
        if not w or w in STOP_WORDS:
            continue
        # 英文单词保留，中文词保留
        tokens.append(w)
    return tokens
