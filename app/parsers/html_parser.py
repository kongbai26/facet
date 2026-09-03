"""HTML 解析器（trafilatura）"""

from __future__ import annotations

import logging
from html.parser import HTMLParser
from pathlib import Path
from typing import Generator

from lxml import etree, html as lxml_html
import trafilatura

from app.parsers.base import BaseParser, ParserPrelude, StructuredBlock

logger = logging.getLogger(__name__)
_HEADING_TAGS = {f"h{i}" for i in range(1, 7)}
_BLOCK_TAGS = _HEADING_TAGS | {"p", "li", "blockquote", "pre", "code", "table"}
_CONTAINER_TAGS = {"body", "main", "article", "section", "div", "nav", "aside"}
_NOISE_TAGS = {"nav", "aside", "footer", "form", "script", "style", "noscript"}
_NOISE_MARKERS = ("toc", "table-of-contents", "sidebar", "breadcrumb", "menu", "pagination", "footer", "nav")


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _tag_name(node) -> str:
    tag = getattr(node, "tag", "")
    return tag.lower() if isinstance(tag, str) else ""


def _has_block_descendant(node) -> bool:
    for descendant in node.iterdescendants():
        if _tag_name(descendant) in _BLOCK_TAGS:
            return True
    return False


def _text_from_node(node) -> str:
    return _normalize_text(" ".join(node.itertext()))


def _node_anchor(node) -> str:
    for attr in ("id", "name"):
        value = getattr(node, "attrib", {}).get(attr)
        if value:
            return str(value).strip()
    return ""


def _table_rows(node) -> tuple[list[str], list[str]]:
    headers: list[str] = []
    rows: list[str] = []
    for tr in node.xpath(".//tr"):
        cells = []
        row_headers = []
        for cell in tr.xpath("./th|./td"):
            text = _text_from_node(cell)
            if text:
                cells.append(text)
                if cell.tag.lower() == "th":
                    row_headers.append(text)
        if row_headers and not headers:
            headers = row_headers
        if any(cells):
            rows.append(" | ".join(cells))
    return headers, rows


def _clean_document_root(root):
    """Keep article content and remove navigation/boilerplate before block walking."""
    for node in list(root.xpath(".//*")):
        tag = _tag_name(node)
        attrs = getattr(node, "attrib", {}) or {}
        marker = " ".join(
            str(attrs.get(key) or "")
            for key in ("id", "class", "role", "aria-label")
        ).lower()
        if (
            tag in _NOISE_TAGS
            or attrs.get("aria-hidden") == "true"
            or any(noise in marker for noise in _NOISE_MARKERS)
        ):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)

    main_nodes = root.xpath(".//main")
    if main_nodes:
        return max(main_nodes, key=lambda item: len(_text_from_node(item)))
    article_nodes = root.xpath(".//article")
    if article_nodes:
        return max(article_nodes, key=lambda item: len(_text_from_node(item)))
    return root


def _structured_blocks_from_html(html: str) -> list[StructuredBlock]:
    try:
        root = lxml_html.fromstring(html)
    except (etree.ParserError, ValueError):
        return []

    body = root.xpath("//body")
    if body:
        root = body[0]
    root = _clean_document_root(root)

    blocks: list[StructuredBlock] = []
    heading_path: list[str] = []
    seen_texts: set[tuple[str, str]] = set()

    def append_block(
        text: str,
        *,
        kind: str = "paragraph",
        node=None,
        headers: list[str] | None = None,
    ) -> None:
        normalized = _normalize_text(text)
        if not normalized:
            return
        dedupe_key = (kind, normalized)
        if dedupe_key in seen_texts and kind in {"paragraph", "list_item"}:
            return
        seen_texts.add(dedupe_key)
        blocks.append(
            StructuredBlock(
                text=normalized,
                kind=kind,
                section_title=heading_path[-1] if heading_path else "",
                heading_path=tuple(heading_path),
                table_headers=tuple(headers or ()),
                source_anchor=_node_anchor(node) if node is not None else "",
            )
        )

    def walk(node) -> None:
        tag = _tag_name(node)
        if tag in _NOISE_TAGS:
            return

        if tag in _HEADING_TAGS:
            text = _text_from_node(node)
            if not text:
                return
            level = int(tag[1])
            while len(heading_path) >= level:
                heading_path.pop()
            heading_path.append(text)
            append_block(text, kind="heading", node=node)
            return

        if tag == "table":
            headers, rows = _table_rows(node)
            if rows:
                append_block("\n".join(rows), kind="table", node=node, headers=headers)
            return

        if tag in {"pre", "code"}:
            append_block(_text_from_node(node), kind="code", node=node)
            return

        if tag == "li":
            append_block(_text_from_node(node), kind="list_item", node=node)
            return

        if tag in {"p", "blockquote"}:
            append_block(_text_from_node(node), kind="paragraph", node=node)
            return

        if tag in _CONTAINER_TAGS or tag:
            if not _has_block_descendant(node):
                append_block(_text_from_node(node), kind="paragraph", node=node)
                return
            for child in node:
                walk(child)
            return

    for child in root:
        walk(child)

    return blocks


class _PreludeHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.h1 = ""
        self.subtitle = ""
        self._capture: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        class_names = set(attrs_dict.get("class", "").split())
        if tag == "title" and not self.title:
            self._capture = "title"
            self._buffer = []
        elif tag == "h1" and not self.h1:
            self._capture = "h1"
            self._buffer = []
        elif "subtitle" in class_names and not self.subtitle:
            self._capture = "subtitle"
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self._capture == "title" and tag == "title":
            self.title = " ".join("".join(self._buffer).split())
            self._capture = None
            self._buffer = []
        elif self._capture == "h1" and tag == "h1":
            self.h1 = " ".join("".join(self._buffer).split())
            self._capture = None
            self._buffer = []
        elif self._capture == "subtitle" and tag in {"div", "p", "span"}:
            self.subtitle = " ".join("".join(self._buffer).split())
            self._capture = None
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buffer.append(data)


class HtmlParser(BaseParser):
    def _extract_prelude(self, html: str) -> ParserPrelude:
        parser = _PreludeHTMLParser()
        parser.feed(html)
        title = parser.h1 or parser.title
        subtitle = parser.subtitle
        if subtitle and subtitle == title:
            subtitle = ""
        return ParserPrelude(title=title, subtitle=subtitle)

    def _dedupe_adjacent_paragraphs(self, text: str) -> str:
        paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
        deduped: list[str] = []
        for paragraph in paragraphs:
            if paragraph in deduped[-3:]:
                continue
            deduped.append(paragraph)
        return "\n".join(deduped)

    def _extract_text(self, html: str) -> str:
        """从 HTML 提取正文，去广告/导航/侧边栏"""
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        return self._dedupe_adjacent_paragraphs(text or "")

    def _extract_blocks(self, html: str) -> list[StructuredBlock]:
        blocks = _structured_blocks_from_html(html)
        if blocks:
            return blocks
        text = self._extract_text(html)
        if not text.strip():
            return []
        return [StructuredBlock(text=segment, kind="paragraph") for segment in text.split("\n") if segment.strip()]

    def _load_parts(self, file_path: Path) -> tuple[ParserPrelude, str]:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            html = f.read()
        return self._extract_prelude(html), self._extract_text(html)

    def _parse_parts(self, file_path: Path) -> tuple[ParserPrelude, str]:
        return self._cached_parts(file_path, lambda: self._load_parts(file_path))

    def get_prelude(self, file_path: Path) -> ParserPrelude:
        prelude, _ = self._parse_parts(file_path)
        return prelude

    def parse(self, file_path: Path) -> str:
        _, text = self._parse_parts(file_path)
        return text

    def parse_blocks(self, file_path: Path) -> list[StructuredBlock]:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            html = f.read()
        return self._extract_blocks(html)

    def parse_stream(self, file_path: Path) -> Generator[str, None, None]:
        for block in self.parse_blocks(file_path):
            yield block.text
