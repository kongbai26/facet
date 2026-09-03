"""Markdown 解析器"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

import markdown as markdown_lib

from app.parsers.base import BaseParser, ParserPrelude, StructuredBlock
from app.parsers.html_parser import _structured_blocks_from_html

_TITLE_FIELD_RE = re.compile(r"^title\s*:\s*(.+?)\s*$", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$")


class MarkdownParser(BaseParser):
    def _read(self, file_path: Path) -> str:
        with open(file_path, encoding="utf-8") as f:
            return f.read()

    def _split_front_matter(self, text: str) -> tuple[str, str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return "", text
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                front_matter = "\n".join(lines[1:index])
                body = "\n".join(lines[index + 1 :])
                return front_matter, body
        return "", text

    def _extract_title(self, front_matter: str, body: str) -> str:
        if front_matter:
            for line in front_matter.splitlines():
                match = _TITLE_FIELD_RE.match(line.strip())
                if match:
                    return match.group(1).strip().strip("'\"")
        for line in body.splitlines():
            match = _HEADING_RE.match(line)
            if match:
                return match.group(1).strip()
        return ""

    def _load_parts(self, file_path: Path) -> tuple[ParserPrelude, str]:
        raw = self._read(file_path)
        front_matter, body = self._split_front_matter(raw)
        body = body.strip()
        return ParserPrelude(title=self._extract_title(front_matter, body)), body

    def _parse_parts(self, file_path: Path) -> tuple[ParserPrelude, str]:
        return self._cached_parts(file_path, lambda: self._load_parts(file_path))

    def get_prelude(self, file_path: Path) -> ParserPrelude:
        prelude, _ = self._parse_parts(file_path)
        return prelude

    def parse(self, file_path: Path) -> str:
        _, body = self._parse_parts(file_path)
        return body

    def parse_blocks(self, file_path: Path) -> list[StructuredBlock]:
        raw = self._read(file_path)
        front_matter, body = self._split_front_matter(raw)
        html = markdown_lib.markdown(
            body,
            extensions=["tables", "fenced_code", "sane_lists"],
            output_format="html5",
        )
        blocks = _structured_blocks_from_html(html)
        if not blocks:
            body = body.strip()
            if body:
                blocks = [StructuredBlock(text=segment.strip(), kind="paragraph") for segment in body.split("\n\n") if segment.strip()]
        title = self._extract_title(front_matter, body)
        if title and (not blocks or blocks[0].text != title):
            blocks.insert(0, StructuredBlock(text=title, kind="heading", section_title=title, heading_path=(title,)))
        return blocks

    def parse_stream(self, file_path: Path) -> Generator[str, None, None]:
        for block in self.parse_blocks(file_path):
            yield block.text
