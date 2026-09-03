"""TXT 解析器，带编码检测"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

from charset_normalizer import from_bytes

from app.parsers.base import BaseParser, StructuredBlock

_PARAGRAPH_RE = re.compile(r"\n\s*\n+")


class TxtParser(BaseParser):
    def _detect_encoding(self, file_path: Path) -> str:
        with open(file_path, "rb") as f:
            raw = f.read()
        match = from_bytes(raw).best()
        return str(match.encoding) if match is not None and match.encoding else "utf-8"

    def parse(self, file_path: Path) -> str:
        encoding = self._detect_encoding(file_path)
        with open(file_path, encoding=encoding, errors="replace") as f:
            return f.read()

    def parse_blocks(self, file_path: Path) -> list[StructuredBlock]:
        encoding = self._detect_encoding(file_path)
        with open(file_path, encoding=encoding, errors="replace") as f:
            raw = f.read()
        blocks = []
        for part in _PARAGRAPH_RE.split(raw):
            text = " ".join((part or "").split())
            if text:
                blocks.append(StructuredBlock(text=text, kind="paragraph"))
        return blocks

    def parse_stream(self, file_path: Path) -> Generator[str, None, None]:
        for block in self.parse_blocks(file_path):
            yield block.text
