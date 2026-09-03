"""PDF text parser backed by the permissively licensed pypdf package."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

from app.parsers.base import BaseParser, ParserPrelude, StructuredBlock

_PAGE_MARKER_RE = re.compile(r"^(page\s+\d+|\d+)$", re.IGNORECASE)


class PdfParser(BaseParser):
    def _is_good_title(self, text: str) -> bool:
        candidate = " ".join((text or "").split())
        if not candidate or len(candidate) > 120:
            return False
        if _PAGE_MARKER_RE.match(candidate):
            return False
        if not any(ch.isalpha() or ('\u4e00' <= ch <= '\u9fff') for ch in candidate):
            return False
        return True

    def _extract_title(self, reader) -> str:
        metadata = reader.metadata or {}
        meta_title = " ".join(
            str(getattr(metadata, "title", None) or metadata.get("/Title") or "").split()
        )
        if self._is_good_title(meta_title):
            return meta_title

        if not reader.pages:
            return ""
        first_page_text = reader.pages[0].extract_text() or ""
        for line in first_page_text.splitlines():
            candidate = " ".join(line.split())
            if self._is_good_title(candidate):
                return candidate
        return ""

    def _load_parts(self, file_path: Path) -> tuple[ParserPrelude, list[str]]:
        from pypdf import PdfReader

        reader = PdfReader(file_path, strict=False)
        pages = [page.extract_text() or "" for page in reader.pages]
        return ParserPrelude(title=self._extract_title(reader)), pages

    def _parse_parts(self, file_path: Path) -> tuple[ParserPrelude, list[str]]:
        return self._cached_parts(file_path, lambda: self._load_parts(file_path))

    def get_prelude(self, file_path: Path) -> ParserPrelude:
        prelude, _ = self._parse_parts(file_path)
        return prelude

    def parse(self, file_path: Path) -> str:
        _, pages = self._parse_parts(file_path)
        return "\n\n".join(page for page in pages if page.strip())

    def parse_blocks(self, file_path: Path) -> list[StructuredBlock]:
        _, pages = self._parse_parts(file_path)
        blocks: list[StructuredBlock] = []
        heading_path: list[str] = []
        for page_index, raw_text in enumerate(pages, start=1):
            for part in re.split(r"\n\s*\n+", raw_text.strip()):
                text = " ".join(part.split())
                if not text:
                    continue
                if page_index == 1 and not heading_path:
                    first_line = text.split("。", 1)[0]
                    if self._is_good_title(first_line):
                        heading_path = [first_line]
                        blocks.append(
                            StructuredBlock(
                                text=first_line,
                                kind="heading",
                                section_title=first_line,
                                heading_path=tuple(heading_path),
                                page=page_index,
                            )
                        )
                        remainder = text[len(first_line):].strip()
                        if remainder:
                            blocks.append(
                                StructuredBlock(
                                    text=remainder,
                                    kind="paragraph",
                                    section_title=heading_path[-1],
                                    heading_path=tuple(heading_path),
                                    page=page_index,
                                )
                            )
                        continue
                blocks.append(
                    StructuredBlock(
                        text=text,
                        kind="paragraph",
                        section_title=heading_path[-1] if heading_path else "",
                        heading_path=tuple(heading_path),
                        page=page_index,
                    )
                )
        return blocks

    def parse_stream(self, file_path: Path) -> Generator[str, None, None]:
        for block in self.parse_blocks(file_path):
            yield block.text
