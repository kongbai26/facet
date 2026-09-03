"""DOCX 解析器（python-docx）"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

from docx.document import Document as DocxDocumentType
from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.parsers.base import BaseParser, ParserPrelude, StructuredBlock

_PURE_NUMBER_LINE_RE = re.compile(r"^[\d\s./-]+$")


class DocxParser(BaseParser):
    def _iter_blocks(self, doc: DocxDocumentType):
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, doc)
            elif isinstance(child, CT_Tbl):
                yield Table(child, doc)

    def _table_to_text(self, table: Table) -> str:
        rows: list[str] = []
        for row in table.rows:
            cells = [" ".join(cell.text.split()) for cell in row.cells]
            if any(cell for cell in cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _is_good_title(self, text: str) -> bool:
        candidate = " ".join((text or "").split())
        if not candidate or len(candidate) > 120:
            return False
        if _PURE_NUMBER_LINE_RE.match(candidate):
            return False
        return True

    def _extract_title(self, doc: DocxDocumentType) -> str:
        core_title = " ".join((doc.core_properties.title or "").split())
        if self._is_good_title(core_title):
            return core_title

        fallback = ""
        for block in self._iter_blocks(doc):
            if isinstance(block, Paragraph):
                text = " ".join(block.text.split())
                if not text:
                    continue
                style_name = getattr(getattr(block, "style", None), "name", "") or ""
                if style_name.lower().startswith("heading") and self._is_good_title(text):
                    return text
                if not fallback and self._is_good_title(text):
                    fallback = text
        return fallback

    def _load_parts(self, file_path: Path) -> tuple[ParserPrelude, list[str]]:
        doc = DocxDocument(str(file_path))
        blocks: list[str] = []
        for block in self._iter_blocks(doc):
            if isinstance(block, Paragraph):
                text = block.text.strip()
            else:
                text = self._table_to_text(block)
            if text.strip():
                blocks.append(text.strip())
        return ParserPrelude(title=self._extract_title(doc)), blocks

    def _parse_parts(self, file_path: Path) -> tuple[ParserPrelude, list[str]]:
        return self._cached_parts(file_path, lambda: self._load_parts(file_path))

    def get_prelude(self, file_path: Path) -> ParserPrelude:
        prelude, _ = self._parse_parts(file_path)
        return prelude

    def parse(self, file_path: Path) -> str:
        _, blocks = self._parse_parts(file_path)
        return "\n\n".join(blocks)

    def parse_blocks(self, file_path: Path) -> list[StructuredBlock]:
        doc = DocxDocument(str(file_path))
        structured_blocks: list[StructuredBlock] = []
        heading_path: list[str] = []
        for block in self._iter_blocks(doc):
            if isinstance(block, Paragraph):
                text = " ".join(block.text.split())
                if not text:
                    continue
                style_name = getattr(getattr(block, "style", None), "name", "") or ""
                if style_name.lower().startswith("heading"):
                    match = re.search(r"(\d+)", style_name)
                    level = int(match.group(1)) if match else 1
                    while len(heading_path) >= level:
                        heading_path.pop()
                    heading_path.append(text)
                    structured_blocks.append(
                        StructuredBlock(
                            text=text,
                            kind="heading",
                            section_title=text,
                            heading_path=tuple(heading_path),
                        )
                    )
                    continue
                structured_blocks.append(
                    StructuredBlock(
                        text=text,
                        kind="paragraph",
                        section_title=heading_path[-1] if heading_path else "",
                        heading_path=tuple(heading_path),
                    )
                )
            else:
                text = self._table_to_text(block)
                if not text.strip():
                    continue
                headers: list[str] = []
                if block.rows:
                    headers = [" ".join(cell.text.split()) for cell in block.rows[0].cells if " ".join(cell.text.split())]
                structured_blocks.append(
                    StructuredBlock(
                        text=text.strip(),
                        kind="table",
                        section_title=heading_path[-1] if heading_path else "",
                        heading_path=tuple(heading_path),
                        table_headers=tuple(headers),
                    )
                )
        return structured_blocks

    def parse_stream(self, file_path: Path) -> Generator[str, None, None]:
        for block in self.parse_blocks(file_path):
            yield block.text
