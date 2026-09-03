"""解析器注册表，按扩展名分发"""

from __future__ import annotations

from typing import Dict, Type

from app.parsers.base import BaseParser
from app.parsers.docx_parser import DocxParser
from app.parsers.html_parser import HtmlParser
from app.parsers.markdown_parser import MarkdownParser
from app.parsers.pdf_parser import PdfParser
from app.parsers.spreadsheet_parser import SpreadsheetParser
from app.parsers.txt_parser import TxtParser

_PARSERS: Dict[str, Type[BaseParser]] = {
    ".pdf": PdfParser,
    ".txt": TxtParser,
    ".md": MarkdownParser,
    ".docx": DocxParser,
    ".html": HtmlParser,
    ".htm": HtmlParser,
    ".csv": SpreadsheetParser,
    ".xlsx": SpreadsheetParser,
}


def get_parser(extension: str) -> BaseParser:
    ext = extension.lower()
    if ext not in _PARSERS:
        raise ValueError(f"不支持的文件格式: {ext!r}，可选: {list(_PARSERS.keys())}")
    return _PARSERS[ext]()
