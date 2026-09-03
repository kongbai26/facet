"""解析器抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class ParserPrelude:
    title: str = ""
    subtitle: str = ""
    header_lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StructuredBlock:
    text: str
    kind: str = "paragraph"
    section_title: str = ""
    heading_path: tuple[str, ...] = field(default_factory=tuple)
    table_headers: tuple[str, ...] = field(default_factory=tuple)
    source_anchor: str = ""
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    def _parts_cache(self) -> dict[str, Any]:
        cache = getattr(self, "_parsed_parts_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_parsed_parts_cache", cache)
        return cache

    def _cache_key(self, file_path: Path) -> str:
        try:
            stat = file_path.stat()
        except OSError:
            return str(file_path.resolve())
        return f"{file_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"

    def _cached_parts(self, file_path: Path, loader: Callable[[], T]) -> T:
        cache = self._parts_cache()
        key = self._cache_key(file_path)
        if key not in cache:
            cache[key] = loader()
        return cache[key]

    def get_prelude(self, file_path: Path) -> ParserPrelude:
        """Return high-confidence leading metadata for indexing."""
        return ParserPrelude()

    def parse_blocks(self, file_path: Path) -> list[StructuredBlock]:
        """Return structured blocks when the parser supports them.

        Parsers can override this to expose headings, list items, tables and
        other structure. The default fallback preserves the current stream API.
        """
        blocks: list[StructuredBlock] = []
        for segment in self.parse_stream(file_path):
            text = " ".join((segment or "").split())
            if text:
                blocks.append(StructuredBlock(text=text))
        return blocks

    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """解析文件，返回全文"""

    @abstractmethod
    def parse_stream(self, file_path: Path) -> Generator[str, None, None]:
        """流式解析，按页/段 yield"""
