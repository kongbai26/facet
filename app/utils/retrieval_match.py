"""Helpers for lexical retrieval scoring and exact-match boosts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.:-]*$")
_ERROR_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_:-]{2,}$")
_SHORT_IDENTIFIER_RE = re.compile(r"^[A-Za-z]\d{1,3}$")
_ASCII_ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def normalize_exact_text(text: str) -> str:
    """Normalize text for substring-style exact matching."""
    return _WHITESPACE_RE.sub(" ", (text or "").strip()).lower()


def _sanitize_field_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    return str(value).strip()


def build_lexical_search_text(
    text: str,
    metadata: dict | None = None,
    *,
    lexical_metadata_fields: Sequence[str] = (),
) -> str:
    """Build lexical search text from raw chunk text plus selected metadata."""
    metadata = metadata or {}
    parts = [text or ""]

    for field in lexical_metadata_fields:
        value = metadata.get(field)
        if value:
            parts.append(_sanitize_field_value(value))

    filename = _sanitize_field_value(metadata.get("filename"))
    if filename:
        stem = _sanitize_field_value(metadata.get("file_stem")) or Path(filename).stem
        extension = _sanitize_field_value(metadata.get("extension")) or Path(filename).suffix
        if stem:
            parts.append(stem)
        if extension:
            parts.append(extension)

    return "\n".join(part for part in parts if part)


def normalize_filename(filename: str) -> tuple[str, str]:
    path = Path(_sanitize_field_value(filename))
    stem = path.stem
    suffix = path.suffix.lower()
    return stem, suffix


def _looks_like_identifier(token: str) -> bool:
    if _SHORT_IDENTIFIER_RE.fullmatch(token):
        return True
    if len(token) < 3:
        return False
    if token.upper() == token and any(ch.isalpha() for ch in token):
        return True
    return bool(_IDENTIFIER_RE.match(token)) and (
        "_" in token or "-" in token or "." in token or any(ch.isupper() for ch in token) or any(ch.isdigit() for ch in token)
    )


def _looks_like_error_code(token: str) -> bool:
    # Error codes are conventionally uppercase, but matching must remain
    # case-insensitive: users often paste ``err-123`` instead of ``ERR-123``.
    return len(token) >= 4 and bool(_ERROR_CODE_RE.match(token.upper()))


def _query_tokens(query: str) -> List[str]:
    tokens = _TOKEN_RE.findall(query or "")
    return [token for token in tokens if token]


def _contains_exact_term(text: str, term: str) -> bool:
    """Check an ASCII token match without allowing identifier substrings.

    ``str.__contains__`` would treat ``hook`` as a match for ``atlas``.  That
    is particularly harmful for identifier / filename boosts, where the boost
    can make an otherwise unrelated chunk win the lexical ranking.  Non-ASCII
    characters are deliberately valid boundaries so that ``Atlas平台`` can be
    found by a query for ``atlas``.
    """
    normalized_text = normalize_exact_text(text)
    normalized_term = normalize_exact_text(term)
    if not normalized_text or not normalized_term:
        return False

    start = normalized_text.find(normalized_term)
    while start >= 0:
        end = start + len(normalized_term)
        before = normalized_text[start - 1] if start else ""
        after = normalized_text[end] if end < len(normalized_text) else ""
        if not (before and _ASCII_ALNUM_RE.fullmatch(before)) and not (
            after and _ASCII_ALNUM_RE.fullmatch(after)
        ):
            return True
        start = normalized_text.find(normalized_term, start + 1)
    return False


def _field_variants(field: str, value: str) -> list[str]:
    """Return query-matchable forms for one metadata field.

    A filename is normally queried by its stem (``atlas``), not by the full
    ``Atlas.pdf`` value.  Other fields keep their stored value intact.
    """
    if field in {"filename", "file_stem", "extension"}:
        stem, extension = normalize_filename(value)
        variants = [value]
        if stem:
            variants.append(stem)
        if extension:
            variants.append(extension)
        return variants
    return [value]


def _metadata_query_terms(query: str, value: str) -> list[str]:
    """Return query terms that exactly occur in one metadata value.

    The complete metadata value is considered first.  This supports Chinese
    titles and multi-word document names.  Individual ASCII query tokens then
    cover short entities such as ``atlas``.  Generic Chinese fragments are not
    used as standalone metadata matches, which keeps a question like "什么平台"
    from boosting every document whose title happens to contain "平台".
    """
    normalized_query = normalize_exact_text(query)
    if not normalized_query or not value:
        return []

    matched: list[str] = []
    if len(normalized_query) >= 4 and _contains_exact_term(value, normalized_query):
        matched.append(normalized_query)

    for token in _query_tokens(query):
        token_norm = token.lower()
        if (len(token_norm) < 3 and not _SHORT_IDENTIFIER_RE.fullmatch(token_norm)) or token_norm in matched:
            continue
        if _contains_exact_term(value, token_norm):
            matched.append(token_norm)
    return matched


def compute_exact_match_bonus(
    query: str,
    search_text: str,
    metadata: dict | None,
    exact_match_config,
    *,
    lexical_metadata_fields: Sequence[str] = (),
) -> tuple[float, list[str]]:
    """Return an exact-match bonus in [0, 1] and human-readable reasons."""
    if not exact_match_config or not getattr(exact_match_config, "enabled", False):
        return 0.0, []

    bonus = 0.0
    reasons: list[str] = []
    raw_query = (query or "").strip()
    matched_tokens: set[str] = set()

    field_bonus_map = {
        "filename": getattr(exact_match_config, "filename_bonus", 0.0),
        "file_stem": getattr(exact_match_config, "filename_bonus", 0.0),
        "extension": getattr(exact_match_config, "filename_bonus", 0.0),
        "doc_id": getattr(exact_match_config, "doc_id_bonus", 0.0),
        "tenant_slug": getattr(exact_match_config, "identifier_bonus", 0.0),
    }
    # `search_text` is built from the chunk body *plus* metadata.  Metadata
    # must therefore only be rewarded when the *query* matches that metadata;
    # checking `value in search_text` makes every record self-match.
    field_names = list(dict.fromkeys((
        "filename",
        "file_stem",
        "doc_id",
        *lexical_metadata_fields,
    )))
    for field in field_names:
        value = _sanitize_field_value((metadata or {}).get(field))
        if not value:
            continue

        # A query term receives at most one metadata boost.  `filename` and
        # `file_stem` commonly encode the same thing, and a doc id may repeat
        # it; counting all of them would over-amplify one exact hit.
        matched_terms: list[str] = []
        for variant in _field_variants(field, value):
            matched_terms = _metadata_query_terms(query, variant)
            if matched_terms:
                break
        if not matched_terms or all(term in matched_tokens for term in matched_terms):
            continue

        field_bonus = field_bonus_map.get(field, getattr(exact_match_config, "identifier_bonus", 0.0))
        if field_bonus > 0:
            bonus += field_bonus
            reasons.append(f"{field}:{value}")
        matched_tokens.update(matched_terms)

    for token in _query_tokens(query):
        token_norm = token.lower()
        if token_norm in matched_tokens:
            continue
        if _contains_exact_term(search_text, token):
            if _looks_like_error_code(token):
                token_bonus = getattr(exact_match_config, "error_code_bonus", 0.0)
                if token_bonus > 0:
                    bonus += token_bonus
                    reasons.append(f"error_code:{token}")
                    matched_tokens.add(token_norm)
                    continue
            if _looks_like_identifier(token):
                token_bonus = getattr(exact_match_config, "identifier_bonus", 0.0)
                if token_bonus > 0:
                    bonus += token_bonus
                    reasons.append(f"identifier:{token}")
                    matched_tokens.add(token_norm)

    phrase_candidate = normalize_exact_text(raw_query)
    # If the phrase is itself a metadata hit, it has already received exactly
    # one field bonus above.  Suppress the phrase bonus to avoid another
    # self-match through the metadata appended to `search_text`.
    if (
        phrase_candidate
        and len(phrase_candidate) >= 4
        and phrase_candidate not in matched_tokens
        and _contains_exact_term(search_text, phrase_candidate)
    ):
        token_bonus = getattr(exact_match_config, "phrase_bonus", 0.0)
        if token_bonus > 0:
            bonus += token_bonus
            reasons.append("phrase")

    return min(1.0, bonus), reasons
