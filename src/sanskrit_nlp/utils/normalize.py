"""Text normalization helpers for Devanagari / Sanskrit text."""

from __future__ import annotations

import re
import unicodedata

from sanskrit_nlp.consts import PUNCTUATION_CHARS

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[{}]".format(re.escape("".join(PUNCTUATION_CHARS))))
_DANDA_RE = re.compile(r"[।॥]")


def normalize_unicode(text: str) -> str:
    """Return NFC-normalized text.

    Old digitizers often produce decomposed Devanagari; NFC gives a single
    canonical form that keeps OCR and MT scoring comparable.
    """
    return unicodedata.normalize("NFC", text)


def collapse_spaces(text: str) -> str:
    """Collapse runs of whitespace and strip."""
    return _WS_RE.sub(" ", text).strip()


def remove_punctuation(text: str) -> str:
    """Remove punctuation but keep Devanagari danda characters."""
    return _PUNCT_RE.sub(" ", text)


def normalize_sanskrit(text: str, strip_dandas: bool = False) -> str:
    """Normalize a Sanskrit string for scoring and downstream use."""
    text = normalize_unicode(text)
    text = collapse_spaces(text)
    text = remove_punctuation(text)
    text = collapse_spaces(text)
    if strip_dandas:
        text = _DANDA_RE.sub(" ", text)
        text = collapse_spaces(text)
    return text


def word_tokens(text: str) -> list[str]:
    """Split into word-level tokens, treating dandas as separators."""
    text = normalize_sanskrit(text, strip_dandas=True)
    return [tok for tok in text.split() if tok]


def char_sequence(text: str, ignore_spaces: bool = True) -> list[str]:
    """Return a character sequence for edit-distance scoring.

    Devanagari composed codepoints are kept as single characters.
    """
    text = normalize_sanskrit(text, strip_dandas=True)
    if ignore_spaces:
        text = text.replace(" ", "")
    return list(text)
