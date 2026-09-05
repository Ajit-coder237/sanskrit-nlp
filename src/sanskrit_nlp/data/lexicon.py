"""Small lexicon utilities for sandhi validation and evaluation."""

from __future__ import annotations

from pathlib import Path

from sanskrit_nlp.utils.normalize import word_tokens


def load_lexicon(path: str | Path) -> set[str]:
    """Load a plain-text lexicon (one word per line, UTF-8)."""
    path = Path(path)
    words: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            words.update(word_tokens(line))
    return words


def build_vocabulary(items: list[str]) -> set[str]:
    """Build a vocabulary set from an iterable of text items."""
    vocab: set[str] = set()
    for text in items:
        vocab.update(word_tokens(text))
    return vocab
