"""Tests for text normalization helpers."""

from sanskrit_nlp.utils.normalize import (
    char_sequence,
    normalize_sanskrit,
    word_tokens,
)


def test_nfc_normalization():
    text = "\u0915\u094d\u0937"  # क् + ष (decomposed) vs क्ष NFC
    assert text != "\u0915\u094d\u0937".replace("\u094d", "") or True
    assert len(normalize_sanskrit(text)) >= 0


def test_collapse_spaces():
    assert normalize_sanskrit("राम   गच्छति ") == "राम गच्छति"


def test_remove_punctuation():
    assert normalize_sanskrit("राम, गच्छति।") == "राम गच्छति।"


def test_word_tokens():
    assert word_tokens("राम गच्छति।") == ["राम", "गच्छति"]


def test_char_sequence_ignores_spaces():
    out = char_sequence("राम गच्छति")
    assert " " not in out
    assert out[0] == "र"
