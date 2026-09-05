"""Tests for the akshara tokenizer."""

from sanskrit_nlp.sandhi.rules import tokenize_aksharas


def test_simple_aksharas():
    # राम = रा + म : first akshara carries matra "ा", second inherits "अ".
    aksharas = tokenize_aksharas("राम")
    assert [a.base for a in aksharas] == ["र", "म"]
    assert [a.vowel for a in aksharas] == ["ा", "अ"]


def test_conjunct():
    # क्षि = क् + ष + ि : one akshara with conjunct base and vowel ि.
    aksharas = tokenize_aksharas("क्षि")
    assert len(aksharas) == 1
    assert aksharas[0].base == "क्ष"
    assert aksharas[0].vowel == "ि"


def test_modifier():
    # तं = त + ं : inherent vowel + anusvara modifier.
    aksharas = tokenize_aksharas("तं")
    assert aksharas[-1].base == "त"
    assert aksharas[-1].vowel == "अ"
    assert aksharas[-1].modifier == "ं"


def test_standalone_vowel():
    aksharas = tokenize_aksharas("इन्द्र")
    assert aksharas[0].is_independent
    assert aksharas[0].vowel == "इ"
