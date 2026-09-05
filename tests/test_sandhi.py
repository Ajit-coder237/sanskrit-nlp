"""Tests for the lexicon-free vowel sandhi splitter."""

from sanskrit_nlp.sandhi.splitter import (
    best_split,
    filter_by_lexicon,
    split_sentence,
    split_token,
)


def test_matradependent_split():
    # सूर्य + उदय -> सूर्योदय
    candidates = split_token("सूर्योदय")
    assert any(c.left == "सूर्य" and c.right == "उदय" for c in candidates)
    assert any(c.rule == "dependent-vowel-sandhi" for c in candidates)


def test_independent_split():
    # नर + इन्द्र -> नरेन्द्र
    candidates = split_token("नरेन्द्र")
    assert any(c.left == "नर" and c.right == "इन्द्र" for c in candidates)
    assert any(c.rule == "dependent-vowel-sandhi" for c in candidates)


def test_best_split_without_lexicon():
    best = best_split("सूर्योदय")
    assert best is not None
    assert best.left == "सूर्य"
    assert best.right == "उदय"


def test_known_word_not_split_with_lexicon():
    lex = {"राम"}
    assert best_split("राम", lexicon=lex) is None


def test_filter_by_lexicon():
    cands = split_token("नरेन्द्र")
    lex = {"नर", "इन्द्र"}
    filtered = filter_by_lexicon(cands, lex)
    assert filtered and filtered[0].left in lex and filtered[0].right in lex


def test_split_sentence_with_lexicon():
    lex = {"रामः"}
    result = split_sentence("सूर्योदय रामः", lexicon=lex)
    # First is a SandhiCandidate, second is a known word => unsplit string.
    assert not isinstance(result[0], str)
    assert isinstance(result[1], str)
