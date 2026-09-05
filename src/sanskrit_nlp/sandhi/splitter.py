"""Lexicon-free vowel sandhi splitter.

The splitter enumerates possible *pada* (word) splits for a fused Sanskrit
token by re-analyzing each junction akshara. It returns structurally valid
candidates; a lexicon filter can be layered on top to disambiguate.
"""

from __future__ import annotations

from dataclasses import dataclass

from sanskrit_nlp.sandhi.rules import (
    DEPENDENT_COMBINE_RULES,
    INDEPENDENT_COMBINE_RULES,
    tokenize_aksharas,
)

RULE_MATRA = "dependent-vowel-sandhi"
RULE_INDEP = "vowel-sandhi"


@dataclass(frozen=True)
class SandhiCandidate:
    """A proposed (left, right) splitting of a fused Sanskrit token."""

    left: str
    right: str
    rule: str
    junction: str = ""


def _left_with_vowel(prefix: str, base: str, left_vowel: str) -> str:
    """Reconstruct the left word given the original vowel type.

    ``अ`` -> the consonant carries its inherent vowel (bare base).
    ``आ`` -> the consonant carries the long matra ``ा``.
    """
    if left_vowel == "आ":
        return prefix + base + "ा"
    return prefix + base


def split_token(token: str) -> list[SandhiCandidate]:
    """Return all structurally valid sandhi splits for ``token``.

    The returned candidates are *structural*: no lexicon is consulted. Use
    ``filter_by_lexicon`` to keep candidates whose both sides are known words.
    """
    if not token:
        return []

    aksharas = tokenize_aksharas(token)
    candidates: list[SandhiCandidate] = []
    seen: set[tuple[str, str]] = set()

    for ak in aksharas:
        # Dependent-vowel junction (e.g. सूर् + यो + दय -> सूर्य + उदय).
        if ak.is_dependent and ak.vowel in DEPENDENT_COMBINE_RULES:
            for left_vowel, right_vowel in DEPENDENT_COMBINE_RULES[ak.vowel]:
                left = _left_with_vowel(token[: ak.start], ak.base, left_vowel)
                right = right_vowel + token[ak.end :]
                _add(candidates, seen, left, right, RULE_MATRA, ak.vowel)

        # Independent-vowel junction (e.g. नर + इन्द्र -> नरेन्द्र).
        if ak.is_independent and ak.vowel in INDEPENDENT_COMBINE_RULES:
            for left_vowel, right_vowel in INDEPENDENT_COMBINE_RULES[ak.vowel]:
                left = token[: ak.start] + left_vowel
                right = right_vowel + token[ak.end :]
                _add(candidates, seen, left, right, RULE_INDEP, ak.vowel)

    return candidates


def _add(
    candidates: list[SandhiCandidate],
    seen: set[tuple[str, str]],
    left: str,
    right: str,
    rule: str,
    junction: str,
) -> None:
    if not left or not right:
        return
    key = (left, right)
    if key in seen:
        return
    seen.add(key)
    candidates.append(
        SandhiCandidate(left=left, right=right, rule=rule, junction=junction)
    )


def filter_by_lexicon(
    candidates: list[SandhiCandidate], lexicon: set[str]
) -> list[SandhiCandidate]:
    """Keep candidates whose left and right parts are known lexicon words."""
    return [
        c
        for c in candidates
        if c.left in lexicon and c.right in lexicon
    ]


def best_split(
    token: str,
    lexicon: set[str] | None = None,
    prefer_right_shortest: bool = True,
) -> SandhiCandidate | None:
    """Return the most plausible single split.

    When a lexicon is provided:
      1. A token that is itself a known word is left unsplit (no sandhi).
      2. Otherwise lexicon-valid splits are preferred over structural ones.
    Ties are broken by picking the candidate with the shortest (or longest)
    right side, which mimics the ambiguity of Sanskrit word boundaries.
    """
    if lexicon and token in lexicon:
        return None

    candidates = split_token(token)
    if not candidates:
        return None

    if lexicon:
        lex = filter_by_lexicon(candidates, lexicon)
        if lex:
            candidates = lex

    if prefer_right_shortest:
        return min(candidates, key=lambda c: len(c.right))
    return max(candidates, key=lambda c: len(c.right))


def split_sentence(
    text: str, lexicon: set[str] | None = None
) -> list[SandhiCandidate | str]:
    """Split a sentence token-by-token.

    Tokens that cannot be split are returned unchanged as strings; tokens that
    can are returned as :class:`SandhiCandidate`.
    """
    from sanskrit_nlp.utils.normalize import word_tokens

    out: list[SandhiCandidate | str] = []
    for token in word_tokens(text):
        split = best_split(token, lexicon=lexicon)
        if split is None:
            out.append(token)
        else:
            out.append(split)
    return out
