"""Devakshara (akshara) tokenization and vowel sandhi rules.

Phase 1 implements a deterministic, lexicon-free **vowel junction** sandhi
splitter. It covers the common single-vowel junctions used in classical prose
and verse, including *dependent-vowel* (matra) junctions such as
``सूर्य + उदय -> सूर्योदय`` and independent-vowel junctions.

Consonant sandhi, visarga sandhi, and full Pāṇinian coverage are deliberately
deferred to Phase 2 (they are the highest-effort/highest-ambiguity parts).
"""

from __future__ import annotations

from dataclasses import dataclass

from sanskrit_nlp.consts import (
    CONSONANTS,
    DEPENDENT_VOWELS,
    MODIFIERS,
    STANDALONE_VOWELS,
    VIRAMA,
)

# Rule family names (used for reporting/redaction experiments).
RULE_VOWEL = "vowel-sandhi"
RULE_DEPENDENT = "dependent-vowel-sandhi"


@dataclass(frozen=True)
class Akshara:
    """A Devanagari akshara with its location in the source string."""

    start: int
    end: int
    base: str
    vowel: str
    modifier: str = ""

    @property
    def is_dependent(self) -> bool:
        """True when the vowel is a matra attached to a consonant cluster."""
        return bool(self.vowel) and self.vowel not in STANDALONE_VOWELS

    @property
    def is_independent(self) -> bool:
        """True when this akshara is a standalone vowel."""
        return self.base in STANDALONE_VOWELS and self.vowel == self.base


# Combined *dependent* vowel (matra) for a given (left vowel, right vowel) pair.
# ``अ`` means the left consonant carries its inherent vowel; ``आ`` means the
# left akshara already carries a long ``ा``.
#
# These are the Pāṇinian guṇa/vṛddhi junctions (P. 6.1.77-78) as they appear
# as a matra on the final consonant of the left word:
#   अ + इ/ई -> े           नर + इन्द्र -> नरेन्द्र
#   अ + उ/ऊ -> ो           सूर्य + उदय -> सूर्योदय
#   अ/आ + ए/ऐ -> ै
#   अ/आ + ओ/औ -> ौ
#   अ/आ + आ -> ा           कृष्ण + आज्ञा -> कृष्णाज्ञा
DEPENDENT_COMBINE_RULES: dict[str, list[tuple[str, str]]] = {
    "ा": [("अ", "आ"), ("आ", "आ")],
    "े": [("अ", "इ"), ("अ", "ई"), ("आ", "इ"), ("आ", "ई")],
    "ो": [("अ", "उ"), ("अ", "ऊ"), ("आ", "उ"), ("आ", "ऊ")],
    "ै": [("अ", "ए"), ("अ", "ऐ"), ("आ", "ए"), ("आ", "ऐ")],
    "ौ": [("अ", "ओ"), ("अ", "औ"), ("आ", "ओ"), ("आ", "औ")],
}

# Independent (standalone) vowel results. These fire when the left word ends
# directly in a vowel (uncommon because most Sanskrit words are consonant-
# final in short अ), so they are kept deliberately conservative.
INDEPENDENT_COMBINE_RULES: dict[str, list[tuple[str, str]]] = {
    "ए": [("अ", "इ"), ("अ", "ई")],
    "ओ": [("अ", "उ"), ("अ", "ऊ")],
    "ऐ": [("अ", "ए"), ("अ", "ऐ"), ("आ", "इ"), ("आ", "ई")],
    "औ": [("अ", "ओ"), ("अ", "औ"), ("आ", "उ"), ("आ", "ऊ")],
    "आ": [("अ", "आ"), ("आ", "आ")],
    "अर्": [("अ", "ऋ")],
}


def tokenize_aksharas(text: str) -> list[Akshara]:
    """Split a Devanagari string into aksharas.

    This is intentionally conservative: it recognizes standalone vowels,
    consonant clusters joined by virama, dependent vowels, and modifiers.
    Unrecognized characters are skipped.
    """
    aksharas: list[Akshara] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # Standalone vowel akshara.
        if ch in STANDALONE_VOWELS:
            end = i + 1
            while end < n and text[end] in MODIFIERS:
                end += 1
            aksharas.append(
                Akshara(i, end, ch, ch, text[i + 1 : end])
            )
            i = end
            continue

        # Consonant-led akshara.
        if ch in CONSONANTS:
            start = i
            base_chars = [ch]
            j = i + 1

            # Consume virama + consonant clusters.
            while j < n and text[j] == VIRAMA and j + 1 < n and text[j + 1] in CONSONANTS:
                base_chars.append(VIRAMA)
                base_chars.append(text[j + 1])
                j += 2

            # Optional dependent vowel. A consonant with no explicit matra is
            # understood to carry the inherent vowel "अ".
            vowel = "अ"
            if j < n and text[j] in DEPENDENT_VOWELS:
                vowel = text[j]
                j += 1

            # Optional trailing modifiers (anusvara/visarga/etc.).
            modifier = ""
            while j < n and text[j] in MODIFIERS:
                modifier += text[j]
                j += 1

            aksharas.append(Akshara(start, j, "".join(base_chars), vowel, modifier))
            i = j
            continue

        i += 1

    return aksharas
