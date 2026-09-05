"""Lightweight *chhandas* (metre) detection.

Phase 1 counting is syllable-count based; it identifies the common fixed
syllable-count metres (anuṣṭubh, gāyatrī, triṣṭubh, jagatī). The full Pāṇinian
*gaṇa*-based scheme (light/heavy patterns + arṣa/vedha exceptions) is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass

from sanskrit_nlp.consts import (
    LONG_MATRAS,
    LONG_VOWELS,
)
from sanskrit_nlp.sandhi.rules import tokenize_aksharas

# Long vowel nuclei that make an akshara heavy by themselves.
_HEAVY_VOWELS = LONG_VOWELS | LONG_MATRAS


@dataclass(frozen=True)
class Syllable:
    """One metrical syllable of a Sanskrit utterance."""

    text: str
    heavy: bool


@dataclass(frozen=True)
class Meter:
    """Detected metre metadata."""

    name: str
    total_syllables: int
    syllable_count: int | None = None
    padas: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_syllables": self.total_syllables,
            "syllable_count": self.syllable_count,
            "padas": self.padas,
        }


def syllables(verse: str) -> list[Syllable]:
    """Return metrical syllables for a Devanagari string.

    A syllable is counted per *vowel nucleus* (inherent ``अ`` included). A
    trailing ``ं`` / ``ः`` / ``ऽ`` marks the syllable heavy.
    """
    result: list[Syllable] = []
    for ak in tokenize_aksharas(verse):
        if ak.is_independent:
            result.append(Syllable(ak.base, _is_heavy(ak.base, ak.modifier)))
            continue
        # Consonant cluster with inherent vowel or a dependent vowel.
        vowel = ak.vowel if ak.vowel else "अ"
        result.append(Syllable(f"{ak.base}{vowel}", _is_heavy(vowel, ak.modifier)))
    return result


def _is_heavy(vowel: str, modifier: str) -> bool:
    if vowel in _HEAVY_VOWELS:
        return True
    if modifier:  # anusvara / visarga / candrabindu
        return True
    return False


def split_into_padas(verse: str) -> list[str]:
    """Split a verse on single/double dandas.

    In standard notation the two halves of a śloka are separated by a single
    danda (or none), with the final double danda marking the verse end.
    """
    import re

    parts = re.split(r"[।॥]+", verse)
    return [p.strip() for p in parts if p.strip()]


def detect_meter(verse: str, verbose: bool = False) -> Meter:
    """Detect the metre of ``verse`` from its total syllable count."""
    padas = split_into_padas(verse)
    total = len(syllables(verse))

    name = "unknown"
    if total == 24:
        name = "gāyatrī"
    elif total == 32:
        name = "anuṣṭubh"
    elif total == 44:
        name = "triṣṭubh"
    elif total == 48:
        name = "jagatī"

    return Meter(
        name=name,
        total_syllables=total,
        syllable_count=total,
        padas=max(1, len(padas)),
    )
