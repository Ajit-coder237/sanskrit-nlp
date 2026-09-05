"""Devanagari character inventories used across the pipeline."""

from __future__ import annotations

# Independent (standalone) vowels.
STANDALONE_VOWELS = frozenset("अआइईउऊऋॠऌॡएऐओऔ")

# Dependent vowel signs (matras).
DEPENDENT_VOWELS = frozenset("ािीुूृॄॅॆेैॉॊोौ")

# Base consonants (Devanagari).
CONSONANTS = frozenset(
    "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ"
)

# Characters that attach to the end of an akshara (voice, anusvara, etc.).
MODIFIERS = frozenset("ंःँऽ")

# Characters that suppress the inherent vowel of the preceding consonant.
VIRAMA = "\u094d"  # ्

# Long vowels / long matras, which make a syllable "heavy".
LONG_VOWELS = frozenset("आईईऋॠऌॡएऐओऔ")
LONG_MATRAS = frozenset("ाीूृॄेैोौ")

# Devanagari sentence/verse delimiters.
DANDA_CHARS = frozenset("।")

# Punctuation that should be ignored for linguistic scoring.
PUNCTUATION_CHARS = frozenset(".,;:!?()[]{}‘’“”\"'`~-_/\\|<>–—")

# Default transformation target for the demo pipeline.
DEFAULT_META = {
    "project": "sanskrit-nlp",
    "phase": "1",
    "scope": "core-linguistics-and-scaffold",
}
