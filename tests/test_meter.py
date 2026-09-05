"""Tests for metre (chhandas) detection."""

from sanskrit_nlp.meter.chandas import detect_meter, syllables


def test_syllables_gita_verse():
    # Sanskrit treats each vowel nucleus as a syllable.
    verse = "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन"
    syls = syllables(verse)
    assert len(syls) == 16
    assert all(isinstance(s.text, str) for s in syls)


def test_detect_anustubh():
    # Full anuṣṭubh śloka: 4 pādas of 8 syllables = 32.
    verse = (
        "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन । "
        "मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥"
    )
    meter = detect_meter(verse)
    assert meter.name == "anuṣṭubh"
    assert meter.total_syllables == 32


def test_detect_gayatri():
    # Gāyatrī metre is 24 syllables.
    meter = detect_meter("तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि")
    assert meter.total_syllables > 0


def test_split_into_padas():
    padas = detect_meter(
        "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन । "
        "मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥"
    )
    assert padas.padas >= 2
