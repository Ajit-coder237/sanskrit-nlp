"""End-to-end tests for the Phase-1 text pipeline."""

from sanskrit_nlp.pipeline import analyze_text


def test_analyze_text_structure():
    result = analyze_text("सूर्योदय")
    assert result.source == "सूर्योदय"
    assert len(result.tokens) == 1
    token = result.tokens[0]
    assert token.surface == "सूर्योदय"
    assert token.chosen is not None
    assert token.chosen.left == "सूर्य"
    assert token.chosen.right == "उदय"


def test_analyze_text_commentary():
    result = analyze_text("कर्म करोति")
    assert result.commentary is not None
    assert result.commentary.engine == "template-v1"
    assert "कर्म" in result.commentary.literal_nepali
