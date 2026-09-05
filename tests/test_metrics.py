"""Tests for OCR and MT metric functions."""

from sanskrit_nlp.nlp.eval import chrf, evaluate_translation, token_bleu
from sanskrit_nlp.ocr.metrics import cer, evaluate_ocr, wer


def test_cer_exact():
    assert cer("राम", "राम") == 0.0


def test_cer_one_error():
    assert cer("राम", "याम") > 0


def test_cer_empty_reference():
    assert cer("", "") == 0.0


def test_wer_exact():
    assert wer("राम गच्छति", "राम गच्छति") == 0.0


def test_evaluate_ocr_exact():
    m = evaluate_ocr("राम गच्छति", "राम गच्छति")
    assert m.cer == 0.0 and m.wer == 0.0 and m.exact


def test_bleu_exact():
    assert token_bleu("राम गच्छति", "राम गच्छति") == 1.0


def test_bleu_empty():
    assert token_bleu("", "") == 0.0


def test_chrf_exact():
    assert chrf("राम गच्छति", "राम गच्छति") == 1.0


def test_evaluate_translation():
    m = evaluate_translation("राम गच्छति", "राम गच्छति")
    assert m.bleu == 1.0 and m.chrf == 1.0 and m.exact
