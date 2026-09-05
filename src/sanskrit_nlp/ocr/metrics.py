"""OCR evaluation metrics: CER and WER with Devanagari-aware normalization."""

from __future__ import annotations

from dataclasses import dataclass

from sanskrit_nlp.utils.normalize import char_sequence, word_tokens


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein edit distance between two sequences."""
    m = len(b)
    # Rolling two-row DP.
    prev = list(range(m + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * m
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,          # deletion
                cur[j - 1] + 1,       # insertion
                prev[j - 1] + cost,   # substitution
            )
        prev = cur
    return prev[m]


@dataclass(frozen=True)
class OcrMetrics:
    """Computed text-recognition error metrics."""

    cer: float
    wer: float
    chars_ref: int
    chars_hyp: int
    words_ref: int
    words_hyp: int
    exact: bool

    def to_dict(self) -> dict:
        return {
            "cer": self.cer,
            "wer": self.wer,
            "chars_ref": self.chars_ref,
            "chars_hyp": self.chars_hyp,
            "words_ref": self.words_ref,
            "words_hyp": self.words_hyp,
            "exact": self.exact,
        }


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate over normalized, space-stripped sequences."""
    ref = char_sequence(reference)
    hyp = char_sequence(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate over normalized word tokens."""
    ref = word_tokens(reference)
    hyp = word_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def evaluate_ocr(reference: str, hypothesis: str) -> OcrMetrics:
    """Evaluate one OCR prediction against one reference line."""
    ref_chars = char_sequence(reference)
    hyp_chars = char_sequence(hypothesis)
    ref_words = word_tokens(reference)
    hyp_words = word_tokens(hypothesis)

    c = (
        _edit_distance(ref_chars, hyp_chars) / len(ref_chars)
        if ref_chars
        else (0.0 if not hyp_chars else 1.0)
    )
    w = (
        _edit_distance(ref_words, hyp_words) / len(ref_words)
        if ref_words
        else (0.0 if not hyp_words else 1.0)
    )

    return OcrMetrics(
        cer=round(c, 6),
        wer=round(w, 6),
        chars_ref=len(ref_chars),
        chars_hyp=len(hyp_chars),
        words_ref=len(ref_words),
        words_hyp=len(hyp_words),
        exact=ref_chars == hyp_chars,
    )
