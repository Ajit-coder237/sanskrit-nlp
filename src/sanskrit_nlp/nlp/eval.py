"""MT evaluation helpers.

The full national-metric suite (SacreBLEU, chrF, METEOR, COMET, human
Commentary-Faithfulness) is wired to heavy dependencies and is deferred to
the translation phase. Phase 1 ships lightweight, deterministic proxies:
normalized token BLEU, character n-gram overlap (chrF-style), and token
precision/recall.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from sanskrit_nlp.utils.normalize import word_tokens


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _clipped_precision(ref: list[str], hyp: list[str], n: int) -> float:
    ref_counts = Counter(_ngrams(ref, n))
    hyp_counts = Counter(_ngrams(hyp, n))
    total = sum(hyp_counts.values())
    if total == 0:
        return 0.0
    return sum(min(v, ref_counts[k]) for k, v in hyp_counts.items()) / total


def _brevity_penalty(ref_len: int, hyp_len: int) -> float:
    if hyp_len == 0 or hyp_len > ref_len:
        return 1.0
    return math.exp(1 - ref_len / hyp_len)


def token_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """BLEU-4 over normalized word tokens."""
    ref = word_tokens(reference)
    hyp = word_tokens(hypothesis)
    if not ref or not hyp:
        return 0.0
    # Only use n-gram orders for which both sides have at least one n-gram.
    precisions = [
        _clipped_precision(ref, hyp, n)
        for n in range(1, max_n + 1)
        if _ngrams(ref, n) and _ngrams(hyp, n)
    ]
    if not precisions or any(p == 0.0 for p in precisions):
        return 0.0
    geo = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return geo * _brevity_penalty(len(ref), len(hyp))


def chrf(reference: str, hypothesis: str, max_n: int = 6) -> float:
    """Character n-gram F-score (chrF-style, beta=1)."""
    ref = __char_tokens(reference)
    hyp = __char_tokens(hypothesis)
    if not ref or not hyp:
        return 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    for n in range(1, max_n + 1):
        ref_counts = Counter(_ngrams(ref, n))
        hyp_counts = Counter(_ngrams(hyp, n))
        total_hyp = sum(hyp_counts.values())
        total_ref = sum(ref_counts.values())
        if total_hyp == 0 or total_ref == 0:
            continue
        overlap = sum(min(v, ref_counts[k]) for k, v in hyp_counts.items())
        precisions.append(overlap / total_hyp)
        recalls.append(overlap / total_ref)

    if not precisions:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def __char_tokens(text: str) -> list[str]:
    """Normalize to word token sequence, then flatten into char tokens."""
    return [ch for w in word_tokens(text) for ch in w]


@dataclass(frozen=True)
class TranslationMetrics:
    """Deterministic proxy metrics for translation quality."""

    bleu: float
    chrf: float
    precision: float
    recall: float
    exact: bool

    def to_dict(self) -> dict:
        return {
            "bleu": self.bleu,
            "chrf": self.chrf,
            "precision": self.precision,
            "recall": self.recall,
            "exact": self.exact,
        }


def evaluate_translation(reference: str, hypothesis: str) -> TranslationMetrics:
    """Evaluate one translation candidate against a reference."""
    ref = word_tokens(reference)
    hyp = word_tokens(hypothesis)
    ref_counts = Counter(ref)
    hyp_counts = Counter(hyp)
    overlap = sum(min(v, ref_counts[k]) for k, v in hyp_counts.items())
    precision = overlap / len(hyp) if hyp else 0.0
    recall = overlap / len(ref) if ref else 0.0

    return TranslationMetrics(
        bleu=round(token_bleu(reference, hypothesis), 6),
        chrf=round(chrf(reference, hypothesis), 6),
        precision=round(precision, 6),
        recall=round(recall, 6),
        exact=ref == hyp,
    )
