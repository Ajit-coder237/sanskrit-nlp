"""End-to-end text pipeline (Phase 1: text-in -> structural analysis).

This module wires the deterministic linguistic components so a caller can
process a Sanskrit grantha text *before* OCR/translation models exist. The OCR
entry point will be added in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sanskrit_nlp.nlp.commentary import Commentary, generate_commentary
from sanskrit_nlp.sandhi.splitter import SandhiCandidate, split_sentence


@dataclass(frozen=True)
class AnalyzedToken:
    """Output of the sandhi/analysis pass for one surface token."""

    surface: str
    splits: list[SandhiCandidate]
    chosen: SandhiCandidate | None


@dataclass
class PipelineResult:
    """Structured output of the Phase-1 text pipeline."""

    source: str = ""
    tokens: list[AnalyzedToken] = field(default_factory=list)
    commentary: Commentary | None = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "tokens": [
                {
                    "surface": t.surface,
                    "splits": [
                        {
                            "left": s.left,
                            "right": s.right,
                            "rule": s.rule,
                            "junction": s.junction,
                        }
                        for s in t.splits
                    ],
                    "chosen": (
                        {
                            "left": t.chosen.left,
                            "right": t.chosen.right,
                            "rule": t.chosen.rule,
                            "junction": t.chosen.junction,
                        }
                        if t.chosen
                        else None
                    ),
                }
                for t in self.tokens
            ],
            "commentary": self.commentary.to_dict() if self.commentary else None,
        }


def analyze_text(source: str, lexicon: set[str] | None = None) -> PipelineResult:
    """Analyze a Sanskrit text: sandhi candidates + template commentary.

    This is the Phase-1 structural analysis entry point. It does not run any
    model. It is deliberately deterministic and fast so it can be unit-tested
    and run in CI on a CPU-only environment.
    """
    from sanskrit_nlp.utils.normalize import word_tokens

    result = PipelineResult(source=source)
    for token in word_tokens(source):
        candidates = split_sentence(token, lexicon=lexicon)
        chose = None
        for c in candidates:
            if isinstance(c, SandhiCandidate):
                chose = c
                break
        splits = [c for c in candidates if isinstance(c, SandhiCandidate)]
        result.tokens.append(AnalyzedToken(surface=token, splits=splits, chosen=chose))
    result.commentary = generate_commentary(source)
    return result
