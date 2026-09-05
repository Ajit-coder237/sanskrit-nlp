"""Scholar-commentary scaffolding.

Phase 1 provides a deterministic template that mimics what the eventual
fine-tuned LLM will produce, so the downstream pipeline and evaluation are
testable without GPU dependencies. The ``generate_commentary`` function will
later dispatch to the LoRA fine-tuned model when ``nlp.ml`` is installed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gloss:
    """A word-level gloss (source -> target)."""

    source: str
    target: str


@dataclass(frozen=True)
class Commentary:
    """Structured scholar commentary output."""

    literal_nepali: str
    scholarly_nepali: str
    glosses: list[Gloss]
    register: str = "literary-न्याय-शैली"
    engine: str = "template-v1"

    def to_dict(self) -> dict:
        return {
            "literal_nepali": self.literal_nepali,
            "scholarly_nepali": self.scholarly_nepali,
            "glosses": [{"source": g.source, "target": g.target} for g in self.glosses],
            "register": self.register,
            "engine": self.engine,
        }


# Small seed lexicon for the demo template. Replace with a proper wordnet in Phase 2.
SEED_LEXICON: dict[str, str] = {
    "कर्म": "कर्म",
    "फल": "फल",
    "करोति": "गर्छ",
    "अहं": "म",
    "त्वम्": "तिमी",
    "यथा": "जस्तो",
    "तथा": "त्यस्तै",
}


def generate_commentary(verse: str, glosses: list[Gloss] | None = None) -> Commentary:
    """Generate a template-based scholarly commentary.

    This is an intentionally minimal placeholder. The production implementation
    will replace ``engine`` with ``loRA-llm-qwen2.5-7b`` and use the same
    dataclass contract.
    """
    # A deterministic, testable placeholder render. It preserves the input
    # tokens (as a literal gloss) while signalling that the NLP engine will
    # upgrade in Phase 2.
    tokens = [t for t in verse.split() if t]
    gloss_list = glosses or [
        Gloss(source=t, target=SEED_LEXICON.get(t, t)) for t in tokens
    ]
    literal = " ".join(g.target for g in gloss_list)
    return Commentary(
        literal_nepali=literal,
        scholarly_nepali=f"({literal}) — अर्थात्, {tokens[0] if tokens else ''} ...",
        glosses=gloss_list,
        engine="template-v1",
    )
