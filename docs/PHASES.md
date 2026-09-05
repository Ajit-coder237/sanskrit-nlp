# Development Phases

This project is built in reviewable, test-covered phases so that each stage is
runnable and reproducible before heavier model/dependency work begins.

## Status map

| Phase | Focus | Status |
|---|---|---|
| 0 | Repository scaffold + project metadata | ✅ Done |
| 1 | Core linguistics: normalization, aksharas, sandhi, metrics, pipeline + tests | ✅ Done (this PR) |
| 2 | OCR: layout/line detection, TrOCR-style model, synthetic data generator | ⬜ Next |
| 3 | Sanskrit NLP: ByT5-style segmentation, lexicon filtering, metre-based reranking | ⬜ Planned |
| 4 | Translation: IndicTrans2 fine-tune, LoRA commentary LLM, reranker | ⬜ Planned |
| 5 | Data language: source/align/annotate golden `san→npi` corpus | ⬜ Planned |
| 6 | Evaluation suite: COMET, human Commentary-Faithfulness, reports | ⬜ Planned |
| 7 | Deployment: FastAPI + Streamlit + Docker + K8s | ⬜ Planned |

---

## Phase 1 — Core linguistics and scaffold

What is implemented in this phase:

- **Normalization** (`src/sanskrit_nlp/utils/normalize.py`)
  - NFC Unicode normalization, whitespace collapse, punctuation removal while
    preserving Devanagari dandas.
- **Akshara tokenizer** (`src/sanskrit_nlp/sandhi/rules.py`)
  - Splits Devanagari into aksharas (base + vowel + modifier).
  - Handles consonant clusters (virama), dependent matras, and standalone vowels.
  - Consonants without an explicit matra carry the inherent vowel `अ`.
- **Vowel sandhi splitter** (`src/sanskrit_nlp/sandhi/splitter.py`)
  - Lexicon-free enumeration of structurally valid splits (guṇa/vṛddhi junctions).
  - `best_split` uses an optional lexicon; a token that is already a known word
    is left unsplit, and lexicon-valid splits are preferred.
  - Over-generation is expected and documented; lexical filtering is the
    disambiguation mechanism (full Pāṇinian rules are deferred).
- **Metre detection** (`src/sanskrit_nlp/meter/chandas.py`)
  - Syllable counting and fixed-count metre identification
    (gāyatrī 24, anuṣṭubh 32, triṣṭubh 44, jagatī 48).
- **Evaluation metrics**
  - OCR: Devanagari-aware **CER / WER** (`src/sanskrit_nlp/ocr/metrics.py`).
  - Translation: lightweight **token BLEU, chrF, token precision/recall**
    (`src/sanskrit_nlp/nlp/eval.py`). COMET/METEOR are deferred to the MT phase.
- **Commentary placeholder** (`src/sanskrit_nlp/nlp/commentary.py`)
  - Deterministic template commentary contract that the future LoRA model will
    replace while keeping the same dataclass interface.
- **End-to-end text pipeline** (`src/sanskrit_nlp/pipeline.py`)
  - Text → token analysis → potential sandhi splits → commentary, all model-free.
- **CLI** (`src/sanskrit_nlp/cli.py`)
  - `sanskrit-nlp version`, `analyze`, `tokenize`, `metrics`.

### Scope guardrails

- Phase 1 is intentionally **CPU-only and deterministic**: no PyTorch /
  transformers dependency is required to install or test it.
- Sandhi is restricted to the common **vowel junctions** (guṇa/vṛddhi).
  Consonant sandhi, visarga sandhi, and full Pāṇinian ambiguity resolution are
  Phase 3 work.
- The `nlp` extra (`torch`, `transformers`, `datasets`, `peft`) is optional and
  is not needed for Phase 1.

### Run/inspect locally

```bash
# one-time setup
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"

# quality gates
make lint      # ruff
make typecheck # mypy
make test      # pytest

# smoke CLI
sanskrit-nlp version
sanskrit-nlp analyze "सूर्योदय"
sanskrit-nlp tokenize "कर्मण्येवाधिकारस्ते"
sanskrit-nlp metrics --kind ocr "कर्म गच्छति" "कर्म अच्छति"
```

### Known Phase-1 limitations

- Sandhi splitting is structural and over-generates; a proper trained
  lexical/JV segmentation model is the Phase 3 target.
- Metre detection only uses total syllable count; it does not yet validate
  light/heavy *gaṇa* patterns.
- Commentary output is a deterministic placeholder, not a learned translation.
- No image/PDF input path exists yet (Phase 2).

---

## Phase 2 — OCR (next)

Planned work:

- Layout and line segmentation (DB / DBNet++ / Swin baselines).
- TrOCR-style ViT + autoregressive decoder recognizer, with a CRNN/CTC fallback.
- Synthetic page renderer (Devanagari fonts + degradation operators).
- `data/` ingestion for scans/PDFs and the image → text entry point in
  `pipeline.py`.

Acceptance criteria: CER < 5% on clean printed books on a held-out gold set;
runnable `sanskrit-nlp ocr <image>` CLI.
