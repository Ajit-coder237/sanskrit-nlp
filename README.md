# Śiṣya-Grantha — Sanskrit Grantha Digitization & Sanskrit → Nepali Commentary Translation

An end-to-end, open-source pipeline and research project for OCR-ing ancient
Sanskrit *granthas* (both *chhandas* / verse and prose), reconstructing
linguistically safe digital Sanskrit (*saṃhitā-pāṭha* + *pada-pāṭha*), and
generating scholar-style Nepali commentary.

## Project status

> **Phase 1 (core linguistics scaffold) is implemented and tested. See
> [docs/PHASES.md](docs/PHASES.md) for the phase map and next steps.**

| Phase | Focus | Status |
|---|---|---|
| 0 | Repository scaffold + project metadata | ✅ |
| 1 | Normalization, aksharas, sandhi, metre, metrics, text pipeline, CLI | ✅ |
| 2 | OCR (layout/line detection, TrOCR, synthetic data) | ⬜ next |
| 3 | Sanskrit NLP (lexicon/neural segmentation, metrical reranking) | ⬜ |
| 4 | Translation (IndicTrans2, LoRA commentary LLM) | ⬜ |
| 5–7 | Data corpus, full evaluation, deployment | ⬜ |

## Start here

- 📄 **[docs/GRANTHA_DIGITIZATION_BLUEPRINT.md](docs/GRANTHA_DIGITIZATION_BLUEPRINT.md)**
  — the full research + engineering blueprint (OCR, NLP/MT, data, evaluation,
  GitHub structure, paper framework).
- 📄 **[docs/PHASES.md](docs/PHASES.md)** — per-phase implementation roadmap and
  what is currently implemented.

## What Phase 1 implements

- **Devvanagari/Sanskrit normalization** (NFC, whitespace, punctuation, dandas).
- **Akshara tokenizer** with conjuncts, matras, and modifiers.
- **Vowel sandhi splitter** (guṇa/vṛddhi junctions) with optional lexicon
  validation and a "known word left unsplit" guard.
- **Metre detection** (gāyatrī / anuṣṭubh / triṣṭubh / jagatī by syllable count).
- **OCR metrics** (CER / WER) and **lightweight MT proxies** (token BLEU, chrF,
  precision/recall).
- **End-to-end text pipeline** and a **CLI**.

Phase 1 has **no GPU/ML dependency**: the `ml` extra (`torch`,
`transformers`, `datasets`, `peft`) is intentionally optional.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Optional ML dependencies (Phase 2+):

```bash
pip install -e ".[ml]"
```

## Quality gates

```bash
make lint      # ruff check
make typecheck # mypy
make test      # pytest (32 tests)
```

These run automatically on every push/PR in `.github/workflows/ci.yml`.

## CLI quick start

```bash
sanskrit-nlp version
sanskrit-nlp analyze "सूर्योदय"
sanskrit-nlp tokenize "कर्मण्येवाधिकारस्ते"
sanskrit-nlp metrics --kind ocr "कर्म गच्छति" "कर्म अच्छति"
sanskrit-nlp metrics --kind mt "राम गच्छति" "राम अच्छति"
```

## Repository layout

```
src/sanskrit_nlp/     # Python package
 ├── ocr/             # OCR metrics (model code lands in Phase 2)
 ├── sandhi/          # akshara tokenizer + vowel sandhi splitter
 ├── meter/           # chhandas detection
 ├── nlp/             # translation/commentary scaffolding + MT metrics
 ├── data/            # lexicon utilities
 ├── utils/           # normalization, config loading
 ├── pipeline.py      # deterministic text analysis pipeline
 └── cli.py           # command-line interface
tests/                # pytest test suite
configs/              # YAML configuration
data/                 # sample data + (git-ignored) raw/processed/gold
docs/                 # blueprint + phase documentation
```

## License

Code is intended to be released under **Apache-2.0**. Data licenses are
documented per source.

## Research target

LREC-COLING / ACL Rolling Review / ICDAR / WMT — see the blueprint's
[research paper framework](docs/GRANTHA_DIGITIZATION_BLUEPRINT.md#7-part-5--research-paper-framework).
