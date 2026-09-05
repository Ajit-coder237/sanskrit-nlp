# Śiṣya-Grantha — Sanskrit Grantha Digitization & Sanskrit → Nepali Commentary Translation

An end-to-end, open-source blueprint and MVP plan for OCR-ing ancient Sanskrit
*granthas* (both *chhandas* / verse and prose), reconstructing linguistically
safe digital Sanskrit (*saṃhitā-pāṭha* + *pada-pāṭha*), and generating
scholar-style Nepali commentary.

## Start here

📄 **[docs/GRANTHA_DIGITIZATION_BLUEPRINT.md](docs/GRANTHA_DIGITIZATION_BLUEPRINT.md)**

The blueprint covers, in depth:

1. **SOTA OCR architecture** — TrOCR/ViT+decoder and CRNN/CTC baselines, layout
   and line segmentation, page restoration, synthetic data generation with
   degradation simulation, and metrically-aware post-processing.
2. **NLP & translation pipeline** — sandhi splitting, morphological analysis
   with ByT5-Sanskrit, *chhandas* / meter detection, IndicTrans2 fine-tuning,
   LoRA fine-tuning of an open LLM for scholar-commentary output.
3. **Data strategy & evaluation** — sourcing, alignment, annotation protocol,
   and CER/WER for OCR plus BLEU / chrF / METEOR / COMET / LaBSE / human
   **Commentary-Faithfulness (CF)** for translation.
4. **GitHub project structure & deployment** — full `src/`, `scripts/`,
   `configs/`, `data/`, `experiments/`, `tests/`, `ui/`, `deployment/` layout,
   CI/CD, data versioning, FastAPI + Streamlit/Gradio, and Docker/K8s notes.
5. **Research paper framework** — complete outline targetable at
   LREC-COLING / ACL Rolling Review / ICDAR / WMT, with experiment tables and
   a citation anchor list.

## Roadmap highlights

- **MVP-0 (2–3 wk):** baseline OCR (Tesseract, Kraken, PaddleOCR) + zero-shot
  IndicTrans2 `san→npi` on a 100-verse seed.
- **MVP-1 (4–6 wk):** line detector + TrOCR-style recognizer, synthetic data
  generator, rule-based sandhi splitter, 100-verse gold test set.
- **MVP-2 (6–8 wk):** real-data fine-tuning, metrical reranking, IndicTrans2
  fine-tune, LoRA commentary LLM, FastAPI + Streamlit demo.
- **MVP-3 (8–12 wk):** full ablation, human-evaluation (CF) protocol, paper,
  data/model release.

## Status

Blueprint and roadmap only. The repository scaffold (data pipeline, model
training, API/UI) is the next build target.
