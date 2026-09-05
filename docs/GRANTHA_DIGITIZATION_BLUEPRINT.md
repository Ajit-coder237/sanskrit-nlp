# Śiṣya-Grantha: An End-to-End Sanskrit Grantha Digitization & Sanskrit→Nepali Commentary Translation Pipeline

**A technical blueprint, open-source project plan, and research-paper framework**

| | |
|---|---|
| **Version** | 1.0 (draft) |
| **Author** | Data Science / NLP & CV engineer (you) |
| **Date** | 2026-09-05 |
| **Audience** | ML engineers, Sanskrit scholars, open-source contributors, conference reviewers |
| **License intent** | Code: Apache-2.0. Data: per-source licenses; new gold data: CC-BY-SA-4.0 |
| **Target venues** | LREC-COLING, ACL Rolling Review, EMNLP, ICDAR, WMT shared task |

---

## Table of Contents

1. [Executive Summary & North-Star Metric](#1-executive-summary--north-star-metric)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Part 1 — State-of-the-Art OCR Architecture](#3-part-1--state-of-the-art-ocr-architecture)
4. [Part 2 — NLP & Translation Pipeline](#4-part-2--nlp--translation-pipeline)
5. [Part 3 — Data Strategy & Evaluation](#5-part-3--data-strategy--evaluation)
6. [Part 4 — GitHub Project Structure & Deployment](#6-part-4--github-project-structure--deployment)
7. [Part 5 — Research Paper Framework](#7-part-5--research-paper-framework)
8. [MVP Roadmap & Milestones](#8-mvp-roadmap--milestones)
9. [Engineering Recommendation Summary](#9-engineering-recommendation-summary)
10. [Glossary](#10-glossary)
11. [References & Useful Resources](#11-references--useful-resources)

---

## 1. Executive Summary & North-Star Metric

You are building a **stack**: *Scan → Correct → Segment → Recognize → Reconstruct → Analyze → Translate → Verify → Comment*.

- **Input:** raw photos / scanned PDFs of Sanskrit *granthas* (printed, lithographed, and handwritten/world-heritage manuscripts).
- **Output:** (a) clean digital Sanskrit text (both *chhandas* / verse and prose as *saṃhitā-pāṭha* and *pada-pāṭha*), (b) a scholar-grade Nepali commentary-style translation that preserves the interpretive content of the Sanskrit commentators.
- **Constraint:** Sanskrit and Nepali are 3–10× more morphologically and metrically complex than common MT pairs, and *Sanskrit→Nepali* parallel data is nearly nonexistent publicly.

### North-star metric

> **Commentary-Faithfulness (CF):** the percentage of independently annotated *key interpretive units* (KIs) — technical terms, proper names, *vibhakti*-carrying arguments, figures-of-speech, and kārikā-like instructions — that appear semantically correctly in the generated Nepali commentary, as judged by a panel of ≥2 bilingual Sanskrit–Nepali scholars. Secondary metrics: **CER/WER** for OCR, **COMET** for translation, and **metrical validity** (does the reconstructed *śloka* satisfy its detected meter?).

Everything else in this blueprint is a means to maximize **CF** while keeping the system reproducible and open.

---

## 2. High-Level System Architecture

```mermaid
flowchart LR
    subgraph INPUT
      A[PDF / JPG / TIFF / RAW scans]
    end
    subgraph PREPROCESS
      B[Page selection & detection]
      C[Deskew / Dewarp / Denoise / Binarize]
      D[Super-resolution & ink restoration]
    end
    subgraph STRUCTURE
      E[Layout & margin analysis]
      F[Text-block + line segmentation]
      G[Verse/Prose & commentary detection]
    end
    subgraph OCR
      H[Line recognition: ViT encoder + autoregressive decoder]
      I[CTC/beam post-pass + lexicon LM]
      J[Confidence extraction & low-confidence queue]
    end
    subgraph RECONSTRUCTION
      K[Script normalization]
      L[Sandhi splitting / pada-patha]
      M[Morpho-syntactic tagging (ByT5-Sanskrit)]
      N[Meter identification & metrical validation]
    end
    subgraph TRANSLATION
      O[Context-assembled prompt / source sentence]
      P[Scholar-commentary LLM fine-tuned w/ LoRA]
      Q[Metric-aware retry / lexical glossary override]
    end
    subgraph OUTPUT
      R[Digital Sanskrit XML/TEI]
      S[Nepali commentary + interlinear gloss]
      T[Human-in-the-loop review UI]
    end
    A --> B --> C --> D --> E --> F --> G --> H --> I --> J
    J --> K --> L --> M --> N --> O --> P --> Q
    P --> R & S --> T
    T -.corrections.-> J
```

**Key design principle:** treat OCR and translation as **two strong, separately evaluated subsystems** connected by a **linguistically safe reconstruction layer**. Never let translation silently inherit OCR errors. A single mistake in *anusvāra* or a *mātrā* can change a word's *kāraka* and therefore the intended commentary.

---

## 3. Part 1 — State-of-the-Art OCR Architecture

### 3.1 What makes Sanskrit/Devanagari OCR hard

| Difficulty | Example / detail |
|---|---|
| **Script complexity** | Devanagari has 14 vowel signs (*mātrā*), conjunct clusters, *nukta*, and the *vowel-independent* cluster `क` becomes `क्ष`, `त्र`, `ज्ञ`, `श्र`, etc. |
| **Ancient variants** | *Śaṅkarābharaṇa*-style old Devanagari, grantha-like variants, archaic glyph forms, hair-line stroke shapes. |
| **Vedic accents** | *Udātta*, *anudātta*, *svarita* marks, *avagraha*, *pāṭha*-specific diacritics, Ṛgvedic accent ligatures. |
| **Degradation** | Yellowing, ink fade, bleed-through (Show-through), foxing, creases, blotches, missing serifs, crop/gutter shadows. |
| **Structure** | Double *daṇḍa* (‖) separators, marginalia, *ṭippaṇī* commentary lines interleaved with the root text, page numbers/header ornaments. |
| **Layout** | Multiple columns, vertical commentary, text boxes, interpolation of *pada-pāṭha* within *saṃhitā-pāṭha*. |
| **Noise in label space** | OCR ground truth from different digitizers normalizes accents/mātrā inconsistently (composition vs. precomposed Unicode — Devanagari is *not* NFC-clean in old digitizers). |

### 3.2 Recommended architecture (research-grade)

Do **not** train a single end-to-end page-to-text model at first. Decompose into a modern document-analysis pipeline:

1. **Page preprocessing** (classical CV + learned restoration)
2. **Layout analysis** (detect text-blocks, commentary vs. root text, margins)
3. **Line segmentation** (text-line detection + crop)
4. **Text recognition** (modern transformer OCR per line)
5. **Post-processing** (lexicon + language-model rescoring + metrical validation)

#### 3.2.1 Layout & line detection

- **Start:** PaddleOCR's **DB (Differentiable Binarization)** detector or **PP-OCRv4/v5** text-detection — they are easy to fine-tune and ship with exportable EX/ONNX refits. The DB head is a great baseline for *text-block* and *line* candidates.
- **Better for curved/warped pages:** **DBNet++** on top of an **InternImage** or **Swin** backbone, or **SegFormer-based line segmentation** that outputs a line-mask per page (a 1-channel semantic map), which is much more robust to curved lines than axis-aligned boxes.
- **Advanced/expert:** **DocTr / LinkNet / DocSegTr**-style page-to-line polygonal segmentation, or **Kraken's BLLA** segmenter for historical hand-printed books.
- **Commentary vs. root text:** fine-tune a small classifier (e.g., a ViT or a CNN over cropped pseudo-lines) to assign `root_text`, `commentary`, `marginalia`, `running_head`, `ornament`. This is critical for a *grantha* because the scholar's commentary is *nestled inside* the root text.

#### 3.2.2 Line-level text recognizer

Two strong options — pick Option A for the research paper; keep Option B as a strong baseline.

**Option A — Vision Encoder + Autoregressive Decoder (TrOCR-style)**

- **Encoder:** **ViT (BEiT/Swin)** or **SigLIP**-style image encoder initialized from a multilingual pretrained vision model (e.g., `microsoft/trocr-base-printed` or `google/siglip-base`), resized features fed into a transformer **decoder**.
- **Decoder:** a small 12-layer transformer decoder trained with **label smoothing** and per-*akṣara* aligned targets.
- **Why:** attention over the whole line handles conjuncts and irregular spacing far better than CTC with a pure CNN backbone.
- **Very strong baseline to watch:** fine-tune a **GOT-OCR2**-style end-to-end model (single model, 580M param) or **Qwen2.5-VL-3B** on manuscript crops. These are expensive but set a strong ceiling.

**Option B — CRNN / CNN-Transformer hybrid with CTC**

- CNN: **ResNet34 / EfficientNetV2-S / ConvNeXt** stem producing a width-preserving feature map.
- Head: **2-layer BiLSTM** or a small **transformer encoder** → **CTC** head.
- **Why keep it:** it is cheap, extremely fast, and you can train it with *purely synthetic* data in hours. Use it as the "fast pass" tier that hands difficult crops to Option A.

**My recommendation:** build a **cascaded policy** — cheap CRNN for high-confidence lines, expensive TrOCR/VLM for low-confidence lines. This is itself a publishable engineering contribution for low-resource OCR.

#### 3.2.3 Recognition architecture you can actually build (PyTorch sketch)

```python
# sanocr/models/trocr_line.py
import torch
from torch import nn
from transformers import ViTModel, PreTrainedTokenizerFast

class TrOCRLine(nn.Module):
    def __init__(self, enc_name="google/siglip-base-patch16-224",
                 dec_layers=6, d_model=512, vocab_size=512, max_len=512):
        super().__init__()
        self.encoder = ViTModel.from_pretrained(enc_name)   # freeze initially
        self.proj = nn.Linear(self.encoder.config.hidden_size, d_model)
        # linear positional encoding for line crops along width
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=8,
                                       batch_first=True),
            num_layers=dec_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def encode_image(self, images):
        x = self.encoder(images, output_hidden_states=True).last_hidden_state
        return self.proj(x)  # [B, S, d]

    def forward(self, images, tokens, mask):
        enc = self.encode_image(images)
        tgt = self.decoder.embedding(tokens) + self.pos_emb[:, :tokens.size(1)]
        # (memory = image features, tgt = shifted token embeddings)
        out = self.decoder(tgt=tgt, memory=enc, tgt_mask=mask,
                           memory_key_padding_mask=None)
        return self.head(out)

    def generate(self, images, tokenizer, max_len=256, beam=4):
        # teacher-forced / beam-search autoregressive decoding
        ...
```

**Practical sizing:** an 8–12M parameter decoder on top of a frozen ViT encoder already reaches strong CER after a few days on ~100k synthetic + 20k real lines, single A100 (or 2× RTX 4090).

### 3.3 Preprocessing / restoration for degraded book scans

Pipeline (in order):

1. **Deskew** — estimate skew angle (Hough/radon/Fast Fourier orientation) and rotate. For *curved* scans use a learned **document dewarping** model (e.g., DocUnet, DocTR, or a small transformer dewarp net).
2. **Binarization** — Sauvola (preferred for historical ink), then adaptive Otsu. Keep *two* channels: grayscale + binary. Never destroy grayscale info before recognition.
3. **Ink restoration & denoise** — a small **U-Net / SRResNet** trained with paired *clean-vs-degraded* synthetic data or unsupervised *self-supervised* clean-up (e.g., a CycleGAN or a masked autoencoder inpainting for bleed-through).
4. **De-bleed-through / de-shadowing** — physical bleed-through forms a mirror image of the reverse page; use a **two-channel model** (front + back page) or a learned **shadow-diffusion** removing bright/dark blotches.
5. **Super-resolution** — if the scan is low-DPI (e.g., <200 dpi), upscale ×2 with **Real-ESRGAN / SwinIR** (careful: may introduce hallucinated glyphs). Prefer high-quality native scans.
6. **Cropping & normalization** — crop line at fixed height, normalize contrast, pad to `(B, 3, H, W)`.

**You should keep (or even publish) a *degradation sandbox*** that produces paired `(clean, degraded)` versions of the same page for controlled ablation.

### 3.4 Synthetic data generation (the secret to low-resource OCR)

This is the highest-RoI part of the whole project.

#### 3.4.1 Renderer

1. **Collect real Devanagari/Sanskrit fonts**: Noto Sans Devanagari, Noto Serif Devanagari, **Siddhanta/ITF Devanagari**, **Kalam**, **Rozha One**, *lithographic-style* fonts, and older **Kraken / Tesseract** legacy fonts.
2. **Render a *glyph-correct* page**: To render correct conjuncts, build the string using **e.g. `devanagari` + `indic-layout` + `matra` ordering** or render at the *akṣara* level via `Pillow` + `fontTools` after `unicodedata.normalize("NFC", ..)`.
3. **Use a text-to-image pipeline for printing**: `tesseract`'s `text2image` (has built-in Devanagari configs? — if not, use `kraken.crec` + `blla` segmenter) to create the *clean* page.
4. **Generate *pseudo-real* manuscript pages** from a *line corpus* by taking a clean line image, and then applying **degradation**.

#### 3.4.2 Degradation operators

| Operator | Typical use for old granthas |
|---|---|
| Gaussian / motion blur | camera blur |
| Gaussian + salt-pepper / Poisson noise | film grain, scanner noise |
| Multiplicative illumination gradient | uneven lighting / gutter shadow |
| Color shift / hue drift | yellowing, browning |
| Random regions opacity/fade | ink fade, missing glyph strokes |
| Bleed-through synthesis | overlay mirrored faint page from a reversed "front page" |
| Elastic / perspective warping | curved book pages |
| Random crop + white/black padding | eroding line boundaries |
| Drop parts of glyphs + morphological dilation/erosion | broken/stain-degraded glyphs |
| JPEG / TIFF compression | scan artifacts |

3. **Create a *real-content* synthetic corpus**: normalize real *printed* Sanskrit text (from GRETIL, Wikisource, etc.) into lines/pages, render each with a random font at 200–600 dpi, and apply a **random composition of 3–6 degradation operators**.

4. **Do not ignore *old Devanagari* and *Vedic* diacritics** in the synthetic font set — for Vedic texts render `udātta`, `anudātta`, `svarita`, `Ṛgvedic` accents explicitly.

#### 3.4.3 Label-space augmentation

- Normalize to **NFC**; also produce **NFD-style** decomposition where useful.
- Randomly add/remove *halant* (`्`) ambiguity, alternate between `ः` and `ᳵ`/characters, and vary the presence of *avagraha* (`ऽ`).
- Use a **lexicon** (a large Sanskrit word list, see §5) to replace glyphs with confusable alternatives (e.g., `व` ↔ `भ`, `घ` ↔ `छ`) — this builds an *error model* that helps both OCR training and post-processing.

#### 3.4.4 Real-data fine-tuning (most important)

Synthetic data gives *glyph coverage*; it does **not** generalize to *manuscript texture*. Therefore:

1. Curate **1,000–5,000 human-verified real page->text pairs** from public scans (Internet Archive, eGangotri, Digital Library of India, Sanskrit Manuscripts project).
2. Auto-generate weak labels (existing OCR), then **human-correct** the most error-prone subset.
3. Use **active learning / low-confidence sampling** (rank crops by entropy / uncertainty) to prioritize correction.
4. **Self-training with pseudo-labels** on the remaining in-domain data — but always measure on a *gold* held-out set.

### 3.5 Evaluation of OCR

| Metric | Definition | Why for you |
|---|---|---|
| **CER** | `1 - (1 - (i+s+d)/len(ref))` over characters | The token-level error; best single metric |
| **WER** | word-level edit distance | More interpretable for scholars |
| **UA / WA** | `1 - CER`, `1 - WER` | Easier to report |
| **Sequence Accuracy** | exact line-match fraction | Upper-bound quality signal |
| **DAN / WD** | DAWG / word-delimiter success | Check punctuation & spacing |
| **Metrical validity** | A *śloka* passes if the reconstructed text satisfies the *chandas* meter | Unique to verse OCR |

**Publish CER/WER on** (a) clean printed books, (b) a *degraded* test set, (c) handwritten manuscripts, (d) Vedic-accented pages. **Report per-*boundary* error too** (error within *pada* boundaries) — that catches *sandhi*-induced errors.

**Baseline targets to beat** (from the literature, e.g. the `ihdia/sanskrit-ocr` evaluation): IndicOCR-v2 achieved ~**3.9% CER / 13.9% WER** on *classical Sanskrit* images; Google OCR got ~**7.0% CER / 34.6% WER**; Tesseract Devanagari got ~**13% CER / 52% WER**. Aim for **< 2.5% CER** on print, and *publish* a scientific honest comparison.

---

## 4. Part 2 — NLP & Translation Pipeline

### 4.1 Sanskrit-specific preprocessing

#### 4.1.1 Script normalization

```python
import unicodedata
def normalize_sanskrit(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    # Optionally NFD for OCR alignment; but keep NFC canonical.
    # Legacy digitizers often insert chandrabindu/anusvara inconsistently.
    t = t.replace("\u0951", "")  # optional: strip obsolete zh/vedic marks
    return t
```

#### 4.1.2 Sandhi splitting (the single most impactful linguistic step)

Sanskrit is **not tokenized by whitespace**: e.g. `गच्छति पुरः` may appear fused in an OCR or *saṃhitā-pāṭha* as `गच्छतिपुरः`. The pipeline must produce both:

- **Saṃhitā-pāṭha** (as printed, with sandhi fused — the source of truth for OCR).
- **Pada-pāṭha** (sandhi-split, word-by-word, with case/morphological tags) — needed for translation and scholarly commentary.

Use a **hybrid**:

1. **Rule-based splitter** from **Sanskrit-Parser / Samsaadhanii (UoH) / Heritage Sanskrit Platform** — deterministic, *exhaustive*, but over-generates. It handles ~80–87% of junctions correctly.
2. **Neural sequence-to-sequence splitter**: fine-tune **ByT5-Sanskrit** (Sebastian Nehrdich's model) or **SanskritShala** on the **Digital Corpus of Sanskrit (DCS)** word-segmentation data. ByT5-Sanskrit reaches strong/SoTA results on DCS word segmentation.
3. **Combine**: rule candidates → filter by a **Sanskrit lexicon + morphological analyzer** → keep the globally-consistent path (DP / beam) that maximizes valid tokens.

**In a dual-language pipeline** the splitter must *favor splits that make the source more translatable into Nepali* — i.e., score candidate splits by whether the resulting words exist in your bilingual lexicon.

#### 4.1.3 Morphological analysis & lemmatization

- **Use ByT5-Sanskrit** for lemmatization, morpho-syntactic tagging, and dependency parsing (DCS-trained). It gives you one unified model for *sandhi splitting + lemmatization + morphological tagging + dependency parsing*, which is a great fit for a research paper.
- **Lexicon-backed:** Monier-Williams / Apte / Bhattacharya dictionaries, **Sanskrit Wordnet**, **Ambaa/Sanskrit-Dictionary**, and the **Sanskrit Heritage lexicon**.
- For each *pada*, produce: `lemma`, `pos`, `case/vibhakti`, `number`, `gender`, `tense`, `person`, `voice`, `kṛt/taddhita/upasarga` features.

#### 4.1.4 Meter (chhandas) identification

- Identify the meter of each *śloka* (anuṣṭubh, triṣṭubh, jagatī, gāyatrī, etc.) via **syllable counting + gana analysis** and the *svarita/udātta* marks.
- **Use metre as both a *constraint* and an *evaluation signal***: if the OCR output violates the detected metrical pattern at a *pada*-boundary, this is almost certainly a recognition error, and you can **re-rank OCR candidates** (this "metrically-constrained OCR" is a nice novel contribution).

#### 4.1.5 Sentence/discourse structuring

- Root-text boundaries: `।` (single *daṇḍa*) for prose, `॥` (double *daṇḍa*) for verse ends, but beware historical encoding where `॥` varies.
- **Verse splitting**: split into *pāda* (quarter-verses) with the understanding that the 4 *pādas* may be printed as 2 lines, 4 lines, or as a prose block with visible *daṇḍas*.
- **Commentary alignment**: separate *root śloka/tīkā* from *vyākhyā* (commentary). The translation of the root text and the commentary are different tasks.

### 4.2 Translation architecture

**Recommended two-tier system** (this gives the best balance of quality and publishability):

#### Tier 1 — Strong classical MT backbone: **IndicTrans2**

- **IndicTrans2** (AI4Bharat) explicitly supports **Sanskrit (`san_Deva`)** and **Nepali (`npi_Deva`)** among its 22 scheduled Indic languages, and uses **script unification** so that Sanskrit, Hindi, Nepali, Marathi, and Maithili share a Devanagari sub-space. This is the highest-quality *open-weight* baseline you can start from.
- **Fine-tune IndicTrans2** on your Sanskrit→Nepali parallel data (which you will construct), using **Fairseq** (as prescribed in the IndicTrans2 repo) or the **Hugging Face `mBART`-compatible** conversion.
- **Multilingual transfer trick:** since direct `san→npi` data is scarce, **use `san→hin` and `hin→npi` in the same model**, and **document zero-shot generalization to `san→npi`**. (IndicTrans2 already does script-unified Devanagari, which makes this feasible.)
- **Chhandas-aware decoding:** feed the *meter label* as a control token (`<meter:anuṣṭubh>`). This nudges the model to produce a *translation that maps to the same number of pāda blocks*, not necessarily the same syllables. 

#### Tier 2 — Scholar-commentary LLM (LoRA fine-tuned)

- **Base models to consider** (roughly increasing cost): `Qwen2.5-3B/7B-Instruct`, `Mistral-7B`, `Mistral-NeMo-12B`, `Llama-3.1-8B`, `Gemma-2-9B`, `Qwen2.5-14B/32B`. For the MVP, **Mistral-7B / Qwen2.5-7B with LoRA** is the sweet spot — you can train on 1× A100 80GB or 2× RTX 4090.
- **Why an LLM, not only an NMT model?** The goal is *scholar-commentary* — a translation that (i) preserves the *semantic unit structure*, (ii) explains technical terms, (iii) follows a *vyākhyā* template (gloss → literal → fluent), (iv) handles multi-verse context across pāda boundaries. This is a *conditional text-generation* task, not a sentence-pair translation task.
- **Fine-tuning setup:**
  - **LoRA adapters** on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
  - `r = 32–64`, `alpha = 2r`, `dropout = 0.05`, `lr = 1–2e-4` with cosine, one epoch.
  - **Instruction format**:

```
<sanskrit verse including meter>
Sanskrit verse: ...
Chhandas: anuṣṭubh
Interlinear gloss: ...
Scholar consultation: (optional commentary/hints)
Nepali commentary: ...
```

- **Decode** with top-p (`0.95`) / temperature (`0.7`), and optionally add **A* retry** based on the *commentary-faithfulness* metric.

#### 4.2.1 Why two tiers?

- **IndicTrans2 gives *literal* fidelity** — it keeps case/vibhakti and lexical meaning, and reports strong Indic-translation quality. Good for a *mechanical baseline* and for producing an **interlinear gloss**.
- **LoRA LLM gives *scholar-style* output** — it handles *historical context*, *technical terms*, *multi-verse continuity*, and *commentary style*. Good for the *final deliverable*.
- **Ensemble with a reranker:** score candidate outputs with **COMET / LaBSE similarity / a bilingual lexicon match** (see §5) and pick the best of `{IndicTrans2 literal, LLM commentary, rule-based glossary digest}`.

### 4.3 SOTA modeling options table (for your paper)

| Approach | Pros | Cons | Use in your paper |
|---|---|---|---|
| **CRNN + CTC** | Fast, strong baseline, easy synthetic train | Weak on complex conjuncts, rigid | Baseline OCR |
| **ViT + Transformer decoder (TrOCR)** | Strong, handles context, scalable | Needs data, slower | Primary OCR |
| **Fine-tune IndicTrans2** | High-quality, multilingually grounded | Strong data needs for san→npi | Baseline MT |
| **mBART / M2M-100 fine-tune** | Open, familiar | Less script-unified | Comparison |
| **LoRA LLM (Mistral/Qwen/GPT-NeoX)** | Best for commentary, few-shot, in-context | Costly, harder evaluation | Primary commentary model |
| **Multitask ByT5-Sanskrit** | Best at segmentation/morphology | Not designed for target-language prose | Preprocessing / joint model |
| **Verse-aware prompt + chain-of-thought** | Flexible, handles ambiguity | Harder reproducibility | Main proposal |

### 4.4 Sanskrit→Nepali specifics

- **Word order**: Sanskrit *kāraka* arguments are case-suffixed; Nepali is SOV-ish with postpositions but uses fewer case markers. The translator must **re-map case to postpositions** (`-ने`, `-लाई`, `-मा`, `-को`), and often zero-drop nominal arguments.
- **Verb agreement & honorifics**: Nepali has polite *āp/तपाईं* vs. ordinary *तिमी/तू*. Commentary typically uses **neutral/literary** register.
- **Poetic vs. prose register**: for ślokas, the Nepali rendering should be *metrical-flavored* prose (not strict meter usually), but *commentary on meaning* must stay precise. The paper should explicitly define "scholar commentary" as: *literal gloss + interpretive paraphrase + optional Nyāya/Tarkha-style reasoning*.

---

## 5. Part 3 — Data Strategy & Evaluation

### 5.1 Sourcing

#### 5.1.1 Sanskrit source text (monolingual, for OCR training / LM / lexicon)

| Source | Notes |
|---|---|
| **Digital Corpus of Sanskrit (DCS)** | Manually validated, annotated, classical + Vedic. Core gold resource. |
| **GRETIL** | Göttingen Register of Electronic Texts in Indian Languages. Large classical text corpus. |
| **SARIT** | Sanskrit corpus with TEI markup and translations, primary for *commentary embedding*. |
| **Wikisource Sanskrit** | Open, editable, useful for *printed-page* synthetic rendering. |
| **Sanskrit Documents / Sacred-Texts** | Public-domain Unicode texts. |
| **Muktabodha** | Digital library with many *śāstra* texts. |
| **Sanskrit Wordnet / DCS lemma+cases** | Lexicon / morphology. |
| **Internet Archive / eGangotri / Digital Library of India / DIV** | Scanned *pages* (raw OCR input). |

#### 5.1.2 Sanskrit→Nepali parallel (the hard part)

**There is very little public Sanskrit→Nepali parallel data.** This is the most important data contribution you can make, and it should be a headline contribution in your paper. Build it yourself:

1. **Pivot approach (linguistically reasonable):**
   - Use available `san→hin` parallel **test** data (e.g., existing Sanskrit-Hindi MT benchmarks), then mine/back-translate `hin→npi` from Samanantar / OPUS / CLE.
   - *Important caveat:* do **not** treat pivot translation as gold. Use it only for *model pretraining / silver corpus*, and isolate **gold bilingual expert annotations**.
2. **Seeded seed corpus from Nepali publications:** works on Sanskrit *granthas* are sometimes published with **Nepali commentary** (especially Gita / Upanishad editions). Digitize those parallel pages.
3. **Manual expert annotation process:**
   - Create a **wiki-like annotation interface** where a Sanskrit scholar and a Nepali scholar jointly annotate: verse → interlinear gloss → (optional) commentary notes → final Nepali commentary.
   - Aim for **5,000–10,000 verses** for a defensible study. Even **1,000 high-quality verses** is a strong low-resource dataset.
4. **Exploit *existing translations* of known texts:** Many editions of the *Bhagavad Gita*, *Ramayana*, *Upanishads*, and *Sankhya* treatises already exist in both Sanskrit and Nepali. Text-level alignment can be lifted to verse/sentence level with **Laser-BERT / mBERT** and **fast-align / awesome-align**.

#### 5.1.3 Alignment & quality filtering

1. **Normalize:** Unicode NFC, collapse whitespace, strip diacritics vs. Devanagari, remove page numbers.
2. **Sentence/verse align:** use **mBERT/LaBSE embeddings + centroid clustering** (like Samanantar's ASR/ANNS pipeline), then **awesome-align** for word alignment.
3. **Quality gates** (in order):
   - Remove empty/misaligned, length ratio > 4×, language dropout, and near-duplicates.
   - **LaBSE / LAS score** threshold (keep pairs with similarity >0.7 for *near-silver*, >0.85 for *silver*).
   - **Keyword/lexicon coverage** — keep only pairs where a high % of content words map into the bilingual Sanskrit-Nepali lexicon.
4. **Data versioning:** use **DVC (or Git LFS)** for image/corpus artifacts. **Never** commit large binary datasets to git.
5. **Dataset card:** document provenance, licenses, languages, splits, annotation protocol, and any **sensitive cultural/relic** issues. Do **not** republish scans from commercial/licensed archives without permission.

### 5.2 Data schema (recommended)

```yaml
# san-parallel.example.yaml (or sqlite/jsonl)
id: "gita-02-47"
source:
  page_uri: "archive.org:gita1872/page/42"
  page_index: 42
  scan_dpi: 324
ocr:
  raw_text: "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।"
  cer: 0.012
  wer: 0.033
  confidence: 0.94
reconstruction:
  pada_patha: "कर्मणि एव अधिकारः ते मा फलेषु कदाचन"
  meter: "anuṣṭubh"
  morph: [["कर्मणि","saptami","sg",...], ...]
translation:
  gloss_nepali: "कर्म मा मात्र तिम्रो अधिकार छ; फलमा कदापि होइन।"
  commentary_nepali: "कर्मणि एव — कर्ममा मात्र... । अर्थात्, कर्तव्य-पालनमा मात्र तिम्रो अधिकार छ; फलप्राप्तिमा होइन।"
  notes: "श्रीमद्भगवद्गीता 2.47 — कर्मणई अधिकार, फलवैराग्य सिद्धान्त"
```

### 5.3 Evaluation metrics

#### 5.3.1 OCR

- **CER/WER** (edit distance after normalization)
- **Note the *normalized-string* caveat:** Devanagari NFC/NFD and old *pāṭha* encodings (e.g., long `ः` vs `ᳵ`) make CER misleading. Use **the normalized, canonical NFK mapping** for all scoring.
- **Metrical-validity score** as a *task-aware metric* (unique to verse): `mean(valid candidate pāda-sequences)`. Publish how much metre captures residual CER errors that plain CER misses.

#### 5.3.2 Translation

| Metric | What it measures | Recommended |
|---|---|---|
| **BLEU / SacreBLEU** | n-gram overlap | Baseline, not enough alone |
| **chrF** | character-level overlap | Better for morphologically rich languages |
| **METEOR** | paraphrase/word-form matching | Good additional proxy |
| **RIBES** | word-reorder-based similarity | Important for Indic word-order |
| **COMET** (e.g., `wmt22-cometinho-da` or `Unbabel/wmt22-comet-da`) | neural semantic quality | **Best low-cost MT quality proxy** (check Nepali coverage) |
| **LaBSE / multilingual-E5 / XLM-R** sentence cosine | semantic similarity | Cheap, works in Devanagari space |
| **Human adequacy / fluency / fidelity** | 1–5 scholar ratings | Gold metric |
| **Commentary-Faithfulness (CF)** | % of key interpretive units correctly conveyed | Your north-star |

**A practical evaluation stack:**
1. *Automated:* COMET (or LaBSE cosine) + SacreBLEU + chrF + METEOR.
2. *Human:* a **Man-Machine-internal panel** of ≥2 bilingual scholars scoring adequacy (does this translation preserve the original meaning?) and commentary-style fidelity (does it sound like a scholarly vyākhyā?).
3. *Ablation:* compare your fine-tuned LoRA LLM vs. zero-shot LLM, vs. IndicTrans2, vs. LLM+retriever, and report **CF**.

### 5.4 Evaluation split design

- **Gold test sets:** 500–1,000 verses, ~50% prose, 50% chhandas, including Vedic-accented lines where possible.
- **Degraded test set:** the same content rendered with the degradation sandbox + real scans. Separate CER for `clean`, `degraded`, `real`.
- **Per-domain:** śāstras (grammar/nyāya/vedānta) vs. poetry vs. narrative — all in your paper.

---

## 6. Part 4 — GitHub Project Structure & Deployment

### 6.1 Ideal repository layout

```
sanskrit-nlp/
├─ LICENSE
├─ README.md
├─ pyproject.toml
├─ setup.cfg
├─ ruff.toml / .flake8
├─ .pre-commit-config.yaml
├─ .gitignore
├─ Makefile
├─ Dockerfile
├─ docker-compose.yml
├─ .github/workflows/
│  ├─ ci.yml
│  ├─ data.yml
│  └─ deploy.yml
├─ data/
│  ├─ README.md                # explain every source & license
│  ├─ raw/                     # ignored (use DVC)
│  ├─ processed/               # cleaned jsonl/parquet
│  └─ gold/                    # human-verified test sets
├─ configs/
│  ├─ base.yaml
│  ├─ ocr.yaml
│  ├─ mt.yaml
│  ├─ llm.yaml
│  └─ eval.yaml
├─ src/
│  ├─ __init__.py
│  ├─ consts.py
│  ├─ utils/
│  │  ├─ config.py
│  │  ├─ logging.py
│  │  ├─ normalize.py
│  │  └─ io.py
│  ├─ ocr/                     # API: ocr pipeline
│  │  ├─ api.py
│  │  ├─ detector.py
│  │  ├─ recognizer.py
│  │  ├─ lineclean.py
│  │  ├─ post.py
│  │  ├─ synthetic.py
│  │  └─ metrics.py
│  ├─ sandhi/                  # Sanskrit sandhi splitting
│  │  ├─ split.py
│  │  ├─ rule_based.py
│  │  ├─ neural.py
│  │  └─ byt5_utils.py
│  ├─ morph/                   # lemmatization, tagging
│  │  ├─ lemma.py
│  │  └─ tagger.py
│  ├─ meter/                   # chhandas detection & validation
│  │  ├─ detect.py
│  │  └─ validate.py
│  ├─ nlp/                     # MT / commentary generation
│  │  ├─ translate.py
│  │  ├─ lora_finetune.py
│  │  ├─ prompt.py
│  │  └─ rerank.py
│  ├─ eval/
│  │  ├─ ocr_eval.py
│  │  ├─ mt_eval.py
│  │  ├─ human.py
│  │  └─ cf.py
│  ├─ data/
│  │  ├─ sources.py
│  │  ├─ align.py
│  │  └─ synth.py
│  └─ api/
│     ├─ app.py
│     ├─ schemas.py
│     └─ routes/
├─ scripts/
│  ├─ download_data.py
│  ├─ build_synth.py
│  ├─ train_ocr.py
│  ├─ train_mt.py
│  ├─ finetune_llm.py
│  └─ run_eval.py
├─ notebooks/
│  ├─ 01_explore.ipynb
│  ├─ 02_eval.ipynb
│  └─ 03_paper_figures.ipynb
├─ experiments/
│  ├─ README.md
│  ├─ 2026-09-05_run_1/
│  ├─ ...
│  └─ mlflow.db
├─ tests/
│  ├─ test_sandhi.py
│  ├─ test_meter.py
│  ├─ test_ocr_metrics.py
│  └─ test_api.py
├─ docs/
│  ├─ GRANTHA_DIGITIZATION_BLUEPRINT.md
│  ├─ DATA_SOURCES.md
│  ├─ MODEL_CARDS.md
│  └─ PAPER_OUTLINE.md
├─ deployment/
│  ├─ Dockerfile
│  ├─ k8s.yaml
│  └─ nginx.conf
└─ ui/
   ├─ streamlit_app.py
   └─ gradio_app.py
```

### 6.2 Best practices that will make this publishable

- **Configuration-driven experiments** (`hydra` or `pydantic + YAML`). Never hardcode paths/params.
- **Pinned environment & container.** Use a `Dockerfile` with CUDA 12.x, PyTorch 2.x, and a locked `requirements.lock`.
- **Experiment tracking:** **MLflow (or W&B)** for metrics; logs/checkpoints in `experiments/`.
- **Data & model versioning** with **DVC** / **HuggingFace Hub**; `data/raw` and `data/processed` are git-ignored.
- **CI/CD:** GitHub Actions that run `pytest`, `ruff`, `mypy`, a small smoke end-to-end test on 5 pages, and only then build the Docker image.
- **Model cards & Datasheet:** every fine-tuned model and dataset gets a card (language, license, training data provenance, known failure cases, ethical notes).
- **Reproducibility for the paper:** set `torch.manual_seed`, `seed`, `deterministic`, and record all configs in a `runs/<run_id>/config.yaml`.
- **Human evaluation harness:** export JSONL for an annotation tool, with inter-annotator agreement (Krippendorff's α / Cohen's κ).

### 6.3 Deployment

#### 6.3.1 User-facing app (MVP)

- **Streamlit/Gradio** real-time demo which accepts `(a URL to PDF/JPG/TIFF)` or `(upload)`, runs the OCR → reconstruction → translation → commentary pipeline, and displays:
  - Page image with detected blocks/lines overlays.
  - OCR text with confidence coloring (red = low confidence).
  - `saṃhitā-pāṭha` + `pada-pāṭha` + morphological tags.
  - Nepali commentary with `literal gloss` + `scholar annotation`.
  - Export as `.txt`, `.json`, `.docx`, `.html`.
- **API:** **FastAPI** exposing `POST /process`, `GET /jobs/{id}`. Use **Celery + Redis** for page queues and **object storage** (S3/MinIO) for PDFs/images.
- **Production ETL:** a **Docker Compose** stack (or Kubernetes) with a dedicated CPU worker for pre-processing and a GPU worker for inference, plus an indexing service (**Elasticsearch/Weaviate/Qdrant**) so scholars can search digitized corpus and commentary.
- **HuggingFace Spaces** for a shareable demo; **ONNX** or **TORCHScript** export for cheaper CPU inference.

#### 6.3.2 Reliability

- **Human-in-the-loop correction queue:** low-confidence OCR lines, sandhi-unsolved peaks, and low-fidelity translations are routed to a review queue; corrections become future training data.
- **Offline batch mode:** a `scripts/process_book.py` for a full PDF in one run, plus chunked output and checkpoint-resume.

---

## 7. Part 5 — Research Paper Framework

### 7.1 Proposed title

> **"Gītā-Śāstra-Nepālī: An End-to-End Open Pipeline for OCR, Reconstruction, and Scholar-Style Sanskrit-to-Nepali Commentary Translation of Ancient Granthas"**

(Or: *"Transcribing and Translating Ancient Granthas: Metrically-aware OCR and Linguistically Grounded Sanskrit→Nepali Commentary MT."*)

### 7.2 Target venues / suggested paper type

- **LREC-COLING** (resources + evaluation) — best fit since the main contribution is *data + tooling*.
- **ACL Rolling Review / EMNLP main** — if you can show *CF* superiority over IndicTrans2 and strong ablations.
- **ICDAR / Document Analysis** — if you emphasize the *page-level document-analysis* contribution (layout + metric-aware OCR).
- **WMT shared task (Indic language translation)** — if you release the gold Sanskrit→Nepali corpus.

### 7.3 Complete outline

```
Title
Authors & affiliations
Abstract
1.  Introduction
    1.1 Why digitizing Sanskrit granthas matters (cultural preservation, scholarly access)
    1.2 Why it is technically hard (morphology, metre, degradation, low-resource)
    1.3 Contributions (C1 OCR, C2 sandhi/morphology, C3 Sanskrit→Nepali commentary MT,
         C4 dataset, C5 human-evaluation protocol)
2.  Related Work
    2.1 Historical OCR for Indic/Devanagari
    2.2 Document analysis for historical books
    2.3 Sanskrit NLP (sandhi splitting, lemmatization, dependency parsing)
    2.4 Low-resource Indic MT and LLM fine-tuning
    2.5 Human evaluation & culturally-aware language generation
3.  Problem Definition & Task Setup
    3.1 Inputs/outputs; saṃhitā-pāṭha vs. pada-pāṭha; chhandas vs. prose
    3.2 Commentary-faithfulness (CF) formal definition
4.  Data
    4.1 Monolingual Sanskrit sources
    4.2 Sanskrit→Nepali parallel creation (pivot, seed, manual)
    4.3 Synthetic OCR data (fonts, degradation)
    4.4 Annotation protocol, human agreement
    4.5 Splits, licensing, dataset card
5.  Methodology
    5.1 OCR: layout, line segmentation, recognizer, post-processing
    5.2 Metrically-aware reconstruction
    5.3 Sandhi splitting & morphological analysis (hybrid rule + ByT5)
    5.4 MT: IndicTrans2 baseline, LoRA LLM, prompt-based commentary
    5.5 Reranking / ensemble
6.  Experiments
    6.1 Setup (hardware, hyperparameters, seeds)
    6.2 OCR baselines (Tesseract, Kraken, IndicOCR-v2, Google OCR)
    6.3 OCR results (CER/WER on clean/degraded/handwriting)
    6.4 Sandhi & morphology impact (on-text F1)
    6.5 MT results (BLEU/chrF/METEOR/COMET/CF)
    6.6 Ablation: synthetic data fraction, control token, prompt format, LoRA rank
    6.7 Error analysis (per-pāda, per-domain, per-glyph)
7.  Discussion
    7.1 Which errors remain (mātrā, accents, missing verses)
    7.2 Where CF breaks
    7.3 Cultural & ethical considerations (manuscript ownership, religious sensitivity)
8.  Conclusion & Future Work
9.  Reproducibility & Responsible Use
    9.1 Code/dataset release
    9.2 Compute/environment
    9.3 Ethics
References
Appendix A: Full data schema
Appendix B: Detailed model card
Appendix C: Human annotation template
Appendix D: Additional result tables
```

### 7.4 Example result table template

| Model / config | OCR CER | OCR WER | BLEU | chrF | COMET | CF (Human) |
|---|---|---|---|---|---|---|
| Tesseract-Devanagari baseline | 13.2 | 52.8 | — | — | — | — |
| IndicOCR-v2 | 3.86 | 13.86 | — | — | — | — |
| **Our OCR (ViT+decoder)** | **2.1** | **8.4** | — | — | — | — |
| **+ metrical reranking** | **1.3** | **5.2** | — | — | — | — |
| IndicTrans2 (zero-shot san→npi) | — | — | 12.4 | 28.1 | 0.51 | 0.42 |
| LoRA Mistral-7B (poetic only) | — | — | 14.2 | 33.0 | 0.62 | 0.58 |
| **Our full pipeline + reranker** | — | — | **17.1** | **38.5** | **0.72** | **0.79** |

(Representative illustrative numbers — you must fill with real results.)

### 7.5 Papers to cite (anchor your related work)

- TrOCR (Li et al., 2021) — "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models."
- IndicTrans2 (2023) — "IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages."
- Samanantar (Ramesh et al., 2022), **Aksharantar** (Madhani et al., 2022), **IndicCorp**.
- ByT5-Sanskrit (Nehrdich, 2024) — "One Model is All You Need: ByT5-Sanskrit."
- SanskritShala (Tukeyev et al.), **DCS** (Hellwig).
- **Sanskrit-Parser / Samsaadhanii** for sandhi and morphology.
- **IndicOCR-v2 / ihdia** for classical Sanskrit OCR baselines.
- **mmOCR / PaddleOCR / Kraken** for document analysis.

---

## 8. MVP Roadmap & Milestones

### MVP-0 — Proof of feasibility (2–3 weeks)

- Install `Tesseract` Devanagari + `kraken_devanagari`.
- Process 10 scanned pages via an *existing* OCR (Tesseract, PaddleOCR) as a baseline.
- Run IndicTrans2 on a 100-verse seed corpus to see *rough* zero-shot `san→npi`.
- **Acceptance criterion:** end-to-end demo where 10 pages produce readable Devanagari + rough Nepali; document the failure modes.

### MVP-1 — Research vertical slice (4–6 weeks)

- Build line-detection + TrOCR/CRNN line recognizer.
- Build the **sync data generator** (fonts + degradation) and train on ~50k synthetic lines.
- Implement the `pada-pāṭha` splitter (rule-based + lexicon).
- Produce a hard-coded 100-verse `san→npi` gold test set.
- **Acceptance criterion:** **CER < 5%** on clean printed books; baseline IndicTrans2 `san→npi` evaluated.

### MVP-2 — Phase 1 of research system (6–8 weeks)

- Real-data fine-tuning with 5k–10k human-verified pages.
- Add **metrical validation + reranking**.
- Fine-tune IndicTrans2 on `san→npi` + pivot `san→hin`, `hin→npi`.
- Fine-tune **LoRA LLM** (Mistral-7B/Qwen-7B) on instruction dataset of 5k–10k commentary pairs.
- Build **FastAPI + Streamlit** demo and Docker.
- **Acceptance criterion:** **CER < 2.5%** on *clean* and **<7%** on *degraded*; **CF > 0.6** on 100-verse gold; reproducible GitHub CI.

### MVP-3 — Submission-ready prototype (8–12 weeks)

- Expand gold dataset to 1,000+ verses; add prose.
- Add **Vedic-accented** OCR data and commentary extraction.
- Run full ablation study; prepare paper figures + code/data release.
- **Acceptance criterion:** submission-ready paper with all tables, code, data card, and a live demo link.

### Rough timeline table

| Week | Deliverable | Owner |
|---|---|---|
| 1–2 | Repo scaffold, data sources inventory, Tesseract baseline | You |
| 3–4 | Synthetic data generator + TrOCR line encoder train/test | You |
| 5–6 | Sandhi/morphology + meter validate | + Scholar collaborator |
| 7–8 | San→Nepali corpus construction + IndicTrans2 fine-tune | You |
| 9–10 | LoRA LLM commentary fine-tune + reranker | You |
| 11–12 | Human eval, CF metric, ablation, paper outline | You / Scholars |
| 13–14 | Docker deployment + demo + release | You |

---

## 9. Engineering Recommendation Summary

- **Language / stack:** Python 3.10+, PyTorch 2.2+, `transformers`, `torchvision`, `opencv-python`, `albumentations`, `huggingface_hub`, `datasets`, `mlflow`/`wandb`, `fastapi`, `uvicorn`, `gradio`/`streamlit`, `dvc`, `pyyaml`, `pydantic`.
- **OCR model:** **Vit + weighted cross-entropy autoregressive decoder** (TrOCR-style, frozen pretrained encoder + small decoder) with a **CRNN/CTC** fallback.
- **Sanskrit NLP:** **ByT5-Sanskrit** for SWS/lemma/tagging/dependency; **Sanskrit-Parser + Samsaadhanii** for rule-based sandhi; **custom lexicon filter**.
- **MT/commentary:** **IndicTrans2** for *literal* baseline; **LoRA-LLM** for *commentary*; **retrieval/reranking** with COMET/LaBSE.
- **Data:** synthetic OCR from real fonts + degradation; pivot corpus from `san→hin`/`hin→npi`; manual gold `san→npi`; DVC-versioned.
- **Evaluation:** CER/WER + metrical validity for OCR; SacreBLEU/chrF/METEOR/COMET/LaBSE + human `CF` for translation.
- **Deployment:** FastAPI + Celery + S3 + Streamlit demo + Docker; GPU worker for inference.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Saṃhitā-pāṭha** | Continuous text as printed, with sandhi intact (e.g., `संधि` as written). |
| **Pada-pāṭha** | Word-segmented text after sandhi splitting. |
| **Sandhi** | Phonological fusion at word boundaries in Sanskrit. |
| **Pāda** | Quarter-verse of a *śloka* (metrical unit). |
| **Chhandas** | Vedic/classical metre; the rhythmic structure of verse. |
| **Kāraka** | Semantic case role in Sanskrit grammar (kartā, karma, karaṇa...). |
| **Vibhakti** | Grammatical case (nominative, accusative, etc.). |
| **Vyākhyā** | Scholarly commentary/interpretation. |
| **Tīkā** | Commentary composed in a specific style. |
| **Cf (Commentary-Faithfulness)** | Your north-star metric: fraction of key interpretive units correctly conveyed. |
| **CER/WER** | Character / word error rate. |
| **DCS** | Digital Corpus of Sanskrit. |

---

## 11. References & Useful Resources

**Core resources referenced above:**

- **IndicTrans2** — https://github.com/AI4Bharat/IndicTrans2
- **Samanantar / IndIC** — https://indicnlp.ai4bharat.org/samanantar/
- **Aksharantar** — https://huggingface.co/datasets/ai4bharat/Aksharantar
- **ByT5-Sanskrit** — https://github.com/sebastian-nehrdich/byt5-sanskrit-analyzers
- **Digital Corpus of Sanskrit (DCS)** — https://www.dcs.uni-heidelberg.de/
- **GRETIL** — https://gretil.sub.uni-goettingen.de/
- **SARIT** — https://sarit.indology.info/
- **Sanskrit-Parser** — https://github.com/kmadathil/sanskrit_parser (PyPI: `sanskrit-parser`)
- **SanskritShala / Samsaadhanii (UoH)** — https://sanskrit.uohyd.ac.in/scl/
- **Kraken Devanagari** — https://github.com/Shreeshrii/kraken_devanagari
- **IndicTrOCR** — https://github.com/iitb-research-code/indic-trocr
- **IndicOCR-v2 (ihdia)** — https://github.com/ihdia/sanskrit-ocr
- **TrOCR** — https://github.com/microsoft/unilm/tree/master/trocr
- **PaddleOCR** — https://github.com/PaddlePaddle/PaddleOCR
- **mmOCR** — https://github.com/open-mmlab/mmocr
- **COMET** — https://github.com/Unbabel/COMET
- **LaBSE** — https://huggingface.co/sentence-transformers/LaBSE
- **Indian Language / Sanskrit docs** — https://sanskritdocuments.org/

---

## Appendix A — Minimal start-up command plan

```bash
# 1. Clone & setup
git clone https://github.com/<your-user>/sanskrit-nlp.git
cd sanskrit-nlp
pip install -e ".[dev]"

# 2. Download a small text corpus
python scripts/download_data.py --source gretil --subset gita --out data/raw/gita

# 3. Build synthetic OCR pages
python scripts/build_synth.py --config configs/ocr.yaml --out data/synthetic

# 4. Train a line OCR recognizer
python scripts/train_ocr.py --config configs/ocr.yaml

# 5. Build the sandhi/splitting package
python scripts/train_mt.py --config configs/mt.yaml --mode indictrans2

# 6. Fine-tune a LoRA commentary model
python scripts/finetune_llm.py --config configs/llm.yaml

# 7. Evaluate on the gold test set
python scripts/run_eval.py --config configs/eval.yaml
```

## Appendix B — Example config (`configs/eval.yaml`)

```yaml
seed: 42
device: cuda:0
ocr:
  test_sets:
    - data/gold/clean/x.jsonl
    - data/gold/degraded/x.jsonl
  metrics: [cer, wer, sequence_accuracy]
  meter_check: true
mt:
  src_lang: san
  tgt_lang: npi
  test: data/gold/san_npi/test.jsonl
  metrics: [bleu, chrf, meteor, comet, labse, cf]
  comet_model: wmt22-comet-da
panel:
  annotators: 2
  scale: [1, 2, 3, 4, 5]
  interannotator: krippendorff_alpha
```

---

*This guide is a living blueprint. Iterate on it as the pipeline matures; keep the data cards, model cards, and evaluation protocol updated so that your project is both a strong GitHub portfolio artifact and a publication-ready research contribution.*
