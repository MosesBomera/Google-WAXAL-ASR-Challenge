# Experiment‑02 — MMS‑1B (CTC) fine‑tune for WAXAL (Lingala · Shona · Luganda)

> **Thesis (grounded in [WAXAL‑NET, arXiv 2606.02375](https://arxiv.org/pdf/2606.02375)):** our three languages are all **Bantu**, and on the WAXAL benchmark **CTC models (MMS) win or tie on every Bantu language, lead CER in 17/19 languages, and are structurally immune to the repetition‑loop hallucination that caps Whisper.** Experiment‑01 (Whisper‑large‑v3 LoRA) hit **0.74** and stalled on Lingala. Experiment‑02 pivots the backbone to **`facebook/mms-1b-all`** and goes all‑out: full data cleaning, maximum usable model capacity, a **cased+punctuated** CTC vocabulary, native long‑audio handling, and **KenLM‑guided decoding** — the last of which directly targets Lingala's word‑boundary weakness.
>
> **Target:** beat 0.74, with the biggest gains expected on **CER** (half the metric) and **Luganda/Shona**, plus an LM‑driven **Lingala WER** improvement Whisper couldn't deliver.

---

## 1. Why MMS‑1B, and why CTC changes the game

| Property | Whisper (exp‑01, AR seq2seq) | **MMS‑1B (exp‑02, CTC)** |
|---|---|---|
| Architecture | encoder → autoregressive decoder | wav2vec2 encoder → **frame‑level CTC head** |
| Bantu fit ([2606.02375](https://arxiv.org/pdf/2606.02375)) | loses to CTC on all 6 Bantu | **wins/ties on all 6 Bantu** |
| CER | good | **leads 17/19 languages** |
| Failure mode | **repetition loops / hallucination** (silence, OOD) | **immune** — worst case is a recoverable phonetic approximation |
| 30 s cap | **hard cap → we had to chunk & dropped 6% of training** | **no cap** — handles long audio natively |
| Word boundaries (Lingala's wall) | fixed by the decoder's LM prior only | **can bolt on an explicit n‑gram LM at decode** → direct lever |

> **Concept — CTC (Connectionist Temporal Classification).** Instead of generating text token‑by‑token, a CTC model outputs, for every ~20 ms audio frame, a probability over the character vocabulary (plus a special "blank"). A collapse rule (merge repeats, drop blanks) turns the frame sequence into text. Consequences that matter for us: (a) **no fixed input‑length cap** — long clips just mean more frames; (b) **no autoregressive loop** → it physically cannot hallucinate a repeating sentence; (c) decoding can be combined with a **word‑level language model** to fix spelling/word‑boundary errors — exactly Lingala's problem.

> **Concept — MMS adapters vs. our need.** `mms-1b-all` is a wav2vec2 backbone with a small **per‑language adapter + CTC head** for 1000+ languages; normally you tell it the language and it swaps in that adapter. **Phase 2 gives us no language id**, so we can't route per‑language at test time. We therefore fine‑tune **one shared model** (single CTC head over a shared vocab) on all three languages jointly — a deliberate departure from MMS's per‑language design, required by the audio‑only constraint.

---

## 2. Design decisions — the strongest config

### 2a. Model & capacity ("use as much of the model as possible")

**Evidence‑based decision: full transformer fine‑tuning** (not the paper's frozen‑encoder adapter). WAXAL‑NET froze the encoder because it was building **300 M edge models for phones** — an *edge* constraint, not an accuracy‑optimal choice. The adapter‑vs‑full‑FT question is really about **data volume**, and we're on the right side of it:

- The official MMS guide: *"Adapter training … yields better performance **for low‑resource languages**. However, **for medium‑to‑high resource languages it can be advantageous to fine‑tune the whole checkpoint.**"* ([HF MMS adapters](https://huggingface.co/blog/mms_adapters)); MMS wins only when data is *extremely* small (~minutes–1 h), with XLS‑R at parity past ~1 h.
- Our data is **~40 h (lug) to ~100 h (lin/sna) per language** — solidly *medium‑resource*, past the regime where adapters dominate → **full FT favored.**

Config:
- **Backbone:** `facebook/mms-1b-all` (1B, African‑adapted, mature HF support). *(Alternatives noted: `mms-1b-fl102` slightly better zero‑shot in one study; **Omnilingual‑1B** has the best WAXAL zero‑shot but immature fairseq2 tooling and no fine‑tuned track record — higher upside, higher risk. Stay on `mms-1b-all` unless it stalls.)*
- **Scope:** fully fine‑tune the **transformer** + a fresh **cased CTC head**; **freeze only the CNN feature extractor** — the canonical strong wav2vec2/XLS‑R recipe ([HF XLS‑R blog](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)): freezing the conv front‑end "preserves rich acoustic priors, removes 300 M+ params from optimization, and gives stability." Optionally freeze the transformer for the first ~10% of updates, then unfreeze (an XLS‑R stability trick).
- **Cheap control (not a big ablation):** run **one** adapter‑only fine‑tune in parallel. Evidence says full‑FT should win at our data size, but it's data‑dependent enough to verify once; keep whichever is better on held‑out CER/WER.
- **Memory (A100‑40GB):** 1B in bf16 (~2 GB) + grads + **8‑bit AdamW** + **gradient checkpointing** fits at per‑device batch 4–8 with grad‑accum; bucket by duration since wav2vec2 activations scale with audio length.

### 2b. Cased + punctuated vocabulary (protects CER regardless of Zindi's normalization)
- **Build the CTC character vocab from the cleaned training transcripts**, **keeping case and punctuation** (MMS's own pipeline lowercases and strips punctuation — we do **not**, because half our metric is CER on cased, punctuated references).
- Vocab = the union of characters actually present across lin/sna/lug transcripts (letters upper+lower, digits, punctuation, any diacritics like `ɛ ɔ ŋ` if present) + CTC specials: `<blank>`, `<unk>`, `<pad>`, and the word‑delimiter `|` (space).
- One **shared** vocab across the three languages (single head, language‑agnostic) — validated by the fact that a shared Latin‑script vocab covers all three.
- This makes MMS emit cased+punctuated text directly, so it competes on CER whether or not the leaderboard normalizes. (We still run the "does Zindi lowercase?" check from the exp‑01 report — but the cased vocab means we win either way.)

### 2c. Data‑cleaning pipeline (validated on WAXAL by [2606.02375 §3.2](https://arxiv.org/pdf/2606.02375))
Run once, produce a cleaned manifest reused by all runs:
1. **Drop corrupt/empty** audio or transcripts.
2. **Duration floor:** drop clips **< 1.5 s**.
3. **Speech‑rate filter:** drop clips where `num_words / duration > 4` (physically impossible → corrupt reference). *The paper reports this alone took Lingala WER from 113.5% → 49.0%.*
4. **Truncated‑reference check:** scan our three languages for the mid‑sentence truncation artifacts the paper found (references that stop while audio continues) — flag/drop; report the count per language.
5. **Text normalization:** NFC unicode, unescape CSV quotes, collapse whitespace — **keep case & punctuation** (unlike MMS default).
6. **Dedup** exact `(audio, text)`.
7. **Model‑agreement cleaning (optional, from MMS §3.1.6):** score training labels with our exp‑01 Whisper model (or the first MMS model), drop the worst‑agreement tail — targets noisy Lingala labels.
- Emit a **per‑language before/after report** (counts, hours, drop reasons) so we know exactly what changed.

### 2d. Managing clips longer than 30 s (the concern that motivated this pivot)
CTC removes the hard cap, so we have options Whisper didn't:
- **Train directly on longer clips** up to a memory‑bounded limit (e.g. ≤ 40 s) — no truncation, no dropping. Bucket by duration to keep batches memory‑even.
- **Very long clips** (rare): segment with **MMS forced alignment** ([2305.13516 §3.1.3](https://arxiv.org/pdf/2305.13516), native `lin/lug/sna` support) or VAD into ≤ ~30 s aligned pieces — recovering the ~6% experiment‑01 dropped (disproportionately Luganda).
- **Inference:** CTC transcribes arbitrary‑length audio in one pass → **no chunking/stitching needed at test time** (a clean win over Whisper's windowing).

### 2e. Training config (CTC) + regularization
- **Loss:** CTC loss (`Wav2Vec2ForCTC`), `ctc_zero_infinity=True` (guards against inf loss on bad alignments).
- **Optimizer:** 8‑bit AdamW, lr ~1e‑4 for the head / ~1e‑5–3e‑5 for the unfrozen transformer (lower LR for pretrained layers), warmup ~10%, linear/cosine decay, weight decay 0.01.
- **Regularization (counter the exp‑01 overfit):** **SpecAugment / masking** (wav2vec2 has built‑in `mask_time_prob`, `mask_feature_prob`), **layerdrop** (~0.05–0.1), attention/hidden dropout ~0.1.
- **Precision/memory:** bf16 + gradient checkpointing; freeze the CNN feature extractor.
- **Sampling:** keep temperature sampling (α≈0.5) for the Luganda imbalance; monitor whether Lingala needs an explicit up‑weight.
- **Epochs:** up to ~10–15 with early stopping on the combined metric (CTC fine‑tunes need more epochs than seq2seq LoRA; the paper used up to 30 with patience).
- **Checkpoint selection:** per‑language WER/CER, `combined` (macro), same discipline as exp‑01. Auto‑zip best‑N before teardown.

### 2f. Decoding — the Lingala WER lever
- **Baseline:** CTC **greedy** (argmax + collapse) — fast, no LM.
- **Then add an n‑gram LM + beam search** via **`pyctcdecode` + KenLM**:
  - Build a **word‑level KenLM** (3–5‑gram) from the WAXAL **training transcripts** (per‑language or pooled), optionally augmented with other **open, disclosable** text in these languages.
  - Beam search over CTC logits with the LM re‑scores word boundaries — **this directly attacks Lingala's compound‑splitting / morpheme‑boundary errors** ([2606.02375 §5.3](https://arxiv.org/pdf/2606.02375)), which the seq2seq Whisper decoder could not.
  - Tune `alpha` (LM weight) / `beta` (word bonus) / beam width on held‑out.
- **Cased output note:** KenLM/pyctcdecode operate on our cased vocab; we keep casing through decoding.

### 2g. Language handling
- **Single shared model, no language id at inference** (Phase‑2‑safe). We do **not** use MMS's per‑language adapter routing; the shared cased head + jointly‑trained transformer works from audio alone.

---

## 3. Evaluation & metric
- Reuse the exp‑01 harness: **per‑language WER/CER, macro + micro**, on the full validation set (final) and a balanced subset (during training). CTC eval is *fast* (no autoregressive generation) — a big iteration‑speed win over Whisper.
- **Cased vs normalized:** report both; run one controlled leaderboard submission to learn Zindi's normalization (case/punct sensitivity), then align our reported metric to it.
- Public test set stays **evaluation‑only** (never trained on), per the standing rule.

## 4. Staged plan
- **S1 — Data pipeline:** cleaning (§2c) + cased vocab (§2b) + duration audit + long‑clip policy (§2d). Ship the per‑language before/after report.
- **S2 — MMS‑1B fine‑tune (CTC):** full‑transformer FT (freeze CNN), regularized (§2e), per‑language WER/CER checkpointing. Ablate frozen‑encoder vs full‑transformer. Submit greedy‑decoded.
- **S3 — LM decoding:** KenLM + pyctcdecode beam search; tune α/β/beam; measure the Lingala WER delta. Submit.
- **S4 — Recover long clips + optional label cleaning:** fold in forced‑aligned >30 s segments and model‑agreement‑cleaned data; retrain.
- **S5 — Ensemble with Whisper‑v1:** combine MMS (Bantu/CER, LM word boundaries) + Whisper‑large‑v3 (AR disambiguation) — the family‑split finding says they're complementary.

## 5. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Full‑transformer FT overfits (exp‑01 pattern) | Regularizers (§2e: SpecAugment + LayerDrop 0.1 + dropout); one cheap adapter‑only control run; pick on held‑out |
| CTC can't learn cased/punct well | Cased vocab from data; monitor CER; fall back to normalized + a truecasing step only if needed |
| Shared multilingual head weaker than per‑language | Justified by no‑lang‑id constraint; α‑sampling balances; ablate a per‑language‑then‑merge variant if needed |
| KenLM tooling / build friction | `pyctcdecode`+`kenlm` are standard; greedy is the always‑works fallback |
| Lingala stays hard | Expected — even MMS‑300M FT ≈ 42% WER; focus wins on CER + LM word boundaries; ensemble as the finisher |
| 1B FT memory on 40 GB | 8‑bit AdamW + grad checkpointing + duration bucketing + batch 4 × accum |

## 6. Compute & tooling
- **Hardware:** single A100‑40GB (vast.ai), same layout as exp‑01 (`data/` + scripts siblings; `prepared/`, `artifacts/`).
- **Libs:** `transformers` (`Wav2Vec2ForCTC`, MMS classes), `datasets`, `torchaudio` (forced alignment), `bitsandbytes` (8‑bit optim), `pyctcdecode` + `kenlm` (LM decoding), `jiwer`.
- **Reuse:** exp‑01's `common.py` helpers (per‑language metrics, normalization, logging, zip‑before‑destroy), adapting the collator/model‑loading for CTC.

## 7. Expected outcome
- Bigger **CER** gains (MMS leads CER on Bantu) → directly lifts the metric.
- **Luganda/Shona** should improve clearly (paper: MMS‑300M FT Luganda 16.9/3.4, Shona 25.0/4.3 vs our Whisper 0.32/0.07 & 0.39/0.08 — even accounting for normalization differences, headroom exists).
- **Lingala:** WER near the field ceiling (~40%), but the **KenLM decoder is the first real lever** on its word‑boundary errors, and its CER should drop.
- **No inference chunking, no hallucination loops** — cleaner, faster, more robust for Phase 2.

> **Bottom line:** experiment‑01 proved the pipeline and found the wall; experiment‑02 changes the architecture to the one the WAXAL evidence says is right for Bantu, and adds the two levers Whisper structurally lacked — **native long‑audio** and an **explicit word‑boundary language model**.
