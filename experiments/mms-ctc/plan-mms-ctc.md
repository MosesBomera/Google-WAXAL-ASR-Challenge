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

### 2c. Data‑cleaning pipeline — **thresholds set by our own EDA, not imported**

> Fully specified by [`report-mms-ctc-eda.md`](report-mms-ctc-eda.md). We downloaded all 38,199 clips raw and measured before filtering — which mattered: the paper's headline `wps>4` lever does **not** transfer (it fires on 6.4%/2.2%/**0%** of lin/sna/lug, and their 113.5→49.0 result was zero‑shot Whisper‑Tiny, a different regime).

Run once (offline, from the local metadata CSV — no GPU), producing a frozen manifest reused by all runs:

1. **Drop the corpus corruption** — ~1,400 clips whose audio is truncated to exactly **1.008/1.045 s** while keeping a full 12–44 word transcript (11–44 words/sec = physically impossible). **Verified corpus‑side**: the Hub itself serves 48,384 samples @ 48 kHz with a matching transcript, so our decode is faithful. Catch with **`duration < 1.5 s`** (belt‑and‑braces: **`wps > 4`** catches the same clips). Cost: lin **6.43%** / sna 2.24% / lug 0% of *clips* but only **0.35%/0.11%/0% of audio hours**.
   - It is **speaker‑scoped**, not random: 15 lin / 9 sna speakers, most **100% corrupt** (one lin speaker = 796/796 = 76% of all lin corruption). Cleaning removes ~5 lin speakers entirely (96→~91) — but they were never learnable.
   - **experiment‑01 trained on every one of these** → a plausible cause of its Lingala wall and Whisper's repetition/hallucination.
2. **Drop empty transcripts** — exactly 1 clip (lin, 17.7 s, 0 words).
3. **No upper duration drop** — see §2d.
4. **Keep low‑words/sec clips** — *not* a truncated‑reference artifact: the same pattern appears in `lug`, which has zero corruption. It's natural pausing in image‑prompted spontaneous speech; CTC handles silence with blanks. Flag only.
5. **Text normalization:** NFC, unescape CSV quotes, collapse whitespace — **keep case & punctuation** (unlike MMS's default lowercasing), plus a **character normalization map** for the 39 junk chars found (emoji `⭐`+`U+FE0F`, smart quotes `“ ”`, guillemets `« »`, `×`, Latin‑Extended singletons `ĝ þ œ Ķ ķ Ĺ Œ Ŋ ā ĺ`). **Keep** the legitimate French‑influenced diacritics (`é à ï ê ç ù û ü ò ó á ì ú`).
6. **Dedup:** negligible (12 texts over 24 clips) — no action needed.
7. **Corrupt clips' TEXT → the KenLM corpus (§2f)** — their transcripts are valid; only the audio is broken. Don't waste ~1,400 real sentences.
8. **Model‑agreement label cleaning (later, optional; MMS §3.1.6):** score train labels with a first model and drop the worst‑agreement tail.

Emit a **per‑language before/after report** (counts, hours, drop reasons). Net: keep **~99.6% of audio hours**, remove ~1,400 hallucination‑teaching examples.

### 2d. Long clips — **RESOLVED: there is no long‑audio problem**

The EDA settled the concern that partly motivated this pivot: **the longest clip in the entire dataset is 46.04 s**, nothing ≥ 60 s, and the long clips are *legitimate* (lug max 46.0 s @ 65 words, wps 1.41 — normal speech).

- **Keep 100% of clips.** With CTC's no‑cap and a bound of **~46 s** we drop *nothing*. `lin` (max 37.3 s) and `sna` (max 35.9 s) lose zero; only `lug` has anything >40 s (395 clips) — so an earlier 40 s bound would have needlessly discarded them.
- **No forced alignment, no segmentation, no VAD.** There is nothing long enough to need it. *(This deletes the old Stage‑4 "recover long clips" workstream entirely.)*
- **Inference:** CTC transcribes any length in one pass → **no chunking/stitching** (a clean win over Whisper's 30 s windowing, which had forced us to drop ~6% of training and chunk at test).
- **The one real consequence:** durations span **1.5–46 s**, so **length‑grouped batching** is required to keep memory even (46 s ⇒ ~2,300 wav2vec2 frames).

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

**We build our own SPEAKER‑DISJOINT validation split.** The EDA found the provided split is **~100% speaker‑leaked** (lin 99.95%, sna 99.77%, lug 99.25% of val clips have a speaker seen in training). So experiment‑01's validation was scoring **memorised voices**, while **Phase 2 is unseen speakers** — it never measured what the competition rewards.

- **Split rules:** clean **first**, then split **by `speaker_id`** (else we'd hold out speakers whose clips all vanish). No speaker above **5% of a language's clips** may sit in val (keeps Lingala's 1,737-clip speaker in train, where it's useful, and out of val, where it would dominate).
- **The real tension — train diversity vs. val stability.** Holding out clip-mass costs *voices*, and train speaker diversity is what drives generalisation. Measured at a fixed ~10% clip budget (`--val-speakers-per-lang`), Lingala's **train** speakers go **74 → 70 → 42 → 34** as we ask for 10/15/25/40 val voices. **Default 15** keeps **77% of Lingala's voices in train** while still holding out 15 per language. *(A naive "hold out many small speakers" rule left Lingala with 34 of 91 — it optimised val and gutted the thing that actually matters.)*

- **Gender is stratified too.** Picking speakers purely by size let gender land lopsided (lin val 80% male, sna 78% female) — with only ~15–25 val voices that happens by chance and biases the estimate toward one voice type. Each gender bucket is now filled **in proportion to that language's own clip-level mix**. Buckets below `--min-gender-share` (lug's `unknown`, 0.9%) are excluded from val and kept in train — a speaker-level pick can't hit a 6-clip budget without overshooting it 9×.
- **Where the constraints collide (Lingala only).** Its ~85 voices can't satisfy all three of {train diversity, gender balance, no val domination}. Measured: a 0.10 val‑share cap holds the top voice to 9.0% but skews val to **33F/67M** (population 54/46); 0.15 allows 14.8% and gets **59F/41M**. We take **0.15** — a *systematic* gender bias is worse than one noisy speaker. lug/sna don't bind (6.8%/9.3%).

**Realized split** (`clean_split.py`, seed 42 — frozen in `manifests/`, reproducible offline):

| Lang | train clips / speakers | val clips / speakers | val % | top val speaker | val gender vs population |
|---|---|---|---:|---:|---|
| lin | 13,667 / **60** (72.6 h) | 1,530 / 25 (8.3 h) | 10.1% | 14.8% | 59F/41M vs 54/46 |
| lug | 5,476 / **323** (37.9 h) | 643 / 16 (4.0 h) | 10.5% | 6.8% | 50F/50M vs 46/53 |
| sna | 13,902 / **138** (80.4 h) | 1,580 / 15 (8.9 h) | 10.2% | 9.3% | 55F/45M vs 57/43 |
| **total** | **33,045 / 520** (190.3 h) | **3,753 / 57** (21.2 h) | 10.2% | — | within ~4 pp everywhere |

Verified **0 speaker overlap**. Cleaning first shrank the pools as expected (lin 96→85, sna 160→153 speakers — the 100%-corrupt ones vanished entirely; lug unchanged at 339).
- **Expect the honest number to be worse than experiment‑01's** — that's the point. A speaker‑disjoint score that tracks Phase 2 beats a flattering one that doesn't.
- **Metrics:** per‑language **WER/CER, macro + micro**, on the full held‑out split (final) and a balanced subset (during training). CTC eval is *fast* — no autoregressive generation, a big iteration‑speed win over Whisper.
- **Cased vs normalized:** report both; one controlled leaderboard submission tells us Zindi's case/punct sensitivity. The cased vocab (§2b) means we're covered either way.
- **Public test set stays evaluation‑only** (never trained on). Note the organisers state the **Phase‑2 test set — which decides awards — is clean**, so any corruption in the *public* split is an artifact we can't act on; our public‑LB Lingala number may be **pessimistic** and we should not tune to it.

## 4. Staged plan

| Stage | What | Status |
|---|---|---|
| **1a** | **Raw download** — all 38,199 clips, unfiltered, incl. `speaker_id`/`gender`; emits the portable metadata CSV | ✅ **done** |
| **1b** | **EDA + verification** — distributions, threshold sweep, extremes; corruption verified corpus‑side; speaker metadata analysed | ✅ **done** → [`report-mms-ctc-eda.md`](report-mms-ctc-eda.md) |
| **1c** | **Freeze the training set** — cleaning (§2c), **speaker‑disjoint split** (§3), **cased CTC vocab + normalization map** (§2b), **KenLM corpus** (§2f). Ran **offline from the metadata CSV — no GPU** | ✅ **done** → `manifests/`, `vocab/`, `lm/`, `clean_split_report.json` |
| **2** | **MMS‑1B CTC fine‑tune** — full‑transformer FT (freeze CNN), regularized (§2e), length‑grouped batching, per‑language WER/CER checkpointing + one cheap adapter‑only control run. Then **checkpoint‑average the top 3** (gated: kept only if it beats the single best) and **push the model to the HF Hub**. Submit greedy‑decoded | 🔨 **built** (`src/train_ctc.py`, `average_checkpoints.py`, `push_model.py`; run book: `src/RUN.md`) — not yet run |
| **3** | **KenLM decoding** — pyctcdecode beam search; tune α/β/beam; measure the Lingala WER delta. Submit | |
| **4** | **Label cleaning** — model‑agreement filtering of noisy train labels; retrain. *(The old "recover long clips via forced alignment" is **deleted** — the EDA showed max clip = 46 s, nothing to recover.)* | |
| **5** | **Ensemble with whisper‑lora** — MMS (Bantu/CER, LM word boundaries) + Whisper‑large‑v3 (AR disambiguation); the family‑split finding says they're complementary | |

**Stage 1c is the bridge from analysis to training:** it turns the raw download + EDA findings into the exact, frozen, inspectable training set Stage 2 consumes — deterministic, cheap, reproducible across git commits, and buildable before we spin a box.

## 5. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Full‑transformer FT overfits (exp‑01 pattern) | Regularizers (§2e: SpecAugment + LayerDrop 0.1 + dropout); one cheap adapter‑only control run; pick on held‑out |
| CTC can't learn cased/punct well | Cased vocab from data; monitor CER; fall back to normalized + a truecasing step only if needed |
| Shared multilingual head weaker than per‑language | Justified by no‑lang‑id constraint; α‑sampling balances; ablate a per‑language‑then‑merge variant if needed |
| KenLM tooling / build friction | `pyctcdecode`+`kenlm` are standard; greedy is the always‑works fallback |
| Lingala stays hard | Expected — even MMS‑300M FT ≈ 42% WER; focus wins on CER + LM word boundaries; ensemble as the finisher |
| **Lingala is speaker‑poor → weak speaker generalisation** | Only **96 speakers** for 16,244 clips (top speaker = 10.7%), vs lug's 339 for 6,119 — and speaker diversity rank‑orders exp‑01 WER exactly. Phase 2 = unseen speakers, so this is a first‑class risk: rely on the speaker‑disjoint split to *see* it, plus speaker‑invariance augmentation (SpecAugment, LayerDrop, optional speed‑perturb) to mitigate it. |
| **Speaker‑disjoint val is a small‑sample estimate** | lin has ~91 speakers post‑cleaning, so a held‑out slice is high‑variance. Hold out many *small* speakers; report per‑language n; consider GroupKFold if a decision hinges on a small delta. |
| 1B FT memory on 40 GB | 8‑bit AdamW + grad checkpointing + **length‑grouped batching** (1.5–46 s span) + batch 4 × accum |

## 6. Compute, tooling & running it

- **Hardware:** single A100‑40GB (vast.ai). Versioned by **git commits** (no `vN` folders).
- **Libs:** `transformers` (`Wav2Vec2ForCTC`, MMS classes), `datasets`, `bitsandbytes` (8‑bit optim), `pyctcdecode` + `kenlm` (LM decoding), `jiwer`. *(`torchaudio` forced alignment is no longer needed — §2d.)*
- **Reuse:** `whisper-lora`'s `common.py` helpers (per‑language metrics, text normalization, logging, zip‑before‑destroy), adapted to a CTC collator/model.

### Layout
**Everything hangs off one folder and all paths auto‑resolve** — no flags needed. The code supports the two layouts we actually use, via `common.workdir()`:

```
# vast.ai (how we run) — scripts FLAT in the folder:
folder/
  data/          Train.csv, Test.csv, SampleSubmission.csv, waxal_clip_metadata.csv
  *.py           common.py prepare_data.py eda.py verify_data.py clean_split.py train_ctc.py
  prepared/      <- 1a writes raw 16 kHz Arrow chunks
  manifests/     <- 1c writes the frozen speaker-disjoint split
  vocab/  lm/    <- 1c writes the cased CTC vocab + KenLM corpus
  artifacts/     <- Stage 2 writes model + logs

# repo checkout — same tree, code under src/:
experiments/mms-ctc/{src/*.py, manifests/, vocab/, lm/, artifacts/}   (data/ at the repo root)
```
`data_dir()` finds `data/Train.csv` (flat) *or* `data/google-waxal-.../Train.csv` (repo) by walking up. Never use `SCRIPT_DIR.parent` in a script — on the flat layout it escapes to `/`; use `workdir()`.

### Commands
Run from inside the folder (`python <script>.py`); in the repo prefix with `src/`.
```bash
export HF_TOKEN=hf_...            # to stream google/WaxalNLP

# 1a — raw download (all clips, unfiltered, + speaker_id/gender + metadata CSV)
python prepare_data.py --max-examples-per-language 100   # smoke test first
python prepare_data.py                                   # full

# 1b — EDA + verification (metadata only; no audio decode; fast to re-run)
python eda.py
python verify_data.py                                    # corruption + speaker checks

# 1c — freeze the training set (OFFLINE from the metadata CSV — no GPU, no download)
python clean_split.py

# 2 — MMS-1B CTC fine-tune (reads manifests/ + vocab/ + prepared/)
python train_ctc.py --num-train-epochs 0.05 --eval-steps 10   # smoke test first
python train_ctc.py                                           # full FT — the primary
python train_ctc.py --train-mode adapter --artifact-dir artifacts/mms-1b-adapter   # control

# 2b — checkpoint-average + save off-box. BOTH must run BEFORE teardown.
python average_checkpoints.py                                 # top 3, gated on full val
python push_model.py --repo-id <user>/waxal-mms-1b-ctc        # ~3.9 GB, private
```
Step-by-step run book with checks, teardown checklist and gotchas: **[`src/RUN.md`](src/RUN.md)**.

**Two things that cannot be done retroactively** (the box takes them with it):
- **Checkpoint averaging** needs the `checkpoint-*/` dirs — hence `--save-total-limit 3` (~6 GB each: 3.9 GB fp32 weights + ~2 GB 8-bit optimizer state, on top of ~49 GB of prepared audio). It's **gated**: we score the average *and* the single best on the full speaker-disjoint val and keep the average only if it wins — averaging usually helps (flatter minimum → better unseen-speaker generalisation, free at inference) but not always, and shipping a worse model because the technique is fashionable would be silly.
- **Storing the model.** A full-FT MMS-1B is ~3.9 GB — push it to the **HF Hub (private)** rather than downloading through the browser; a fresh box pulls it back with `from_pretrained("<user>/<repo>")`. (Kaggle Datasets works too but needs separate credentials and isn't model-shaped.) Checkpoint dirs are excluded from the upload — they're only useful on-box.

**Realized Stage-2 config** (`src/train_ctc.py` defaults, all overridable):

| | value | why |
|---|---|---|
| scope | full transformer FT, **CNN feature extractor frozen** | canonical wav2vec2 recipe; our data is medium‑resource (§2a) |
| head | fresh **cased** CTC head, vocab **103** | MMS lowercases by default; half our metric is CER on cased text |
| lr | body **3e‑5** / head **1e‑4** (two param groups) | pretrained layers move slowly, the fresh head doesn't |
| optim | **8‑bit AdamW** + grad checkpointing + bf16 | fits a 1B full FT in 40 GB |
| batch | 4 × accum 4 = **16**, `group_by_length` on `input_length` | clips span 1.5–46 s |
| regularization | SpecAugment `mask_time 0.065`/`mask_feature 0.10`, LayerDrop 0.1, dropout 0.05 | counters the whisper‑lora overfit |
| epochs | 8, early stopping (patience 4) on `combined` | CTC needs more epochs than seq2seq LoRA |
| selection | per‑language WER/CER → `combined` (macro) | never loss |

### Outputs
| Stage | Artifact | Why it matters |
|---|---|---|
| 1a | `prepared/waxal-raw/{train,validation}/<lang>/` | Raw Arrow chunks (`id, audio, transcription` [cased], `language, duration, speaker_id, gender`). Nothing dropped. |
| 1a | **`data/waxal_clip_metadata.csv`** (~4 MB, 38,199 rows, 595 speakers) | **The one file to keep.** Every cleaning/split decision is answerable from it offline — no GPU, no re‑download. |
| 1b | `eda_logs/{eda_report,verify_corruption,verify_metadata,speaker_summary}.json` | Distributions, threshold sweep, the corruption verdict, speaker leakage. |
| 1c | `manifests/{train,val}.csv` + `vocab/vocab.json` + `lm/corpus.txt` + `clean_split_report.json` | The frozen, inspectable training set Stage 2 consumes. Small enough to commit — the split is reproducible from git. |
| 2 | `artifacts/mms-1b-all/` — best model + processor + `logs/{train_ctc.log, metrics.jsonl, run_config.json, trainer_log_history.json, final_eval_full_val.json}` | The honest speaker‑disjoint number lands in `final_eval_full_val.json`. **Full‑FT checkpoints are GBs** (unlike whisper‑lora's ~200 MB adapters) — `save_total_limit=2`; grab the logs + the final model before teardown. |

**Why raw‑download‑then‑analyse:** the download is the expensive step; cleaning decisions are cheap. Materialise once → EDA → *then* pick thresholds, so changing one costs seconds instead of a re‑download, and we see exactly what a filter removes before committing. It already paid off — the paper's `wps>4` lever fires on **0%** of Luganda, so importing thresholds on faith would have been a mistake.

## 7. Expected outcome
- Bigger **CER** gains (MMS leads CER on Bantu) → directly lifts the metric.
- **Luganda/Shona** should improve clearly (paper: MMS‑300M FT Luganda 16.9/3.4, Shona 25.0/4.3 vs our Whisper 0.32/0.07 & 0.39/0.08 — even accounting for normalization differences, headroom exists).
- **Lingala:** WER near the field ceiling (~40%), but the **KenLM decoder is the first real lever** on its word‑boundary errors, and its CER should drop.
- **No inference chunking, no hallucination loops** — cleaner, faster, more robust for Phase 2.

> **Bottom line:** experiment‑01 proved the pipeline and found the wall; experiment‑02 changes the architecture to the one the WAXAL evidence says is right for Bantu, and adds the two levers Whisper structurally lacked — **native long‑audio** and an **explicit word‑boundary language model**.
