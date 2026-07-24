# plan — qwen-asr (adapt Qwen3-ASR-1.7B to WAXAL lin/sna/lug)

**Goal.** Fine-tune **Qwen3-ASR-1.7B** (Alibaba, generative audio-LLM ASR) on WAXAL and see whether
its built-in LLM decoder can beat our best CTC model — **MMS-1b-v2, speaker-disjoint val combined
0.2033 (WER 0.323 / CER 0.084), board 0.7937** — *or* serve as a decorrelated ensemble partner.

**Why this is worth a shot (the dark-horse thesis).** Two reasons, both "maybe." **(1) Recency/capability:**
Qwen3-ASR-1.7B is a **much newer, SOTA-class** model than MMS-1b — it's plausible its stronger acoustic
encoder + LLM decoder transfer further on low-resource speech even without in-language pretraining. **(2) The
right tool for our specific wall:** our whole MMS line is stuck on a WER-dominated error (0.32 WER vs 0.08
CER, ~4×) where the acoustics are already decent and **KenLM/neural-LM rescoring were dead** (the CTC model is
too confident to rescore). That residual is word/morphology-level — exactly what a **generative LLM decoder**
is built to fix and what a bolt-on n-gram LM cannot. Qwen3-ASR *is* that decoder, natively fused to the
acoustic model. This is a **low-cost probe of a dark horse**, not a conviction bet — hence the gate in Lever 0.

**Why it might fail (the headline risk — read first).** Qwen3-ASR-1.7B officially supports **52
languages/dialects — none of them lin/sna/lug** (no Bantu at all; not even Swahili). The Children's-ASR
winners we borrow the recipe from were doing **domain** adaptation (children) inside a **supported**
language (English). We are asking Qwen to learn **new languages from ~200 h**, which is a strictly harder
ask. MMS-1b-all, by contrast, has *explicit pretraining coverage* of all three — its structural home turf.
So this is an **unconventional, higher-variance bet**, and the plan is built around a **cheap GO/NO-GO
gate** (Lever 0) so we spend real compute only after evidence it can clear the language barrier.

---

## 0. Current status (2026-07-23) — LEVER 0 PASSED: **strong GO**

Both pre-gate checks green **and the Lever-0 gate ran on an A100** → **strong GO**: best combined
**0.2242** (WER 0.348 / CER 0.101), within **0.021** of MMS's 0.2033, on **one epoch, WAXAL-only,
no levers pulled**. Qwen learns all three languages; Lingala (the headline risk) fell 0.66→0.34. What exists:

| Artifact | State |
|---|---|
| [`src/common.py`](src/common.py) + [`src/prepare_data.py`](src/prepare_data.py) | Ported **verbatim** from `../mms-ctc-v2/src/`. `prepare_data.py` ran on the box → `prepared/waxal-raw/`. |
| [`src/tok_check.py`](src/tok_check.py) | Built + run (`fourforfour` env). Tokenizer gate → **PASS (GO, keep raw)**, see §4. |
| Val comparability | **Verified** — all 3,753 frozen-val ids resolve (§1); gate loaded 3,753 val / 33,037 train. |
| [`src/train_qwen.py`](src/train_qwen.py) + [`src/RUN.md`](src/RUN.md) + [`requirements.txt`](requirements.txt) | **Built + RUN.** LoRA SFT, 3rd-place scaffold + our LoRA r16, generation-eval on `combined`. Smoke passed, gate passed. Debugged live on the box — see the lessons box below §3. |
| **Lever-0 gate result** | **STRONG GO** — trajectory + why in §3 (Lever 0). Best = `artifacts/qwen-gate/checkpoint-2065`. |
| [`src/push_model.py`](src/push_model.py) | Built. Pushes a checkpoint to HF (adapter+configs; resume-state excluded by default). |
| [`src/infer_qwen.py`](src/infer_qwen.py) | Built. Base+adapter val eval under **matched training conditions** (imports `train_qwen` helpers) + test-submission stub. |
| ✅ **Checkpoint-save bug — FIXED** | `train_audio_encoder=true` params weren't serialized → gate adapter was LoRA-only (measured: **0.5527 vs 0.2242**, encoder worth Δ0.33). Fixed in `train_qwen.py` (`trained_base.safetensors` + hard save guard) & `infer_qwen.py` (restore + mismatch guard). Validate via `--smoke` on the next box. See lessons box §3. |

**Next on the box:** (1) `infer_qwen.py` to measure the LoRA-only val (how much the lost encoder cost);
(2) **fix the encoder-save** in `train_qwen.py` (DONE); (3) full **5-epoch** recipe + Lever 2 (`score_filter.py`)
+ soup (`soup.py`); (4) the MMS⊕Qwen ensemble (Lever 4).

---

## 1. What we CARRY from the mms-ctc line (hard-won, do not re-derive)

| Lesson | How it applies here |
|---|---|
| **The speaker-disjoint val is the yardstick AND a faithful board proxy** — v2 proved val Δ = board Δ (−0.0035 ≈ −0.0037). | Reuse the **byte-identical** fixed val: [`../mms-ctc/manifests/val.csv`](../mms-ctc/manifests/val.csv) (3,753 clips). **Verified: all 3,753 ids resolve** against `data/…/Train.csv` by `id` (the manifest is metadata-only; transcripts join from Train.csv column `transcription`). Every Qwen run judged on this file → directly comparable to MMS v2's 0.2033, predicts the board without spending submissions. **Do NOT re-split.** |
| **Board-faithful metric**: micro-pooled, **WER lowercased+punct**, **CER cased+punct**, combined = 0.5·WER+0.5·CER. | `per_language_metrics` is already ported (in `src/common.py`); select checkpoints on `combined`, never loss (the reference winners selected on loss — we do better). |
| **Cased + punctuation targets** protect the CER half. | Train Qwen to emit cased+punct text (it's generative, so this is just the target string). |
| **Data cleaning**: drop <1.5 s, wps>4, empty (the ~1.008 s corruption fingerprint). | Same filters; reuse `clean_verdict` (in `src/common.py`) / the frozen split. |
| **Lingala is the hard, board-dominant language** (v2 lin val 0.44 vs lug 0.18 / sna 0.27); read-speech external data helped lug/sna, **not lin**. | Watch **per-language** val. Don't assume external data rescues lin. |
| **Generative decoders hallucinate on corruption**; the board has ~1 s truncated clips. | Cap `max_new_tokens`; reuse `infer.py`'s empty→placeholder submission handling (never the reference text). |

## 2. Reference solutions — provenance & why the **3rd-place scaffold**

The reference is `data/on-top-of-pasketti-main/word-track/{1st,2nd,3rd Place}` — the DrivenData *Children's
Speech Recognition Challenge, Word Track* winners. **Same task type (ASR), different everything else**: their
data is **children's speech in English** (a Qwen-*supported* language); they did **domain** adaptation, we do
**new-language** adaptation. We borrow **methodology only**, never any assumption that their data/results transfer.

**What each solution is:**
- **1st place** — a **two-model pipeline**: fine-tune **Parakeet-CTC-1.1b (NVIDIA NeMo)**, use it to score &
  quality-filter the data, *then* fine-tune Qwen3-ASR (configs `qwen_exp022/023/025/026`) and **model-soup** them.
  Ships as a heavy multi-package `uv` workspace (`csrc` + `csrc_prep` + `csrc_parakeet` + `csrc_qwen`), needs a
  pinned-commit NeMo clone + flash-attn. Its **LoRA config values live in `csrc_qwen/configs/qwen_exp022.yaml`**.
- **2nd place** — a fuller `qwen_asr` package + `csrw/` trainer; **souped 18 models**; `-use_lora -lora_rank 32`.
  Reference for the **soup at scale** (Lever 4).
- **3rd place** — a single self-contained **326-line file** `qwen3_asr_sft.py`: `patch_outer_forward`
  (patches `.thinker.forward`), `DataCollatorForQwen3ASRFinetuning` (the audio collator),
  `make_preprocess_fn_prefix_only` / `build_prefix_messages` (**prompt-prefix conditioning = our Lever 1**), and
  `MakeEveryCheckpointInferableCallback` (every checkpoint is vLLM-inferable). It is a **full-FT** scaffold —
  **no LoRA of its own** (default lr 2e-5, batch 32, 1 ep).

**Why we base the trainer on 3rd place and not 1st:** we *do* use 1st place — for the parts that transfer as
**ideas**: (a) its exact **LoRA config values** (from `qwen_exp022.yaml`) and (b) its two winning **techniques**,
quality-stratified data filtering (Lever 2) and model soup (Lever 4). What we don't adopt is its **code**, for
three concrete reasons:
1. **1st place's system is Parakeet-CTC-centric.** Its winning half is a *second CTC model* — the exact
   architecture MMS already is. Rebuilding it gives us nothing new and contradicts the whole thesis (test the
   *generative* decoder as the escape from the CTC WER wall). Parakeet's role as the data-filter scorer we cover
   **for free with MMS v2** (Lever 2).
2. **1st place's Qwen code is a heavy multi-package build** (NeMo pin, flash-attn, config-driven manifests) —
   high port cost, much of it children's-English-specific.
3. **3rd place gives the same Qwen-native plumbing in one file** — and the plumbing (patching the outer forward,
   the audio data-collator, checkpoint-inferability, prompt-prefix) is the *hard, error-prone* part. We bolt our
   LoRA on with a ~15-line PEFT wrap. Minimal surface, fastest clean adaptation, closest to our own single-file
   mms-ctc-v2 style. (2nd place stays on hand as the reference for souping many models in Lever 4.)

### The winning recipe we ADOPT (values from 1st place `qwen_exp022.yaml`, scaffold from 3rd place)
1st = LoRA + quality-stratified data + model soup, WER 0.235→0.188. Concrete, transferable settings:
- **Base:** `Qwen/Qwen3-ASR-1.7B`. Fine-tune the **thinker** (`.thinker.forward`); patch the outer forward (3rd-place `patch_outer_forward`).
- **LoRA (our add-on to the full-FT scaffold):** `r=16, alpha=32, dropout=0.0`, target `[q,k,v,o,gate,up,down]_proj`.
  **`train_audio_encoder=true`, `train_projector=false`** — training the audio encoder matters *more* for us (new phonetics, not just a new domain).
- **Optim/schedule:** lr **1e-4**, wd 0.01, AdamW, **linear**, **warmup 0.1**, **3 epochs** in the reference —
  **we use 5** (gate used 1), **bf16**,
  grad-checkpointing, eff. batch **16** (per-device 4 × accum 4). `max_duration_sec` 30 in the reference —
  **we use 45** (WAXAL-tuned, §3 lessons #3).
- **Light aug:** SpecAugment (freq 1×15, time 1×0.05, p≈0.5); **no speed-perturb** (they disabled it).
- **Quality-stratified data (the killer technique):** 1st place scored every train clip with a *reference model*
  (Parakeet) and dropped/kept by predicted WER. **We already own a strong reference model → use v2 MMS** to score
  WAXAL train per-clip and prune the worst (mislabeled/corrupt) — see Lever 2.
- **Ensembling:** checkpoint-averaging + **model soup** (average several LoRA runs at different data thresholds/seeds).
  2nd place souped 18 models; even 3–4 gives a reliable bump.
- **Inference:** vLLM, batched, `max_new_tokens≈300` in the reference — **we use 384** (§3 lessons #3), a `language`/system prompt.

## 3. The levers we PULL (each measured on the fixed val, board-faithful, per-language)

### Lever 0 — GO/NO-GO gate *(do this FIRST; it decides everything)*
Cheapest possible signal that Qwen can learn these languages at all: **one LoRA run, ~1 epoch**, WAXAL-only,
default recipe, eval on the fixed val. **No quality-filtering yet** (that's Lever 2) — this is the clean baseline.
- **GO** if val `combined` lands in a sane, MMS-competitive range (roughly **< 0.30 and trending toward
  MMS's 0.20**) with coherent per-language output (not hallucinating). → proceed to Levers 1–4.
- **NO-GO** if it's wildly worse (e.g. **> 0.5**, byte-fallback garbage, or Lingala hallucination) → the new-language
  barrier is too high for the data budget. **Stop, write the negative result, keep MMS v2.** (A negative here is
  cheap and genuinely informative — it tells us the ceiling isn't decoder quality but language priors.)

**RESULT (2026-07-23): STRONG GO.** 1-epoch LoRA, WAXAL-only, default recipe, A100 (~3.3 h). Gen-eval
(balanced 600-clip subset, board-faithful `combined`):

| step | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | **2065** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| combined | .471 | .384 | .319 | .298 | .249 | .241 | .231 | .227 | .232 | .230 | **.2242** |

**Best = ckpt-2065: combined 0.2242** (WER 0.348, CER 0.101 | lin 0.345, sna 0.194, lug 0.157). Why we
call it GO: (1) far below the NO-GO line and **within 0.021 of MMS's 0.2033** on one epoch with *none* of
the levers pulled; (2) **coherent, not hallucinating** — CER 0.10 means characters are largely right;
(3) **Lingala, the headline risk, resolved itself** (0.66→0.34); (4) same WER≫CER profile as MMS (the wall
we came to test). Note: 0.2242 is the *balanced 600-subset*; the full-3,753 head-to-head vs MMS is
`infer_qwen.py`'s job. Training was healthy (loss 47→~16 then flat while `combined` kept dropping —
**vindicates selecting on generation-combined, not loss**). Logs of record:
[`training-artifacts/qwen-gate/`](training-artifacts/qwen-gate/) (`gen_eval.jsonl`, `run_config.json`, `train_qwen.log`).

**Per-language insight (from `gen_eval.jsonl`): Lingala peaked mid-epoch, then regressed.** lin CER
bottomed ~step 1000–1400 (cer_lin ≈0.186) and drifted back up to 0.215 by the end, while lug/sna kept
improving — so the **overall-best ckpt-2065 is NOT Lingala's best** (lin was better at ~step 1000). Reads as
mild per-language interference in the back half (the model trading lin for lug/sna). Implication for the full
recipe: this is textbook motivation for **checkpoint-averaging / the soup (Lever 4)** and possibly per-language
checkpoint selection — a single step can't be optimal for all three at once. Watch lin specifically when Lever 3
(external lin data) lands.

### ⚠ Lessons & REQUIRED fixes from the box run (READ before the full recipe)
**#1 — the one we cannot ship: the trained audio encoder was NOT saved.** `apply_lora` sets the audio
encoder's `requires_grad=True` *without* PEFT's `modules_to_save`, so the encoder trains in memory (drove
0.2242) but the checkpoint serializes **only the LoRA adapter** — confirmed: gate `adapter_model.safetensors`
= **79 MB** (LoRA-only; +encoder would be ~675 MB). The run has exited, so **that encoder state is lost**;
the gate's 0.2242 is not reconstructable from the saved adapter (this is exactly what `infer_qwen.py` is
measuring — the LoRA-only floor).
> **✅ FIXED (2026-07-23) — explicit-save approach** (chosen over PEFT `modules_to_save` to avoid a
> LoRA-target collision inside the encoder that I couldn't validate without box time):
> - `train_qwen.py` → `make_save_trained_base_callback`: on every save, writes the non-LoRA trainable params
>   (exactly what `train_audio_encoder`/`train_projector` unfroze — the ~320M encoder) to
>   `<ckpt>/trained_base.safetensors`, and **HARD-FAILS if <50M params are captured** while the encoder was
>   meant to train → a silent LoRA-only checkpoint can no longer happen (the smoke's step-5 save trips it in seconds).
> - `infer_qwen.py` → `_restore_trained_base`: loads it back over the base (local dir OR HF repo) and
>   **aborts on any key mismatch**, so we never silently eval with a base encoder.
> - `push_model.py` already includes `trained_base.safetensors` (it's not resume-state).
> Confirmed by the LoRA-only measurement: base-encoder + adapter = **0.5527** vs the gate's 0.2242, so the
> encoder is worth Δ0.33 — this fix is load-bearing, not optional. **Validate on the next box via `--smoke`
> before the 5-epoch spend** (the save log should print `trained_base.safetensors: … ~320M params … ~0.6 GB`).

**#2 — fixes already applied to `train_qwen.py` (keep them; they're load-bearing):** collator **right-pad**
(label masking) + gen-eval **left-pad** (batched generation); **bf16 cast** of float inputs before `generate`
(audio conv is bf16, autocast is off outside the Trainer); `generate` returns a ModelOutput → **unwrap
`.sequences`** then slice off the prompt; `enable_input_grads()` **manual hook** (the wrapper doesn't expose
`get_input_embeddings`); grad-checkpointing **`use_reentrant=False`** (PEFT needs it); `processing_class=`
(not deprecated `tokenizer=`); manifest-dir finder resolving both flat-box and repo layouts.

**#3 — data knobs, data-driven:** `--max-duration-sec 45` (Qwen native ~30 s, windowed attn ~45 s; keeps val
byte-complete, drops 8 train clips) and `--max-new-tokens 384` (longest WAXAL target ~247 tokens; fences
hallucination). SpecAugment: deferred from the 1-epoch gate; **now ON by default for the 5-epoch runs** (`--no-spec-augment` to ablate).

### Lever 1 — Prompt / language conditioning
Qwen has **no lin/sna/lug** `language` option. Test which conditioning the fine-tune learns best (via 3rd-place's
`build_prefix_messages` prompt-prefix): (a) neutral `"Transcribe the audio."`; (b) language-named
`"Transcribe the Lingala speech."` (per-clip, using the known WAXAL language); (c) closest-supported-language tag.
Cheap A/B on the gate run.

### Lever 2 — Quality-stratified training data via **v2 MMS as the scorer** *(3-source-confirmed; DO IT)*
Drop mislabeled/corrupt train clips that teach a generative decoder to hallucinate. It's the 1st-place
Parakeet-filtering trick, but **free** (we own a strong in-domain scorer). **Where it comes in — a DATA step,
not a trainer change:** `score_filter.py` emits a filtered train manifest; `train_qwen.py` consumes it via
`--train-manifest` (**val is NEVER filtered**). Pipeline:
```
prepare_data → manifests/train.csv (full 33k)
   └─ score_filter.py --scorer v2-mms   [ONE-TIME, ~30–60 min: MMS-v2 transcribes every train clip →
        per-clip WER → manifests/train_scored.csv + views train_wer0.9.csv / train_wer0.7.csv]
   └─ train_qwen.py --train-manifest train_wer0.9    (score once; thresholds are cheap filters)
```
- **Default threshold: drop WER>0.9 (~3–5%).** Convergent across THREE sources: 1st-place `exp023` (0.9),
  Kinyarwanda hackathon (drop WER>0.9 → removed ~4k/125k ≈ 3%), and the general trick. Optional per-word-count
  buckets + a `CER<0.5` "or" mode (1st-place `exp026`) as refinements.
- **Run 1 stays UNFILTERED** — it's the baseline that *measures* the filter's gain, validates the encoder-save
  fix with fewest moving parts, and is itself a soup member (1st place's unfiltered `exp025` was one). Filtering
  enters at runs 2+.
- **Lever 2 IS the soup axis:** 1st place's 4 soup members differ ONLY by this threshold (0.7 / 0.9 / none /
  WER-or-CER). The sweep *generates* Lever 4's members — not a separate step. `score_filter.py` reuses MMS-v2 + `common.py`.

### Lever 3 — External data (optional, staged) — **DEMOTED to low priority**
FLEURS lin/sna/lug + LRSC lin are **verified WAXAL-safe** and already prepared for v2. **But two convergent
signals say external read-speech often doesn't transfer:** (a) our MMS line — external helped lug/sna, **not
lin**; (b) Kinyarwanda hackathon trained on **1M Common Voice examples → no improvement** (out-of-domain: read
speech vs elicited/reaction speech). WAXAL is elicited; FLEURS/LRSC are read → likely out-of-domain. So add
**only after** Levers 1–2, per-language val-gated, and if used, prefer the **curriculum shape** (pretrain on
external → fine-tune on WAXAL, à la warm-start) over mixing. **Note:** LRSC/FLEURS are the *only* source of the
Lingala open vowels `ɔ`/`ɛ` — WAXAL has none (§4); re-run `tok_check.py --external` before adding.

### Lever 4 — Ensembling (the closer)
Soup 3–4 LoRA runs (different data thresholds/seeds), + checkpoint-averaging within each (2nd place is the
reference for souping at scale — 18 models). Then the **cross-architecture prize**: **MMS-v2 (CTC) ⊕ Qwen
(seq2seq)** — maximally decorrelated errors. Even if Qwen only *ties* MMS solo, the ensemble is where a WER-only
competition is won.

### Cross-task evidence & top-3 audit — what we ADOPT / SKIP (2026-07-23)
Audited all three Children's-ASR winners + the Sunbird **Kinyarwanda** hackathon (a Bantu task). Verdicts:

| Technique | Who | Verdict for us |
|---|---|---|
| Quality/mislabel filtering | 1st + Kinyarwanda | **ADOPT** = Lever 2 (WER>0.9), now 3-source-confirmed |
| SpecAugment (freq 1×15, time 1×0.05, p0.5) | all top-3 | **ADOPT** in run 1 (recipe default) |
| Beam search decoding (n_beam 2–5) | 2nd | **ADOPT** — cheap inference-time WER win (we use greedy) |
| Repetition/hallucination post-proc (`detect_and_fix_repetitions`) | 1st | **ADOPT** — cheap, hits our WER wall + corruption |
| Checkpoint-averaging within a run | 1st | **ADOPT** in Lever 4 (near-free) |
| **Noise augmentation** | 1st (in-domain), 2nd, Kinyarwanda (white noise) | **LOW PRIORITY / val-gated A/B.** Honest read: Kinyarwanda used *white* noise **because their test was clean** — it's generic regularization, **redundant with SpecAugment**. In-domain noise needs WAXAL noise we lack. Not a run-1 default. |
| Speed perturbation | Kinyarwanda ✓ / 1st ✗ (`speed_perturb: null`) | **SKIP** for Qwen — the same-model (1st) evidence disabled it. |
| Lowercase/unpunct targets | Kinyarwanda | **DO NOT ADOPT — landmine.** Their metric was case/punct-*insensitive*; ours scores **CER on CASED text**, so cased+punct is required (protects half the score). |
| External data | 3rd (TalkBank ✓ in-domain-ish) / Kinyarwanda (1M CV ✗) | **DEMOTED** = Lever 3 (see above). |
| Bigger base (7B/Omni/Kimi) | 3rd tried → **no gain** | **SKIP** — 1.7B is the horse. Useful negative. |
| Selection metric | all top-3 selected on **loss** | **WE'RE AHEAD** — we select on generation-`combined` (gate proved loss went flat while combined dropped). |

### Full-recipe run plan (post encoder-fix)
1. **Re-prep** (`prepare_data.py`, no GPU) → **`--smoke`** (validates the encoder-save fix: expect
   `trained_base.safetensors … ~320M … ~0.6 GB`, and `infer_qwen` prints `restored trained audio encoder`).
2. **Run 1 (baseline) — THE experiment we run now:** 5 epochs, encoder-saved, **SpecAugment on**, named prompt, 45 s / 384 tok, **unfiltered**
   WAXAL-only. → the honest baseline + fix-validation + soup seed #1.
3. In parallel: `score_filter.py` (MMS-v2 scoring) → `train_wer0.9.csv` / `train_wer0.7.csv`.
4. **Runs 2–N:** `--train-manifest train_wer0.9` (and 0.7) → Lever-2 soup members.
5. **Soup** (+ checkpoint-averaging) → **infer_qwen** full-val vs MMS 0.2033 → **MMS⊕Qwen** ensemble.
6. **Adding epochs if 5 aren't enough:** don't raise `--epochs` on a finished run — **warm-start** a short
   second run at lower LR (`--init-from <ckpt> --epochs 2 --lr 5e-5`). Crash mid-run → `--resume-from-checkpoint auto`
   (same schedule; both paths reload the trained encoder). See §5.

## 4. Tokenizer / text gate — RESULT: **PASS (GO, keep raw)**
The concern: does Qwen's tokenizer round-trip WAXAL targets (cased+punct) without an `[UNK]`/byte-fallback
explosion that would rot CER? Verified offline (no GPU) by `src/tok_check.py` over all 38,198 WAXAL transcripts,
normalized exactly as the model must emit (`normalize_for_ctc`, cased+punct):

- **0 / 38,186 distinct targets fail** the `decode(encode(t)) == t` round-trip (0.00%). No `[UNK]`, no `<0xNN>`
  byte fallback anywhere. `tokens/char ≈ 0.38` (~2.6 chars/token — healthy BPE, no fragmentation blow-up).
  Qwen tokenizer is byte-level BPE (`vocab 151,643`, `unk=None`).
- WAXAL's non-ASCII inventory is **22 chars**: the Bantu eng nasals **`ŋ` (466×)** and **`ᵑ` (60×)** (3 tokens
  each, but round-trip clean) + 20 French-Lingala diacritics (é è î í ô à ñ â ú ê ï ç É ó á ì ù ò û ü — all fine).
- **KEY FINDING:** WAXAL's own transcripts contain **zero `ɔ`/`ɛ`/greek-`ε`**. The plan's original open-vowel fear
  is a **Lever-3 (LRSC/FLEURS) concern only**, not Lever 0/1.

**Decision: keep raw.** Feed cased+punct as-is; **no `CHAR_MAP` change for the gate.** Only if the `--external`
re-run (Lever 3) shows byte-fallback rot on `ɔ`/`ɛ` do we reuse v2's `CHAR_MAP` (which already has `ε→ɛ` + the
junk-char map).

## 5. Rebuild pipeline (mirrors the mms-ctc-v2 flow; new code in `src/`)
```bash
# --- pre-gate checks (CPU-only, DONE) ---
python src/tok_check.py          # §4 gate: tokenizer round-trips WAXAL text?  → PASS (GO, keep raw)

# --- data prep (on the box): audio Arrow chunks; reuse the mms line's if already present ---
#     The FROZEN val (../mms-ctc/manifests/val.csv) is reused as-is — do NOT re-split (comparability).
python src/prepare_data.py       # streams google/WaxalNLP → prepared/waxal-raw/{train,validation}/<lang>/
                                 #   (mms-ctc-v2 downloader, unchanged; or point --audio-dir at existing chunks)

# --- Lever 0: the GO/NO-GO gate (A100 40GB, ~3.3 h) — DONE → STRONG GO (best 0.2242 @ ckpt-2065) ---
python src/train_qwen.py --gate               # 1-epoch LoRA, WAXAL-only. → GO (see §3).
python src/push_model.py --repo-id <you>/waxal-qwen3-asr-gate   # preserve the adapter (disk pressure)
python src/infer_qwen.py --adapter <ckpt> --eval-set val --val-subset-per-lang 200  # measure LoRA-only floor
# ===== encoder-save FIX done → smoke it, then run the full recipe =====

# --- THE experiment: run 1 (full 5-epoch baseline, unfiltered) ---
python src/train_qwen.py --output-dir artifacts/run1-pure   # 5ep default, LoRA r16, encoder SAVED, SpecAug ON
python src/infer_qwen.py --adapter <ckpt> --eval-set val    # full-3753 board-faithful val vs MMS v2 0.2033

# --- Levers 2 & 4 (after run 1) ---
python src/score_filter.py --scorer v2-mms                  # LEVER 2: per-clip WER over train → train_wer0.9.csv
python src/train_qwen.py --train-manifest train_wer0.9 -o artifacts/run2   # filtered soup members
python src/soup.py <run1> <run2> <run3> -o soup/            # LEVER 4: model soup
python src/infer_qwen.py --adapter soup/ --eval-set test    # board proxy + submission (needs prepared test chunks)
```
**Built + run/validated:** `common.py`, `prepare_data.py`, `tok_check.py` (gate passed), `train_qwen.py`
(Lever-0 gate → GO; **+ encoder-save fix, resume/warm-start, SpecAugment, length-grouping** — post-gate, box-smoke pending),
`infer_qwen.py` (+ beam + repetition-fix + encoder-restore), `push_model.py`, `RUN.md`, `requirements.txt`.
**Still to build:** `score_filter.py` (Lever 2); `soup.py`/`average_adapters.py` (Lever 4); `infer_qwen.py`'s test path
(needs prepared board-test chunks — the val path is complete).

## 6. Compute & risk
- **Provision A100 40 GB for Lever 0 + Lever 1.** A 1.7B model, LoRA r16 on the thinker + `train_audio_encoder=true`,
  bf16, grad-checkpointing, per-device batch 4 × 30 s sits comfortably under 40 GB — even the unfrozen encoder's
  AdamW fp32 states don't crowd the ceiling. Lever 0/1 is the cheap-probe phase where a NO-GO is a live outcome;
  don't overpay to probe a dark horse.
- **Go to 80 GB only after GO**, when you hit: open-gate #1 escalation (full-FT thinker+encoder if LoRA underfits —
  fp32 optim states for 1.7B ≈ 20 GB+ alone), souping / bigger batch for wall-clock, or you just want OOM headroom
  on the 30 s tail clips. Insurance, not a requirement.
- **Cost (measured):** the 1-epoch gate ran in **~3.3 h** (`train_runtime` 11,877 s) on one A100 at 45 s /
  batch 4×4 over 33,037 clips — **40 GB held** (no OOM). So the full **5-epoch** recipe ≈ **~16 h** (length-grouping may trim it); add Lever-2
  filtering runs + soup seeds on top. **QLoRA (4-bit)** fits smaller GPUs if needed.
- **Headline risk:** unsupported language (see top). Mitigations baked in: `train_audio_encoder=true`, the Lever-0
  gate, the (passed) tokenizer gate, and willingness to **full-FT the encoder or raise LoRA rank** if LoRA underfits.
- **Hallucination on corruption:** cap `max_new_tokens`, placeholder-fill empties (never the ref).
- **Inference-time budget (Phase-2):** vLLM keeps generative decoding fast (1st place: 220 k clips in ~1 h), so
  a Qwen or MMS⊕Qwen submission is within limits — confirm on the smoke.
- **Attribution:** keep runs single-lever where cheap; the gate isolates "can it learn the language" from "is the recipe tuned".

## 7. Open gates (decide at kickoff)
1. **PEFT vs full-FT:** ✅ **LoRA r16 confirmed sufficient** — the gate hit 0.2242 with **no sign of underfit**
   (loss descended cleanly, combined tracked down). Keep r16 for the full recipe. BUT the audio encoder is trained
   (`train_audio_encoder=true`) and **must be persisted** — see §3 lessons #1 (the save bug); `infer_qwen.py`
   quantifies how much it contributed. Only escalate rank/full-FT if the 5-epoch run plateaus above MMS.
2. **Prompt conditioning (Lever 1):** ✅ **language-named prompt used and works** (gate ran `prompt_mode=named` →
   0.2242). The cheap neutral/closest A/B is still open but low-priority given named already clears the gate.
3. **Data:** gate + Levers 1–2 on **WAXAL-only** first (clean comparison to MMS); add external (Lever 3) only if it helps val.
4. **`ɔ`/`ɛ`:** non-issue for WAXAL (§4). Keep raw unless the Lever-3 `--external` tokenizer re-run shows byte-fallback
   rot → then reuse v2's `CHAR_MAP`.

## 8. Definition of done
- A GO/NO-GO verdict from Lever 0, on the record either way (a clean negative is a valid, useful outcome).
- On GO: a tuned Qwen3-ASR fine-tune evaluated on the **same fixed val + board-faithful metric**, reported
  per-language, head-to-head vs **MMS v2 (0.2033)**.
- The decorrelation check: **MMS ⊕ Qwen** ensemble val vs the better single model — the real payoff even if Qwen only ties.
- External data + licenses disclosed (Qwen3-ASR Apache-2.0; FLEURS/LRSC CC-BY-4.0) per competition rules.
- One-line verdict per lever, and a recommendation: ship Qwen, ship MMS⊕Qwen, or keep MMS v2.
