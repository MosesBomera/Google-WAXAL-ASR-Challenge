# Experiment‑01 — A winning, generalizable ASR model for WAXAL (Lingala · Shona · Luganda)

> **Who this is for:** you, if you're new to ASR and to LLM fine‑tuning (LoRA, SFT, model soup, pseudo‑labeling). This document is **both a tutorial and a battle plan**. Read it top‑to‑bottom once to learn the concepts, then use the staged checklist as your build guide.
>
> **Goal:** take our current baseline (experiment‑00: **WER 0.498 / CER 0.188**, board ≈ 0.65) and build the strongest, most *generalizable* model we can — one that wins **Phase 1** and, more importantly, holds up on the **hidden Phase‑2 test set** where we get audio only (no language labels, no metadata).
>
> **How we got here:** we studied the top‑3 solutions of a sibling ASR competition (DrivenData "On Top of Pasketti — Children's Speech Recognition, Word Track"). All three independently converged on the **same recipe**. This plan repurposes that recipe for WAXAL's African languages.

---

## 0. TL;DR — the one‑paragraph strategy

Stop fine‑tuning a general‑purpose multimodal LLM (Gemma 3n) and instead fine‑tune a **speech‑specialized foundation model** with **LoRA**, on **all three languages jointly**, with the transcript‑length bug fixed, **data augmentation** (SpecAugment + noise), **pseudo‑labeling** to add more training signal, and finish with **checkpoint averaging + model soup + a 2‑backbone ensemble**. Validate everything against the public test set as an *evaluation yardstick only* — the test set never enters training in any form (not its labels, not its audio) and we never submit its labels. That combination is exactly what won the sibling competition, and every ingredient in it independently improves generalization to unseen speakers — which is what Phase 2 actually scores.

**Expected trajectory** (estimates, see §12 for the ladder):

| Milestone | What changes | WER | CER | Board* |
|---|---|---|---|---|
| experiment‑00 (baseline) | Gemma 3n, tiny LoRA, ~3000 steps/~1 epoch, `max_seq_length=64`, checkpoint by token‑accuracy | 0.498 | 0.188 | ~0.65 |
| S1 Whisper swap | Whisper‑large‑v3 LoRA, correct seq lengths, **WER‑based checkpointing**, Luganda→`sw` token | ~0.24 | ~0.09 | ~0.84 |
| S2 augmentation + label cleaning | SpecAugment + noise; agreement‑based label cleaning; tune sampling α | ~0.19 | ~0.072 | ~0.86 |
| S3 pseudo‑labeling | semi‑supervised on non‑test unlabeled audio | ~0.17 | ~0.066 | ~0.88 |
| S4 soup + ensemble | checkpoint‑avg, model soup, 2‑backbone ensemble | ~0.15 | ~0.058 | ~0.90 |

*(We skip the earlier "S0 fix the old model" stage — experiment‑00 already spent ~5 h; we apply every fix directly on Whisper in S1.)*

\* Zindi appears to display the aggregate as **higher‑is‑better ≈ 1 − (0.5·WER + 0.5·CER)**. Don't anchor on the display number — **track WER and CER directly** (we compute both locally). All numbers past the baseline are informed guesses to set expectations, not promises.

---

## 1. Concept primer (read once)

New terms are introduced as **concept cards**. If you already know one, skip it.

> **Concept — ASR (Automatic Speech Recognition).** A model that takes an audio waveform and outputs text. Modern ASR models have two parts: an **audio encoder** (turns sound into a sequence of vectors) and a **text decoder** (turns those vectors into words). This is the same "encoder → decoder" shape as translation, except the input is sound instead of a sentence.

> **Concept — WER & CER (our metrics).** **Word Error Rate** = (word substitutions + deletions + insertions) ÷ number of reference words. **Character Error Rate** = the same thing at the character level. Both are *error* rates, so **lower is better**. Our leaderboard score is the weighted mean `E = 0.5·WER + 0.5·CER`. CER is included because African orthographies have lots of spelling variation — getting most characters right even when a word is "wrong" should still earn credit. **Design implication:** a model that reproduces *casing and punctuation* faithfully wins on CER; a model that drops or truncates words loses badly on WER.

> **Concept — Foundation model & fine‑tuning.** A *foundation model* is a big network already trained on enormous data (thousands of hours of speech). *Fine‑tuning* nudges it to our specific task/languages with a small amount of our data. Starting from a speech foundation model is like hiring someone who already speaks 50 languages and teaching them 3 new ones — vs. teaching a generalist from scratch.

> **Concept — SFT (Supervised Fine‑Tuning).** Fine‑tuning on labeled `(input, correct output)` pairs — here, `(audio, transcript)`. The model is trained to predict the transcript tokens; we compute loss only on the transcript tokens (the audio/prompt tokens are "masked" so they don't contribute to the loss). experiment‑00 already does this via HuggingFace `trl.SFTTrainer`.

> **Concept — LoRA (Low‑Rank Adaptation).** Instead of updating all ~billions of weights (expensive, memory‑hungry, prone to overfitting on small data), LoRA **freezes the original model** and inserts tiny trainable "adapter" matrices next to chosen layers. You train ~0.1–2% of the parameters. Two knobs: **`r`** (rank = adapter capacity; 8 is tiny, 16–32 is a solid default) and **`alpha`** (scaling; effective strength ≈ `alpha/r`, so people set `alpha = 2·r`). **`target_modules`** picks *which* layers get adapters — attention projections `q,k,v,o` and MLP layers `gate,up,down`. experiment‑00 only adapts `v_proj, o_proj` at `r=8`; the winners adapt **all seven** at `r=16`. More target modules + higher rank = more capacity to learn new languages.

> **Concept — SpecAugment & data augmentation.** During training you randomly corrupt each audio a little — mask out time slices and frequency bands (SpecAugment), mix in background noise, slightly change speed/pitch. The label stays the same. This teaches the model to be robust to variation it hasn't seen, which is *exactly* the Phase‑2 challenge (new speakers, new recordings). Augmentation is cheap and one of the highest‑ROI generalization tricks.

> **Concept — Pseudo‑labeling (semi‑supervised learning).** Use your current model to transcribe *unlabeled* audio, keep the confident/consistent predictions as if they were real labels, and train on them too. More (approximate) training signal ⇒ better model. **Hard rule for us:** pseudo‑label **only** unlabeled WAXAL/external audio. **The test set is left entirely out of training** — we do not train on the test audio at all, and we never use the available test labels as targets or submit them. The test set is for **evaluation only** (see §10).

> **Concept — Checkpoint averaging & Model Soup.** Training saves the model at several points ("checkpoints"). **Checkpoint averaging** = average the *weights* of the last few checkpoints of one run into a single model. **Model soup** = average the weights of *several independently fine‑tuned models* (different seeds/data/augmentation). Averaging weights finds a "flatter" point in the loss landscape that generalizes better and is smoother than any single model — free accuracy with **zero extra inference cost**. This was the single biggest finishing move for all three winners (1st place souped 4 models; 2nd place souped **18**).

> **Concept — Ensembling.** Run *two different* models on each clip and combine their outputs (e.g., pick the higher‑confidence transcript, or ROVER‑style word voting). Costs 2× inference but adds robustness because two backbones make *different* mistakes. We use this only for the final submission.

---

## 2. What the top‑3 winners actually did (distilled)

All three fine‑tuned **`Qwen/Qwen3-ASR-1.7B`** — a speech‑specialized foundation model — with **LoRA**, then combined many models. The differences were minor:

| | 1st place | 2nd place | 3rd place |
|---|---|---|---|
| Backbone | Qwen3‑ASR‑1.7B (+ Parakeet‑CTC as a helper) | Qwen3‑ASR‑1.7B | Qwen3‑ASR‑1.7B |
| Adaptation | **LoRA r=16, α=32, all 7 linear modules**, train audio encoder too | SFT + SpecAugment | SFT on competition + external (TalkBank) data |
| Optimizer | lr **1e‑4**, wd 0.01, AdamW, **linear** sched, warmup 0.1, **3 epochs**, bf16 | similar | similar |
| Augmentation | **SpecAugment** (freq/time masks), no speed‑perturb | **SpecAugment + real noise mixing** | — |
| Extra data | Parakeet‑**pseudo‑labeled** data, filtered by WER agreement | **pseudo‑labeled** long clips (semi‑supervised) | external corpus |
| Finishing move | **checkpoint‑avg → merge → average 4 models (soup)** | **model soup of 18 models** | continue‑training + vLLM decode |
| Effective batch | 4 × grad‑accum 4 = **16**; `max_duration 30s`, `max_text_len ~200` | — | — |

**The transferable recipe (model‑agnostic):**

```
Speech foundation model
  → LoRA on all attention+MLP projections (r=16–32)
  → joint multilingual SFT, correct sequence lengths, lr 1e-4, ~3 epochs
  → SpecAugment + noise augmentation
  → pseudo-label extra audio, filter by agreement
  → checkpoint-average each run, then model-soup several runs
  → (final) ensemble a second backbone
  → careful decoding + text normalization
```

We will apply *this recipe*, not blindly copy their exact model, because our task differs in one crucial way (next section).

---

## 3. How our task differs — and what to keep vs. change

| Dimension | Sibling comp | **WAXAL (ours)** | Consequence |
|---|---|---|---|
| Language | English (children) | **Lingala, Shona, Luganda** — low‑resource | Backbone must handle *these* languages; Qwen's English‑centric prior may be weaker here → **validate backbone choice** (§4). |
| No. of languages | 1 | **3, trained jointly** | One multilingual model; **no language id at test time** (Phase 2). |
| Metric | WER only | **0.5·WER + 0.5·CER** | Reproducing casing/punctuation matters (CER) → prefer models that emit cased, punctuated text. |
| Test metadata | some | **none in Phase 2** | The model must infer everything from audio; **don't build any feature that needs the language prefix in the ID**. |
| Transcript length | short words | **long sentences** (Luganda mean **208 chars**, Shona 192, Lingala 150; up to ~230) | The **#1 bug** in experiment‑00 (see §5). Sequence limits must be generous. |

**Keep** from the winners: LoRA sizing, lr/schedule/epochs, SpecAugment + noise, pseudo‑labeling, checkpoint‑avg + soup, and the joint multilingual setup (our starter already trains all three at once — good).

**Change / add** for us: pick the backbone by evidence (§4), fix the length bug, add text normalization tuned to African orthography, and treat the public test set strictly as a generalization yardstick.

---

## 4. Backbone choice — decided: Whisper‑large‑v3

**Decision (locked): the anchor is `openai/whisper-large-v3`.** It's the safest, best‑documented, strongest low‑resource‑African choice; it emits **cased + punctuated** text (great for the CER half of the metric); and it natively has language tokens for **Lingala (`ln`) and Shona (`sn`)** — two of our three languages. Every step below is written for a seq2seq model like Whisper (Qwen notes called out where relevant). We commit to Whisper and get one strong model first rather than running a bake‑off up front.

The other candidates are **held in reserve as the second backbone** for the S4 ensemble (two models that make *different* mistakes buy Phase‑2 robustness):

| Backbone | Type | Role | Watch‑outs |
|---|---|---|---|
| **`openai/whisper-large-v3`** *(anchor)* | encoder‑decoder seq2seq | Primary model, S1 onward | 30 s audio cap (chunk longer clips); **no `lg` token → Luganda uses the `sw` proxy** (§8); `whisper-large-v3-turbo` for fast iteration. |
| **`Qwen/Qwen3-ASR-1.7B`** *(reserve)* | audio‑LLM | Ensemble's 2nd voice (S4) if Whisper plateaus | Exactly what won the sibling comp; heavier tooling; coverage for our languages unproven. |
| **`facebook/mms-1b-all`** *(reserve)* | CTC | Optional 3rd voice | **Natively lists `lug`, `lin`, `sna`** — but CTC output is lowercase/low‑punctuation (can hurt CER); separate decode path. |

---

## 5. Why experiment‑00 scored where it did (diagnosis)

Reading `experiments/gemma3n-baseline/{train.py, infer.py, prepare_data.py}` against our data reveals several concrete, fixable causes. In rough order of impact:

1. **Label truncation — the big one.** `train.py` sets `--max-seq-length 64`. Our transcripts average **150–208 characters** ≈ **40–70+ tokens**, and the sequence *also* includes the system+user prompt and audio placeholder tokens. So during training the *correct answer is being cut off* — the model literally never sees full targets. It learns to stop early → deletions → high WER. **Fix:** raise to ≥ 256 (measure the token‑length distribution and set to ~p99).
2. **Inference truncation.** `infer.py` uses `--max-new-tokens 128`. A 208‑char Luganda sentence can exceed 128 tokens → the tail is chopped off at generation time too. **Fix:** raise to ~256–320.
3. **Wrong tool for the job.** Gemma 3n is a general multimodal LLM, not a speech model. A speech foundation model starts far ahead on ASR (§4).
4. **LoRA too small.** `r=8`, only `v_proj, o_proj`. Not enough capacity to absorb three new languages. **Fix:** `r=16–32`, target all attention + MLP projections.
5. **Checkpoint selected on the wrong signal.** experiment‑00 actually ran ~3000 steps (~5 h, roughly **1 epoch** — not 500), so it wasn't drastically undertrained. The subtler problem: it monitored **mean token accuracy / loss**, which is *not* WER. A checkpoint with lower loss can have *higher* WER (exposure bias, over‑confident repetition). **Fix:** evaluate and select checkpoints by **WER/CER** during training (§7), and train ~3 epochs.
6. **Learning rate likely too hot.** `lr=1e-3` for LoRA on a pretrained model risks instability; winners used **1e‑4** with warmup + decay.
7. **No augmentation, no soup, no pseudo‑labeling.** The three biggest generalization levers are all absent.
8. **No text normalization.** `Train.csv` uses backslash‑escaped quotes (`\"blue\"`), embedded punctuation, and mixed casing; refs/hyps must be normalized *consistently* before scoring, and label text cleaned before training.

Per your call we **skip a same‑model fix stage** and apply all of these directly on the new Whisper backbone in **S1** — experiment‑00 already spent ~5 h, so there's no value in re‑training the wrong model.

---

## 6. The data pipeline (clean once, reuse everywhere)

Our authoritative label file is `data/google-waxal-asr-challenge20260630-10570-elxebu/Train.csv` — **39,013 rows**: `id, transcription, language, original_split`.

**Class balance (train + validation rows):**

| Language | Code | train | validation | Test.csv |
|---|---|---:|---:|---:|
| Lingala | `lin` | 14,397 | 1,843 | 1,866 |
| Shona | `sna` | 14,093 | 1,724 | 1,749 |
| Luganda | `lug` | 5,455 | 664 | 638 |

> **Note the imbalance:** Luganda has ~⅓ the data of the others but the *longest* transcripts. Full strategy in **§6a**.

**Pipeline steps (extend `prepare_data.py`):**

1. **Robust CSV parsing.** Keep `pandas.read_csv(..., escapechar="\\")` (already in `common.load_train_csv`). Verify no rows leaked into wrong columns (some transcripts contain commas, quotes, and newlines).
2. **Text normalization for *training labels*.** Un‑escape `\"`→`"`; normalize unicode (NFC); collapse repeated whitespace; standardize apostrophes/quotes. **Decide a casing/punctuation policy and apply it identically to labels *and* to the scoring of predictions** (see §8). Keep a copy of the raw text too.
3. **Audio → 16 kHz mono float32.** Already handled by `make_audio_to_array`. Confirm the sample rate the chosen backbone expects (Whisper/Qwen = 16 kHz).
4. **Length filtering + misalignment check.** Drop or chunk clips > 30 s (Whisper/Qwen cap). Measure the **duration distribution first**; if long clips are rare, drop them from *training* but still handle them at *inference* by chunking + stitching. Drop empty/whitespace‑only transcripts. Also flag **misaligned pairs** — clips whose *characters‑per‑second* is wildly off (too much/too little text for the audio duration); these are usually bad CSV rows. A stronger, model‑based label‑cleaning pass follows in S2 (§9).
5. **Token‑length audit.** Tokenize all transcripts with the chosen tokenizer, look at the histogram, set `max_target_length` ≈ p99. (This is how we permanently kill the truncation bug.)
6. **Deduplicate** exact `(audio, text)` repeats.
7. **Save per‑(split, language) Arrow chunks** (the current script already does this efficiently — one chunk resident at a time).

> The existing `prepare_data.py` is a good backbone for this; experiment‑01 forks it and adds normalization + length audit + optional external data.

### 6a. Handling the Luganda imbalance

Luganda is the odd one out: **the least data (5,455 train vs ~14k each for `lin`/`sna`) but the longest transcripts** (208 vs 150–192 chars). In one shared multilingual model, the majority languages dominate the gradient and the minority gets under-served. Here is the strategy, cheapest and most important first.

**1. Diagnose before treating — always report per‑language WER/CER.** Never look only at the average. A model can post a great mean while quietly failing Luganda. *Crucially:* the **sequence‑length fix in S1 likely helps Luganda most**, because its long transcripts were the ones being truncated at experiment‑00's `max_seq_length=64`. So measure the S1 model per language before adding more imbalance machinery — some of the gap may close for free. (Luganda is *also* the language with no native Whisper token, §8 — so its treatment is twofold: the `sw`‑proxy token *and* balanced sampling.)

**2. Work out how much Luganda actually matters — macro vs micro averaging.** Two ways a leaderboard can combine languages:
- **Macro** (average the three per‑language scores): Luganda is **⅓ of your score** despite being only ~15% of the clips → fixing it is *very* high value.
- **Micro / pooled** (all clips in one bucket): Luganda is only ~16% of the reference words → still worth fixing, but less decisive.

We don't know which Zindi uses, so **compute and log both locally**, and after the first submission compare them to the board number to infer the convention. That tells us how aggressively to prioritize Luganda.

**3. Balanced sampling — the main lever (introduced in S1, tuned in S2).** Instead of drawing training clips in natural (imbalanced) proportion, draw language *l* with probability ∝ `p_l^α`, where `p_l` is its data share and **α ∈ [0,1]** is a "temperature":
- α = 1 → natural proportions (Luganda stays a minority),
- α = 0 → perfectly uniform (each language equally likely),
- **α ≈ 0.5 → the standard multilingual‑ASR sweet spot** (what Whisper, XLS‑R, and MMS use). At α=0.5 Luganda is upsampled ~1.7× and `lin`/`sna` are lightly downweighted.

A simpler equivalent for a first pass: **duplicate Luganda examples ~2×** in the training list so its effective size approaches the others.

**4. Make the extra Luganda passes *count* — pair oversampling with augmentation.** Oversampling repeats the same clips, which risks memorization. With SpecAugment + noise on (S2), each epoch sees a *different* corrupted version of each Luganda clip, so the extra passes add genuine diversity rather than rote repetition. You can also augment Luganda a bit harder than the others.

**5. Tune α on the held‑out split, and double‑gate.** Sweep α ∈ {1.0, 0.5, 0.3, 0.0}. Keep the value that improves **Luganda** WER/CER **without** hurting the **aggregate** on the internal speaker‑disjoint split (§10). Don't over‑upsample — pushing α→0 eventually trades away `lin`/`sna` (and the pooled score) for diminishing Luganda gains. Moderate wins.

**6. Escape hatch (deferred).** If competition‑only Luganda data proves too thin after all of the above, Mozilla Common Voice has a large Luganda set — the cleanest possible fix. Per your call we **try competition‑only first** and only reach for this if Luganda plateaus.

> **Bottom line:** measure per language → correct sequence lengths + `sw`‑proxy token (S1) → moderate temperature sampling (α≈0.5) + augmented oversampling (S1/S2) → tune α on held‑out. That sequence usually closes most of the gap without external data.

---

## 7. Training configuration (the core of experiment‑01)

Create `experiments/whisper-lora/` by copying experiment‑00's four files and editing. Below are the **recommended settings**, with the winners' exact values in the Qwen column for faithful repurposing.

### 7a. Whisper‑large‑v3 (recommended primary)

| Setting | Value | Why |
|---|---|---|
| Model | `openai/whisper-large-v3` (or `-v3-turbo` to iterate faster) | Strong low‑resource African ASR; cased+punctuated output |
| Method | LoRA (PEFT) | Fits on a single 24–40 GB GPU |
| `lora_r` / `lora_alpha` | **32 / 32** (ratio ≈ 1.0 — *not* 2×) | High α/r (e.g. 64/32) is a known trigger for memorization‑hallucinations in Whisper LoRA; start α≈r and tune |
| `target_modules` | **encoder *and* decoder** `q_proj, k_proj, v_proj, out_proj` (+ `fc1, fc2`) | Adapt the audio encoder too — the acoustic/phonetic space of these languages is new (winners trained the encoder) |
| `lora_dropout` | 0.05 | Light regularization for generalization |
| `learning_rate` | **1e‑4** | Winner value; stable for LoRA (full‑FT fallback: ~1e‑5) |
| Schedule | linear or cosine decay, **warmup_ratio 0.1** | Winner value |
| Epochs | **3–5** over full data | Winners used 3; low‑resource can take a bit more |
| Effective batch | **16** (A100: bs 8 × grad‑accum 2) | Winner value |
| Precision | **bf16**, gradient checkpointing on | Fits memory |
| `max_target_length` | **~256** (from §6.5 audit) | Kills truncation bug |
| Weight decay | 0.01 | Winner value |
| **Checkpoint selection** | `predict_with_generate=True`, `metric_for_best_model="wer"`, `greater_is_better=False`, `generation_max_length≈256`; eval on a ~500–1000 subset for speed | **Select by WER, not loss/token‑accuracy.** Also save often + keep top‑K‑by‑WER for checkpoint‑averaging (§9) |
| **Language token** | Lingala→`<\|ln\|>`, Shona→`<\|sn\|>`, **Luganda→`<\|sw\|>`** | Whisper has ln/sn but **not lg** — use Swahili as the proxy (§8) |
| SpecAugment (from S2) | `apply_spec_augment=True`, `mask_time_prob≈0.05`, `mask_feature_prob≈0.05` | Built into the Whisper encoder — no extra library needed |

### 7b. Qwen3‑ASR‑1.7B (co‑primary — copied from the 1st‑place `qwen_exp022.yaml`)

```yaml
model_name_or_path: "Qwen/Qwen3-ASR-1.7B"
train_audio_encoder: true          # they train the encoder too
train_projector: false
prompt: "Transcribe the audio."     # generic — NOT "children's voice"; no language hint
max_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4      # effective batch 16
gradient_checkpointing: true
precision: "bf16"
lora: { r: 16, lora_alpha: 32, lora_dropout: 0.0,
        target_modules: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] }
optim: { lr: 1e-4, weight_decay: 0.01, optim_name: adamw_torch }
scheduler: { name: linear, warmup_ratio: 0.1 }
augment:
  spec_augment: { freq_masks: 1, freq_width: 15, time_masks: 1, time_width: 0.05, prob: 0.5 }
data: { max_duration_sec: 30.0, max_text_len: 256 }   # raised from 200 for our long transcripts
```

> **When to reach for Qwen (S4 only):** if Whisper plateaus, verify the base model + license are accessible and that Qwen adds value on *our* languages — otherwise use it purely as the ensemble's second voice.

---

## 8. Decoding, language handling, and text normalization

These "boring" details move the metric a lot and are where beginners leak points.

- **No language id at inference.** Phase 2 provides audio only. **Never route by the `lug_/lin_/sna_` prefix in the ID.** Train the model to work from audio alone. For Whisper: during training set the language token per sample (it learns faster), but at inference **let the model auto‑detect** so the same code path works in Phase 2. For Qwen: use one generic prompt for all languages.
- **⚠️ Luganda has no Whisper language token.** Whisper's tokenizer includes Lingala (`ln`) and Shona (`sn`) but **not Luganda (`lg`)**. Options for the Luganda token during training: (a) **`<|sw|>` (Swahili)** — the closest widely‑supported Bantu / East‑African token, our recommended default; (b) a single **universal token for all three** languages. Fine‑tuning overwrites the token's original meaning, so the decoder learns to emit Luganda orthography under whichever token we assign. This is our hardest language *and* the un‑tokened one (§6a) — **decide and A/B‑test this in S1.** At inference, auto‑detect should then route Luganda audio to that same token.
- **Guard against Whisper hallucination** (it invents/loops text on silence — which tanks WER). At inference: `condition_on_prev_tokens=False`, add a **VAD pre‑filter**, and blank/redo segments with `avg_logprob < ~-1.0` or `compression_ratio > ~2.4` (repetition inflates the ratio). Keep `no_repeat_ngram_size=3`. At *training* time the main lever is keeping LoRA **α≈r** (§7a) — over‑large α is a documented hallucination trigger.
- **`max_new_tokens ≈ 256–320`** at inference (never 128). Match your training `max_target_length`.
- **Greedy vs. beam:** start greedy (fast). Try `num_beams=5` at the end — small, reliable gain, but slower. Add `no_repeat_ngram_size=3` to suppress repetition loops that spike WER.
- **Long‑audio inference:** for any clip > 30 s, chunk (e.g. 30 s windows with ~1 s overlap), transcribe each, and concatenate. Whisper has built‑in chunked long‑form decoding you can reuse.
- **Text normalization at scoring time — apply the *same* transform to reference and hypothesis:** unicode NFC, unescape quotes, strip/normalize punctuation consistently, collapse whitespace, and decide casing. **Do not lowercase blindly** — casing/punctuation earn CER credit, so a normalizer that *keeps* them (and just fixes obvious noise) usually scores best when the model reproduces them well. Test both a "keep case/punct" and a "strip" normalizer locally and keep whichever scores better; whatever you pick, it must be identical for refs and hyps.
- **Fill every submission row.** A blank `Target` scores as fully wrong. The provided `infer.py` already warns on missing rows — get it to **0 missing** before submitting.

---

## 9. Generalization engine — augmentation, pseudo‑labeling, soup, ensemble

Phase 2 is scored on **unseen speakers/recordings**, so generalization *is* the prize. Five levers, cheapest first:

1. **Augmentation (do always).** SpecAugment (via Whisper's built‑in `apply_spec_augment`, §7a) + **noise mixing** (mix in ambient/babble noise at random SNR) + optional **reverb**. Free robustness. Open noise sources: MUSAN, WHAM!, DEMAND (all openly licensed — disclose them). *Note:* the 1st‑place solution **disabled speed‑perturb**, so treat speed/pitch as optional and A/B‑test rather than assume it helps.
2. **More/related data — DEFERRED (decision: competition‑only first).** We first see how far WAXAL's own labeled data takes us; external corpora are held in reserve, not used in the initial S1–S4 runs. *If* we plateau, the disclosable open options are: **FLEURS** (has `ln`, `lg`, `sn`), **Mozilla Common Voice** (Luganda is large — also our Luganda‑imbalance escape hatch, §6a), **ALFFA/Kencorpus**, and a few **related Bantu languages** (Swahili, Kinyarwanda) as auxiliary joint‑training data to strengthen the encoder's African‑phonetics prior. Any external set used must be **public, permissively licensed, and disclosed** (challenge rule), and used as training data only — never as eval.
3. **Pseudo‑labeling (semi‑supervised).** Once you have a decent S2 model: transcribe **non‑test** unlabeled audio (unused WAXAL clips, related‑language / external audio — **never the competition test audio**), **filter** to keep only confident/consistent transcripts (the winners kept pairs where a second model's prediction agreed within a WER threshold, e.g. ≤ 0.7), and fold them into training. Retrain → measurable gain. The test set stays out of training entirely (§10).
4. **Training‑label cleaning by model agreement (from the 1st‑place solution).** WAXAL's labels contain noise (garbled/misaligned CSV rows). After S1, run the model over the **train** set, compute per‑sample WER(prediction, label), and **drop or down‑weight the worst tail** (e.g. WER > 0.8) as probable mislabels, then retrain. This is *cleaning existing labels* — distinct from pseudo‑labeling, which *adds* new ones. (Applies only to train data; the test set is never touched, §10.)
5. **Checkpoint averaging → Model Soup → Ensemble (the finishing moves).**
   - **Checkpoint‑avg:** average the weights of the best ~3 checkpoints of each run (**ranked by WER**, not loss). Free.
   - **Model soup:** train 3–6 runs varying seed / augmentation strength / data mix / LoRA rank, merge each run's LoRA into its base, then **average the merged weights**. Free at inference, and the biggest single generalization boost the winners reported.
   - **Ensemble:** for the final submission, run your two best *different* backbones and combine (confidence pick or ROVER word‑voting). 2× inference, extra robustness.

> **Order of operations that mirrors the 1st‑place pipeline:** train several LoRA runs → checkpoint‑average within each → merge adapters → average the merged models (soup) → optionally ensemble a second backbone → decode.

---

## 10. Rules, anti‑leakage, and validation strategy (read carefully)

**What this test set is.** The Phase‑1 test set (our `Test.csv`, 4,253 clips) is a normal split of the WAXAL Hugging Face dataset, so its ground‑truth transcriptions are *available* — not "leaked," just published alongside the rest of the corpus. **It is also the set Zindi scores the public leaderboard on.** So this one set plays two roles: (a) we **do submit predictions for it** to the public leaderboard every stage, and (b) because we hold its labels, we can compute the exact same WER/CER locally — our local number should track the board closely.

**The organizers' rule (verbatim from `docs/data-description.md`):** *"Any Phase 1 submission that uses the publicly available ground‑truth labels for the Phase 1 test set will be treated as a breach of the challenge rules and may lead to disqualification."*

**Our stance is EVALUATION ONLY, which is even simpler to keep clean: the test set never enters training in any form — not its labels, not its audio. We submit model predictions for it; we never feed it back into the model.**

- ✅ **The only allowed use:** measure WER/CER locally on the test set — to track progress, pick hyperparameters, choose checkpoints, and select soup/ensemble members — **and submit our model's predictions to the leaderboard.** Both are fine; neither trains on the labels.
- ❌ **Forbidden — no exceptions:** training with the true test labels as targets; using them to filter or weight data; using them to select anything that feeds back into training; **training on the test audio at all** (including pseudo‑labels derived from it); submitting the true test labels as predictions; any pipeline where a test‑set label or a test clip influences model weights.
- 🔒 **Practical guardrail:** keep the test data physically separate from every training path (a distinct directory/loader), so it is impossible for a test clip to leak into a training set by accident.
- ⚠️ **Don't overfit the public leaderboard.** Only ~30% of the board is public; the other 70% (and all of Phase 2) is hidden. Techniques that only help the public split (e.g. memorizing its quirks) will *hurt* you privately. Prefer changes that help a **held‑out‑by‑speaker** local split too.

**Validation setup:**
- Primary local metric: WER/CER on the **Phase‑1 test set** (the same set behind the public leaderboard), reported **per language** and averaged. Because it *is* the leaderboard set, local and board scores should agree closely — and it's our best available proxy for how Phase 2 will feel.
- Guardrail: keep a **speaker‑disjoint** internal split. First check whether the WaxalNLP HF dataset exposes a `speaker`/`session` field; if so, split with **GroupKFold by speaker** (the sibling 2nd‑place used `StratifiedGroupKFold`). A speaker‑disjoint split is a far better Phase‑2 proxy than a random one and prevents the "validation looks great, private tanks" trap. If no speaker id exists, split by recording/session as a fallback.
- A change ships only if it improves **both** the internal held‑out split and the public test set. That double‑gate is what protects the Phase‑2 score.

---

## 11. Compute plan & fallbacks

**Decided target: a single A100 40 GB on vast.ai.** That's plenty for the whole recipe — we don't need the winners' multi‑GPU 40–60 h rigs to win Phase 1.

**A100 40 GB reference config (what the code targets):**
- Whisper‑large‑v3, LoRA **r=32 / α=32** (encoder+decoder), bf16 + gradient checkpointing.
- Per‑device batch **8**, grad‑accum **2** → effective batch **16** (drop to bs 4 × ga 4 if a long‑clip batch spikes memory).
- `max_target_length ≈ 256`; 3–5 epochs over ~38k clips.
- The full soup (several runs averaged) runs fine here — each run is one training job; averaging is cheap.

**vast.ai practical notes:**
- Use a **persistent volume / mount** for `data/` + checkpoints so a preempted or torn‑down instance doesn't lose work; push final artifacts to HF Hub or a bucket.
- Cache the HF dataset + base model on the volume to avoid re‑downloading each session.
- Iterate cheaply with **`whisper-large-v3-turbo`** for quick loops, then confirm on `large-v3` for real runs.
- If you ever need to trim cost/time: `large-v3-turbo`, r=16, fewer soup members — in that order.

**LoRA vs full fine‑tuning:** for *adding new languages*, LoRA often matches or beats full FT (one study: 22.97% vs 28.33% WER) at ~24× less compute, and its regularization helps Phase‑2 generalization — so **LoRA is our default**. If it plateaus, try full **decoder** fine‑tuning at lr ~1e‑5 as an experiment (more robust, but slower and more forgetting‑prone).

**If time/compute is tight, cut in this order (keep the top items):** correct sequence lengths + WER checkpointing → Whisper backbone → 3 epochs → SpecAugment → checkpoint‑avg. Drop first: 2nd backbone ensemble, large soups, beam search.

---

## 12. Staged build plan (the checklist)

Each stage is shippable and independently measurable. **Submit after each stage** (5/day, 200 total — spend them to confirm local↔board correlation, don't burn them).

### S1 — Whisper‑large‑v3, full training (the foundation) (1–2 days)
Fork experiment‑00 → experiment‑01; swap the model loader to Whisper.
- [ ] Data prep: token‑length audit (`max_target_length≈256`), audio‑duration audit (>30 s policy), text normalization, misalignment flagging (§6).
- [ ] Fine‑tune **`openai/whisper-large-v3`** with the §7a settings (LoRA r=32/**α=32**, encoder+decoder, lr 1e‑4, ~3–5 epochs, effective batch 16, bf16).
- [ ] **Wire WER/CER into eval and select checkpoints by WER** (`predict_with_generate`, `metric_for_best_model="wer"`, `generation_max_length≈256`); save often for later averaging.
- [ ] **Decide the Luganda token** (`<|sw|>` proxy vs universal) and A/B‑test it (§8).
- [ ] Turn on **language‑balanced (temperature) sampling, α≈0.5** for Luganda (§6a); log per‑language WER/CER.
- [ ] **Verify:** clear improvement over baseline, Luganda gap narrowing; submit.

### S2 — Augmentation + label cleaning + sampling tuning (1–2 days) — *competition data only*
- [ ] Add Whisper SpecAugment (`apply_spec_augment`) + noise mixing (MUSAN/WHAM); A/B‑test speed/pitch.
- [ ] **Model‑agreement label cleaning:** score S1 model vs train labels, drop/down‑weight the worst tail, retrain (§9).
- [ ] **Tune the Luganda sampling α** ∈ {1.0, 0.5, 0.3, 0.0}; keep the best by held‑out aggregate *and* Luganda.
- [ ] External corpora stay **deferred** (revisit only if we plateau).
- [ ] **Verify:** improvement on *both* internal held‑out and public test; submit.

### S3 — Pseudo‑labeling (1–2 days)
- [ ] Transcribe **non‑test** unlabeled/related audio with the S2 model (**never the competition test audio**).
- [ ] Filter by agreement/confidence (e.g. keep pairs two models agree on within WER ≤ 0.7).
- [ ] Retrain with real + pseudo data.
- [ ] **Verify:** improvement; submit.

### S4 — Soup + ensemble (finishing, 1–2 days)
- [ ] Checkpoint‑average the best ~3 checkpoints of each run.
- [ ] Train 3–6 varied runs (seed/aug/data/rank) → **model soup**.
- [ ] Build the *other* backbone (from S1) and **ensemble** for the final submission.
- [ ] Add beam search (`num_beams=5`, `no_repeat_ngram_size=3`) if it helps locally.
- [ ] **Verify:** best local WER/CER on both splits; submit the ensemble as the Phase‑2 model.

### S5 — Phase‑2 readiness (before the deadline)
- [ ] Confirm the inference path needs **only audio** (no ID prefix, no metadata).
- [ ] Long‑audio chunking works; every row fills; 0 missing.
- [ ] Freeze the exact model + decoding config that scored best on the *private‑style* internal split (not just the public board).
- [ ] Write the solution doc (models, data, external sources disclosure) — top‑10 must submit code within 48 h.

---

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Whisper weak on our languages | Whisper natively has `ln`/`sn`; Qwen/MMS held as ensemble fallback if Whisper stalls (§4). |
| **Luganda has no Whisper token** | Train it under the **`sw` proxy** token (or universal); A/B‑test in S1 (§8). |
| Whisper hallucination (invented/looping text) | Keep LoRA **α≈r**; at decode use `condition_on_prev_tokens=False`, VAD, logprob/compression‑ratio rejection, `no_repeat_ngram_size` (§8). |
| Checkpoint picked on wrong signal | Select by **WER/CER**, never loss/token‑accuracy (§7). |
| Overfitting the public leaderboard | Double‑gate every change on an internal **speaker‑disjoint** split; prefer generalization levers (aug/soup). |
| Long transcripts truncated | Token‑length audit → `max_target_length`/`generation_max_length` at p99. |
| Luganda under‑served (least data, longest text) | Temperature sampling (α≈0.5); per‑language metrics; watch its WER (§6a). |
| Noisy/misaligned training labels | Char‑per‑second filter + model‑agreement label cleaning (§9). |
| Test‑set leakage into training | Test set = **evaluation only**; its labels *and* audio are kept out of every training path (separate dir/loader); never a training target, never pseudo‑labeled, never submitted. |
| Compute limits | Turbo/medium variants, LoRA, grad‑checkpointing; cut‑list in §11. |
| Repetition/hallucination at decode | `no_repeat_ngram_size`, sane `max_new_tokens`, greedy‑first. |
| Runtime too slow for Phase‑2 window | Prefer a single strong soup over a huge ensemble; benchmark inference early. |

---

## 14. First actions

1. `cp -r experiments/gemma3n-baseline experiments/whisper-lora`; swap the model loader from Gemma 3n to `openai/whisper-large-v3`.
2. Run the **token‑length audit** and **audio‑duration audit** (§6) — these set `max_target_length` and the >30 s policy in minutes.
3. **Wire WER/CER eval into training** (`predict_with_generate`, `metric_for_best_model="wer"`) so checkpoints are selected on the right signal from the very first run.
4. Run **S1** (Whisper‑large‑v3 LoRA), decide the **Luganda token** (`sw` proxy), turn on α≈0.5 sampling, and submit.

> S1 onward is "the winners' recipe, applied to African languages." Keep the discipline of **measuring per‑language WER/CER on a speaker‑disjoint held‑out set for every change**, select checkpoints by WER, and let the model soup do the heavy lifting at the end. That's how this converts into a Phase‑2 win, not just a Phase‑1 leaderboard spike.
