# plan — mms-ctc-v2 (fresh MMS-1b full FT: more data + augmentation + longer)

**Goal:** beat the v1 mms-ctc baseline — speaker-disjoint **val combined 0.205 (WER 0.325 /
CER 0.084)**, board **0.790** — with a *fresh* full fine-tune of `facebook/mms-1b-all` that adds
external data, waveform augmentation, a rebuilt vocab, and a longer (10-epoch) schedule.
Backlog rationale: [`report-next-levers.md`](report-next-levers.md).

**Design principle — "Bundle B" (a clean A/B, not a warm-start).** Train fresh from the
`mms-1b-all` base with the *same proven recipe* as v1, changing only the things we're testing
(data, augmentation, vocab, schedule length). This isolates what the new levers buy — a
warm-start from the v1 checkpoint would confound "more data" with "more training" and would lock
the vocab. Cost: a full training run instead of a cheap resume. Worth it for a trustworthy result.

**The yardstick is sacred.** The WAXAL speaker-disjoint **val set stays byte-identical to v1's**
(same clips, same speakers) so `0.205 → new` is a fair comparison and remains our Phase-2 proxy.
`clean_split.py` is deterministic (seed 42), so rebuilding reproduces the exact same val — we
rebuild rather than depend on copied files, but we verify the val id-set matches.

---

## 1. Lessons we CARRY from v1 (do not re-litigate)

| Lesson | Setting kept |
|---|---|
| Full FT (freeze CNN feature extractor only) beats adapters at our data size | `--train-mode full`, freeze feature encoder |
| Cased + punctuated **shared** vocab — protects the CER half; no language-ID at test | single CTC head, cased vocab |
| Speaker-disjoint val is the only honest yardstick (provided split was ~100% leaked) | fixed val, unchanged |
| Data cleaning: drop <1.5 s, drop wps>4, drop empty (the 1.008 s corruption) | same thresholds in `clean_split.py` |
| Two-group LR: body **3e-5**, head **1e-4**, AdamW8bit (bitsandbytes) | unchanged |
| Regularizers: mask_time 0.065, mask_feature 0.1, layerdrop 0.1, dropout 0.05 | unchanged |
| Length-grouped batching (clips span 1.5–46 s); bf16; grad checkpointing; eff. batch 16 | unchanged |
| **Direct greedy decode is the trustworthy eval** (trainer.predict over-counts WER ~0.013) | eval via `infer.py` direct path |
| Select checkpoints by **val combined**, never loss/token-accuracy | `metric_for_best_model="combined"` |
| `num_train_epochs` sets the **LR-anneal-to-0 horizon** — pick a budget you'll finish | 10 epochs = the anneal point |
| **KenLM decoding is dead**, checkpoint (weight) **averaging is dead** | skipped entirely |

## 2. What we REBUILD vs REUSE

**Rebuild (fresh):**
- **Vocab** — from WAXAL **+ FLEURS** transcripts (`build_vocab.py`), so FLEURS characters are
  first-class outputs, not `[UNK]`. This changes the CTC head size → *forces* fresh-from-base
  (can't warm-start v1's 105-token head). That's intended.
- **Train/val split** — `clean_split.py` (deterministic; same val, verified by id-set match).
- **Train data** — WAXAL cleaned **+ FLEURS lin/sna/lug** (`prepare_external.py`).
- **The collator** — add waveform augmentation (new `augment.py`, wired into `train_ctc.py`).

**Reuse unchanged:** `common.py`, `prepare_data.py`, `clean_split.py`, `build_vocab.py`,
`infer.py` (greedy path), `push_model.py`. The frozen `manifests/`+`vocab/` copied into v2 are
now just placeholders — `clean_split`/`build_vocab` regenerate them.

---

## 3. The levers we PULL (and how each is measured)

All measured on the **fixed val**, greedy, per-language + macro. Baseline row: v1 = 0.205.

### Lever A — More data: FLEURS lin + sna + lug + LRSC lin  *(verified safe)*
~30 h read-aloud Wikipedia speech (≈10 h/lang, FLEURS) + ~5.0 h read Lingala (LRSC, Mendeley,
2,923 clips post-filter), both `INDEPENDENT` of WAXAL. Added to **train only**; val untouched.
- **Why:** Lingala is our weakest, most speaker-poor language (96 voices); the Sunbird Kinyarwanda
  curve shows 0→50 h cuts Bantu WER 33%→12.5% — we're on the steep part. LRSC also brings 31 REAL
  distinct speakers (vs. FLEURS' one synthetic id/language) — pure speaker-diversity signal.
- **Risk — domain gap:** both sources are *read*, Phase-2 is *spontaneous*. Read speech could
  transfer weakly or, for the already-healthy **sna**, even slightly hurt (FLEURS only — LRSC is
  Lingala-only so carries no sna/lug risk). **Guard:** watch **per-language** val deltas; if a
  language regresses, down-weight or drop its external portion. FLEURS is only ~12% of train and
  LRSC ~2%, so neither can dominate.
- **LRSC verification:** its on-disk layout is fairseq-style manifests, not a `datasets`
  saved-to-disk dir, so `verify_external_data.py --from-disk` didn't apply — verified instead with
  the dedicated `src/check_lrsc.py` (2026-07-19): 0/2,940 test, train, or val transcript hits, 0
  duration-confirmed duplicates, 0 speaker overlap, 100% net-new → INDEPENDENT. Ingested by
  `src/prepare_lrsc.py`. Full writeup + numbers: `report-external-data.md`.

### Lever B — Waveform augmentation (NEW), *matched to the test domain*
Today only SpecAugment (feature-map masking) is on. Add **on-the-fly, train-only** waveform
augmentation in the collator (`augment.py`).

**Lesson from Sunbird's Kinyarwanda ASR hackathon (sibling Bantu task):** they *started* with
realistic background noise (their Urban Noise Uganda set) but **switched to plain white noise**
because "the test examples are all fairly free of background noise" — over-realistic augmentation
hurt. **Principle: match augmentation to the test distribution; don't buy robustness the test
won't reward.** This flips my earlier MUSAN+RIR instinct.

- **Start LIGHT (zero-dependency):** speed perturbation ×{0.9–1.1} (p≈0.5) + additive
  **white/Gaussian noise** at SNR uniform[10–20] dB (p≈0.5). Keep SpecAugment on top. No downloads.
- **Escalate ONLY on val evidence:** MUSAN realistic noise and RIR reverb are held in reserve —
  add them only if the light version helped *and* val shows headroom. WAXAL is "natural-environment"
  audio (some real background), so a touch of realistic noise *might* help — but Phase-2 is the same
  distribution and the Sunbird result warns against overdoing it. Let val decide, not instinct.
- **Why:** manufactures the speaker/channel diversity we lack (96 lin voices); mildly roughens the
  clean FLEURS reads to bridge the read-vs-spontaneous gap — *mildly* being the operative word.
- **Measure:** val WER *and* CER, **per language** (watch that noise doesn't hurt the healthy sna).

### Lever C — Longer schedule: 10 epochs
- **Why:** v1 ran 5; the open question was whether it was under-trained. With more data + a fresh
  vocab we commit to a longer anneal.
- **Caveat (carried lesson):** 10 is the LR-anneal horizon. If val is *still climbing* at 10 →
  warm-start-extend. If it *plateaus by ~epoch 6* → we learned 10 was too long (the un-annealed
  early-stop trap), and v3 uses a shorter budget. Best-checkpoint-on-val + early-stopping
  patience is the safety net, not the decision-maker.
- **Cost is fine:** v1's 5 epochs took **~10 h**, so 10 epochs ≈ **~20 h** (+~13% for FLEURS data
  +aug overhead ≈ ~22–25 h, ~$25–35 on A100-40 GB). Affordable — the earlier 55 h estimate was wrong.
- **Reality check (still true):** more data usually needs *fewer* epochs, so watch for an early
  plateau; the cost concern is gone but the "don't over-train" lesson isn't.

### Lever D — Rebuilt vocab (no `[UNK]`) — **cased+punct LOCKED**
Built from WAXAL+FLEURS+LRSC transcripts. Enables Lever A cleanly. Val metric stays comparable —
WER/CER compare *text*, and a superset vocab can only *reduce* forced errors. **Gate:**
`prepare_external.py --dry-run` / `prepare_lrsc.py --dry-run` report out-of-vocab chars; with a
rebuilt vocab these are simply *added*, but we still eyeball the char set for junk (stray
digits/symbols to normalize via `CHAR_MAP`) vs. real graphemes to just add (LRSC's `ɔ`/`ɛ` — real
Lingala open-vowel letters WAXAL's own transcripts never use — are the latter; already handled).

**Case/punct DECIDED by `scorer_probe.py` (2026-07-18):** the board scores **CER case- AND
punct-sensitive**, **WER case-insensitive but punct-kept**, **micro-pooled**. So cased+punct
*directly* earns the CER half — Sunbird's lowercase-unpunct would throw it away. **Keep cased +
punctuated** (v1's choice, now evidence-backed). Reproduction: board CER 0.1037 ≈ cased 0.1035;
board WER 0.3159 ≈ lowercase-keep-punct 0.3139 (cased 0.3340 is far off).

### Lever E — Normalization audit (LOW yield now — see below) + board-faithful metric
**`scorer_probe.py` largely settled this.** The board keeps punctuation for both halves and already
lowercases WER, so there is **no free normalization win to harvest** — the WER gap is genuine
(morphology + real errors), not a scoring artifact. Deprioritize the orthographic-rule hunt.

**Do adopt the board-faithful metric** in v2's eval (`common.per_language_metrics` / `infer.py`):
micro-pooled; **WER** = lowercase + keep-punct; **CER** = cased + keep-punct. This makes val predict
the board tightly (public test under this metric = 0.2087 ≈ board 0.2098, vs the looser macro 0.2015).
Recompute the baseline row under this metric so v1↔v2 stay comparable.

*(Original rationale, now low-priority:)*
The 3.9× WER/CER ratio says many "wrong words" are ~1 char off — often orthographic (affix spacing,
apostrophes, loanword spelling), a *label floor* not an acoustic error.
- **Task:** diff v1's val predictions vs references on the *word-error* cases; classify how many
  are pure orthographic mismatches.
- **If a clear pattern exists:** add a principled rule to `normalize_for_ctc` applied to **both**
  training targets and eval output (must be leaderboard-consistent — normalize both sides).
- **Highest-value question (Sunbird data point):** their Kinyarwanda system trained/scored on
  **lowercase unpunctuated** text — a strong hint that *that* leaderboard normalizes case/punct
  before scoring. **Does the WAXAL scorer lowercase/strip punctuation?** If yes, our cased+punct
  vocab buys nothing and spends head capacity on case/punct the metric discards → switch targets to
  lowercase-unpunct (simpler head, likely-better CER). If no, cased+punct stays justified. v1's
  0.790 used cased+punct, so **don't overturn it without checking the scorer** — but this is the
  single most impactful thing the audit can settle.
- **Guardrail:** derive rules from linguistics + the *train* set only; validate on val; never tune
  to test. Ship only if it improves val; otherwise drop (it's a real risk if refs aren't canonical).

---

## 4. Levers we DEFER (with why — not forgotten)

| Lever | Why not now |
|---|---|
| **Per-language MMS adapters** | Architecture change + needs a language-ID front-end (Phase-2 strips the prefix). Confounds this clean data/aug A/B. → **mms-ctc-v3**, on top of the winner here. |
| **Seed / logit ensemble** | A "closer" — N× compute for a small gain. Do it once we have the best single model. |
| **Pseudo-labeling / self-training** | Needs *unlabeled* audio (FLEURS is already labeled); test-audio pseudo-labeling is a rules gray area. Later, with an external unlabeled source. |
| **Neural-LM rescoring / any decoding** | Same wall KenLM hit — model too confident. Dead. |
| **Backbone swap (w2v-BERT2 / XEUS)** | Whole new experiment (mms-ctc is the proven backbone). Horizon. |

---

## 5. Rebuild pipeline (the exact sequence on the box)

```bash
# 0. deps + HF_TOKEN, ~93 GB disk (WAXAL ~49 + FLEURS ~5 + LRSC ~2 + 2 checkpoints ~12 + download
#    cache/headroom; see RUN.md §0 for the itemized budget + LRSC's transient ~2.8 GB zip)
python prepare_data.py                 # WAXAL audio → prepared/            (slow, ~hours)
python clean_split.py                  # rebuild train/val + corpus; VERIFY val id-set == v1
python prepare_external.py             # FLEURS lin/sna/lug → prepared/train/fleurs-* + append train.csv
python prepare_lrsc.py --download      # LRSC lin (Mendeley) → prepared/train/lrsc-lin + append train.csv
python build_vocab.py                  # rebuild vocab from WAXAL+FLEURS+LRSC (now covers ɔ/ɛ too)
python train_ctc.py --num-train-epochs 10 --augment ...   # FRESH from mms-1b-all (NO --model-id)
python infer.py --model artifacts/... --dtype bfloat16 --skip-submission   # greedy val eval
python push_model.py --repo-id mosesbomera/waxal-mms-1b-ctc-v2             # keep v1 intact
```
**Guardrails baked in:**
- After `clean_split.py`, assert the val id-set equals v1's `manifests/val.csv` (comparability).
- `prepare_external.py` / `prepare_lrsc.py` append to **train.csv only**; never write val.
- `build_vocab.py` must run **after** `prepare_external.py` AND `prepare_lrsc.py` (so it sees
  FLEURS chars and LRSC's `ɔ`/`ɛ`, natively — no lossy character-mapping needed for either).

## 6. Compute & risk

- **Cost:** v1's **5 epochs = ~10 h** measured → 10 epochs ≈ **~20 h**, +~13% for FLEURS +aug
  overhead ≈ **~22–25 h (~$25–35 on A100-40 GB)**. Affordable; 10 epochs is fine. Still measure
  throughput on the smoke run, and if the CPU augmentation bottlenecks the loader, raise
  `--dataloader-num-workers` or move augmentation to GPU.
- **Overfitting:** 10 epochs raises the risk, but more data + waveform aug + existing regularizers
  push back. Watch the train-loss vs val-combined gap; best-checkpoint-on-val protects us.
- **Domain-gap regression:** monitor per-language val; drop a language's external data if it hurts.
- **Attribution:** this run bundles A–E, so a win can't be attributed to one lever. That's the
  right first move (fastest path to a better model). If it *doesn't* beat 0.205, or we want the
  breakdown, run targeted ablations (aug-only, data-only) — noted, not scheduled.

## 7. Open gates (decide before/at kickoff)

1. **Augmentation (Sunbird-informed):** *start light* — speed + white/Gaussian noise (zero-dep) —
   and escalate to MUSAN/RIR *only if val shows headroom*. **Recommended: light-first.** (This
   replaces the earlier MUSAN+RIR default; match aug to the clean-ish test domain.)
2. **FLEURS splits to ingest:** train only (~7–9 h/lang) vs train+dev+test (~10–12 h/lang, all
   external to WAXAL). *Recommended: all three FLEURS splits* — more data, all legal.
3. **Normalization (Lever E): SETTLED** by `scorer_probe.py` — board CER is case+punct-sensitive,
   WER case-insensitive+punct-kept, micro-pooled → **keep cased+punct**; adopt the board-faithful
   metric in eval. No orthographic-rule hunt needed.
4. **Epoch budget:** **10 confirmed** (~20 h / ~$25–35, per the corrected cost). Keep the
   extend-if-climbing / shorten-if-early-plateau logic.

---

## 8. Definition of done
- Fresh mms-1b-all full FT on WAXAL+FLEURS, 10-epoch anneal, waveform augmentation, rebuilt vocab.
- Greedy val (fixed set) reported per-language + macro vs the 0.205 baseline.
- Best checkpoint pushed as `waxal-mms-1b-ctc-v2` (v1 untouched).
- External data + licenses disclosed (FLEURS CC-BY-4.0) per competition rules.
- A one-line verdict per lever where separable (esp. per-language FLEURS deltas).
