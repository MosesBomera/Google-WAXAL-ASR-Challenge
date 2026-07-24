# RUN — qwen-asr (Qwen3-ASR-1.7B on WAXAL) — FULL RECIPE

Self-contained `src/` for a flat vast.ai upload. See `../plan-qwen-asr.md` for the why.

**Status (2026-07-23).** Pre-gate checks PASSED (tokenizer → GO/keep-raw). **Lever-0 gate PASSED —
strong GO** (best 0.2242, 1 epoch, within 0.021 of MMS's 0.2033). A **checkpoint-save bug** (trained audio
encoder wasn't serialized → LoRA-only reload scored 0.55) was found and **FIXED**. The gate box was **torn
down**, so this run starts fresh. **The experiment we are running now is the full 5-epoch recipe (run 1):
re-prep → smoke → the run.** This runbook covers that.

## 0. Box setup (conda, python 3.13; A100 40 GB, ~70 GB disk)
```bash
conda create -n qwen-asr python=3.13 -y && conda activate qwen-asr
pip install -r ../requirements.txt      # pinned to the reference winners
export HF_TOKEN=hf_...                   # needed to stream google/WaxalNLP
```
Upload to the box: `src/*.py`, `requirements.txt`, `manifests/{train,val}.csv`, `data/Train.csv`.
- `manifests/` = the FROZEN speaker-disjoint split from the mms line; val is byte-identical to the MMS
  yardstick. **Reuse as-is — do NOT re-split.** `find_manifest_dir()` auto-resolves `workdir/manifests`
  (flat box) or `../mms-ctc/manifests` (repo) — no `--manifest-dir` flag needed on the box.
- `data/Train.csv` — authoritative transcripts; `common.data_dir()` finds it at `data/Train.csv` or
  `data/google-waxal-…/Train.csv` — no flag.

## 1. Prepare data — REQUIRED (box is fresh)
Trainer reads audio ARRAYS from prepared Arrow chunks (schema: id, audio{array}, transcription, language,
duration, speaker_id, gender), not wavs. Writes `prepared/waxal-raw/{train,validation}/<lang>/` (the default
`--audio-dir`). **No GPU needed** for this step.
```bash
python prepare_data.py                                   # full stream of google/WaxalNLP (~48 GB, ~30–60 min)
python prepare_data.py --max-examples-per-language 100   # tiny set to pair with --smoke (fast)
```
Does NO filtering/splitting — the frozen manifests handle membership by id at train time.

## 2. SMOKE — validate the POST-GATE fixes before the ~16 h spend (do NOT skip)
```bash
python train_qwen.py --smoke --output-dir artifacts/qwen-smoke
python infer_qwen.py --adapter artifacts/qwen-smoke/checkpoint-5 --eval-set val --val-subset-per-lang 4
```
The gate already proved the core; the smoke now guards the four things added SINCE the gate. **Confirm in the log:**
- `[lora] … audio-encoder unfrozen: 319.8M` and `[grad] hooked text input embeddings…` (or `enable_input_require_grads() OK`)
- `[aug] SpecAugment ON (train only) …`  ← SpecAugment wired
- **`[save] trained_base.safetensors: … ~320M params … ~0.6 GB`**  ← the ENCODER-SAVE FIX. If the encoder isn't
  captured the run **hard-fails** with `[encoder-save guard]` — that's intended (a silent LoRA-only save can't recur).
- gen-eval prints a `combined` (garbage value fine at 32 clips).
- The `infer_qwen` line must print **`[trained_base] restored … encoder tensors`** — proves the save↔load round-trips.

## 3. THE EXPERIMENT — full 5-epoch recipe (run 1: pure, unfiltered baseline)
```bash
python train_qwen.py --output-dir artifacts/run1-pure
```
No `--gate` → **5 epochs** (`--epochs` default 5.0). Recipe defaults: LoRA r16/α32/dropout0 on
[q,k,v,o,gate,up,down]_proj, `train_audio_encoder=1`, lr 1e-4, warmup 0.1, linear, bf16, grad-ckpt, eff. batch
16 (4×4), **max_duration 45 s**, **max_new_tokens 384**, **SpecAugment ON**, **named prompt**, length-grouped
batching, **unfiltered** WAXAL-only (`--train-manifest train`). Fits **A100 40 GB**; ~**16 h** (gate 1 ep = 3.3 h;
length-grouping may trim it).

Why unfiltered: this is the honest baseline that (a) measures Lever-2's later gain, (b) validates the encoder-save
fix on a real run, and (c) is itself a soup member. Watch `artifacts/run1-pure/logs/gen_eval.jsonl` — expect it to
beat the gate's 0.2242 (5× epochs + SpecAugment + encoder now saved). Selecting on `combined`, the LR anneals to 0
at the end of epoch 5 — if val is still improving there, **warm-start** more (see below) rather than pre-raising it.
Then the true head-to-head:
```bash
python infer_qwen.py --adapter artifacts/run1-pure/checkpoint-BEST --eval-set val   # full 3,753 val vs MMS 0.2033
```
(pick `checkpoint-BEST` from the lowest `combined` in `gen_eval.jsonl`).

## 4. Lever 2 — quality filtering → runs 2+ (script pending: `score_filter.py`)
Once built: score every train clip with MMS-v2, drop WER>0.9 (~3–5%), then:
```bash
python score_filter.py --scorer v2-mms                       # → manifests/train_wer0.9.csv (one-time)
python train_qwen.py --train-manifest train_wer0.9 --output-dir artifacts/run2-wer0.9
python train_qwen.py --train-manifest train_wer0.7 --output-dir artifacts/run3-wer0.7
```
These threshold variants ARE the soup members (Lever 4). Val is never filtered.

## 5. Soup + ensemble (script pending: `soup.py`) — the closer
Average run1/run2/run3 adapters (+trained_base) → soup; then the cross-arch prize **MMS-v2 ⊕ Qwen**.
Final submission uses the inference levers below.

## 6. Final val eval / submission (infer_qwen)
Loads base + adapter + trained encoder under MATCHED training conditions.
```bash
python infer_qwen.py --adapter <ckpt-or-soup> --eval-set val --num-beams 5   # beam search (WER win)
#   --fix-repetitions ON by default (collapses hallucinated repeats); --no-fix-repetitions to ablate
```
Beam + repetition-fix are inference-only (training/gen-eval stay greedy). `--eval-set test` needs prepared
board-test chunks (stub for now).

## Resuming / adding epochs (à la train_ctc)
- **Crash / preemption → same run, same schedule:**
  ```bash
  python train_qwen.py --resume-from-checkpoint auto --output-dir artifacts/run1-pure
  ```
  Restores optimizer/scheduler/step **and** pre-loads the trained encoder, then finishes the original `--epochs`.
- **"5 epochs weren't enough" → ADD epochs (warm start, fresh LR schedule):**
  ```bash
  python train_qwen.py --init-from artifacts/run1-pure/checkpoint-BEST --epochs 2 --lr 5e-5 \
         --output-dir artifacts/run1-cont
  ```
  Don't just raise `--epochs` on a finished run (its schedule already annealed to 0). The two flags are mutually exclusive.

## Preserve to HF (disk pressure / teardown)
```bash
python push_model.py --path artifacts/run1-pure/checkpoint-BEST --repo-id <you>/waxal-qwen3-asr-run1
```
Pushes adapter + `trained_base.safetensors` + configs (resume-state excluded by default; `--include-optimizer` to keep it).

## Outputs & notes
- `artifacts/<run>/checkpoint-*/` — adapter + `trained_base.safetensors` (~0.6 GB) + configs (vLLM-inferable). With
  optimizer, a checkpoint is ~2.2 GB; `--save-total-limit` default 5 → ~11 GB, plus prepared ~48 GB → provision ~70 GB.
- `logs/gen_eval.jsonl` — per-eval board-faithful metrics (**select on `combined`**, not loss). `logs/run_config.json` — exact args.
- **`--external`** (Lever 3, low priority) pulls `fleurs-*`/`lrsc-*` into TRAIN only; re-run `tok_check.py --external` first.
- Deliberately NOT in the recipe: noise aug (deferred), speed-perturb (Qwen 1st-place disabled it), lowercase targets
  (our metric scores CASED CER). See the plan's §3 audit.
- Pending scripts: `score_filter.py` (Lever 2), `soup.py` (Lever 4), `infer_qwen.py` test path.
