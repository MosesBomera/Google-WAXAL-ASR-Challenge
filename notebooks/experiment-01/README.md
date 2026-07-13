# experiment-01 / v1 — Whisper-large-v3 LoRA (the plan's "S1")

First version of experiment-01. Fine-tunes `openai/whisper-large-v3` with LoRA
on Lingala + Shona + Luganda, jointly, and selects checkpoints by **WER/CER**.
See [`docs/plans/experiment-01.md`](../../../docs/plans/experiment-01.md) for the full rationale.

> **Naming:** every experiment is a version of experiment-01. This is **v1**.
> Later ideas (augmentation, label cleaning, pseudo-labeling, soup) become
> v2, v3, … each self-contained under `notebooks/experiment-01/vN/`.

## Files
| File | Role |
|---|---|
| `common.py` | Shared helpers: text normalization, **per-language WER/CER**, temperature sampling, **log-to-disk**, HF/audio utils. |
| `prepare_data.py` | Streams WaxalNLP audio for the Train.csv ids, resamples to 16 kHz, saves per-(split, language) Arrow chunks. |
| `train.py` | Whisper-large-v3 LoRA training (Seq2SeqTrainer, `predict_with_generate`, WER-based checkpointing). |
| `infer.py` | Per-language evaluation on the test split + `submission.csv` generation, with hallucination guards. |

## What v1 does differently from experiment-00
- **Whisper-large-v3** (speech model) instead of Gemma 3n.
- **No truncation:** `max_target_length` / `generation_max_length` = 256 (was 64/128).
- **LoRA r=32 / α=32** (ratio ≈1.0 to avoid Whisper-LoRA hallucinations), on encoder **and** decoder attention + MLP.
- **Per-language token:** Lingala→`ln`, Shona→`sn`, **Luganda→`sw`** (Whisper has no `lg` token).
- **Language-balanced (temperature) sampling** α=0.5 for the Luganda imbalance.
- **Checkpoint selection by WER/CER**, reported **per language** (not token-accuracy/loss).
- **Logs persisted to disk** (see below).

## Layout on vast.ai
The scripts and the `data/` folder are siblings; the CSVs sit directly in `data/`:
```
<workdir>/
  data/            Train.csv, Test.csv, SampleSubmission.csv   (labels/ids; audio is streamed from HF)
  common.py  prepare_data.py  train.py  infer.py  README.md
  prepared/        <- prepare_data.py writes materialized 16 kHz audio chunks here
  artifacts/       <- train.py writes the LoRA adapter + logs here
```
All default paths are resolved **relative to the scripts**, so just `cd` into the
folder and run — no path flags needed. (`data_dir()` auto-detects this layout; it
also still works from the local repo checkout.)

## Run (single A100 40 GB, vast.ai)
```bash
export HF_TOKEN=hf_...            # needs access to google/WaxalNLP + accept whisper-large-v3 terms

# 1) Smoke test first (a few clips per language) — proves the whole pipeline in minutes:
python prepare_data.py --max-examples-per-language 40
python train.py --num-train-epochs 1 --eval-steps 20 --save-steps 20 \
                --max-eval-samples-per-language 20
python infer.py --eval-num-samples 20 --skip-submission

# 2) Full run:
python prepare_data.py
python train.py                                  # ~3 epochs; defaults match docs/plans §7a
python infer.py --artifact-dir ./artifacts/whisper-large-v3-lora
```
Defaults: LoRA r=32/α=32, lr 1e-4, 3 epochs, per-device batch 8 × grad-accum 2
(eff. 16), bf16 + gradient checkpointing. Override any via `--flag` (see `--help`).

## Logs saved to disk (nothing lost on exit) → `artifacts/whisper-large-v3-lora/logs/`
- `train.log` / `infer.log` — full run logs (FileHandler on the root logger).
- `metrics.jsonl` — **live** stream, flushed every step: each training log dict and every per-language eval. Survives a crash.
- `run_config.json` — exact args for this run (reproducibility).
- `trainer_log_history.json` — the complete `Trainer` history at the end.
- `final_eval_full_val.json` — final per-language metrics on the **full** validation set (auto-detect, Phase-2 style). During training, checkpoint selection uses a balanced 300/lang = 900 subset for speed; the final number uses all of validation.
- `test_eval.json` — per-language metrics on the test split (from `infer.py`).

Per-language keys everywhere: `wer_lin, cer_lin, wer_sna, cer_sna, wer_lug, cer_lug`,
plus `wer_macro/cer_macro`, pooled `wer/cer` (micro), and `combined` (=0.5·WER_macro+0.5·CER_macro,
the checkpoint-selection metric). We log both macro and micro to compare against the board (§6a).

## Rules guardrail
The test labels are used **only** to compute WER/CER (evaluation). They are never
a training target and are never submitted. `train.py` only ever reads
`data-dir/{train,validation}`; the test split lives solely in `infer.py`'s eval path.

## Requirements
Covered by the competition `requirements.txt` (transformers≥4.52, datasets, peft,
jiwer, librosa, soundfile, accelerate, torch, torchcodec, numpy). No extra deps.

## Not yet run end-to-end
The code compiles and the pure-Python logic (sampling, normalization) is unit-checked
locally, but a full train/infer needs the GPU env (torch + HF download). **Run the
smoke test above first** to validate the pipeline before the 3-epoch run.

### Known things to watch on first run
- **PEFT + `predict_with_generate`:** works on transformers ≥4.52; if eval-time
  language auto-detection misbehaves, force per-language eval as a stopgap.
- **Luganda→`sw` token** is an A/B choice — compare `--luganda-language swahili`
  vs a single universal token in a later run; keep whichever gives lower `wer_lug`.
- Eval uses generation (slow); it's capped to `--max-eval-samples-per-language`.
