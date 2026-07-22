# RUN — mms-ctc-v2 on vast.ai (A100-40GB)

Fresh full FT of `facebook/mms-1b-all` on **WAXAL + FLEURS + LRSC**, 10 epochs, light waveform
augmentation, rebuilt cased vocab, **board-faithful** eval. Full rationale:
[`../plan-mms-ctc-v2.md`](../plan-mms-ctc-v2.md). Baseline to beat: v1 public-test board-faithful
**0.2087** (board 0.790); v1 val = run it through this repo's `infer.py --eval-set val` to compare.

```
folder/
  *.py           <- all scripts, FLAT
  data/          <- Train.csv, Test.csv, SampleSubmission.csv, waxal_clip_metadata.csv
  mms-ctc/manifests/val.csv   <- OPTIONAL: v1's val ids, for the comparability guardrail
  prepared/ manifests/ vocab/ artifacts/   <- created by the steps below
```
LRSC needs no upload — `prepare_lrsc.py --download` fetches it directly from Mendeley on the box
(§3). Only use a local copy (`--lrsc-root`) if you'd rather push your own already-verified mirror.

Paths auto-resolve off the folder — no path flags needed. **Order matters** (see §2–4): never
run `clean_split.py` after `prepare_external.py` / `prepare_lrsc.py` (it would overwrite
train.csv and drop both); `build_vocab.py` must run AFTER **both** `prepare_external.py` AND
`prepare_lrsc.py`.

---

## 0. Setup
```bash
cd /workspace
export HF_TOKEN=hf_...          # needs google/WaxalNLP + google/fleurs access
pip install jiwer soundfile     # board-faithful metric + LRSC's local WAV reader (not always in the base image)
python -c "import torch,transformers,datasets,bitsandbytes,jiwer,librosa,soundfile; print('deps ok', torch.cuda.get_device_name(0))"
ffmpeg -version | head -1        # REQUIRED for FLEURS audio decode (apt-get install -y ffmpeg) — LRSC does NOT need it (reads WAV directly via soundfile)
df -h .                          # need ~93 GB: WAXAL ~49 GB + FLEURS ~5 GB + LRSC ~2 GB + 2 checkpoints ~12 GB
                                  # (+ transient ~2.8 GB if using `prepare_lrsc.py --download` — the full
                                  #   Mendeley zip, deleted right after extracting just the lingala subtree)
```
Upload `*.py` + `data/` (+ optionally `mms-ctc/manifests/val.csv`). **Don't** upload
`manifests/`/`vocab/` — steps 2–4 regenerate them. LRSC itself doesn't need uploading (see below).

## 1. Download WAXAL — the slow step (~hours)
```bash
python prepare_data.py --max-examples-per-language 100   # smoke: ~1 min
python prepare_data.py                                   # full: 38,199 clips / ~212 h
```
✅ **Check:** `speaker_id looks good: 595 distinct speakers across 38199 clips`
❌ If `speaker_id is EMPTY` → `--no-skip-existing`.

## 2. Freeze the WAXAL split — offline, ~2 s, no GPU
```bash
python clean_split.py
```
Writes `manifests/{train,val}.csv` + `clean_split_report.json`.
✅ **Check:** `✓ verified 0 speaker overlap` · `✓ val id-set matches v1 exactly` (or a warning if
v1's `val.csv` wasn't uploaded — then trust the split is deterministic).
❌ `VAL DRIFT vs v1` → a threshold/metadata changed; investigate before trusting the 0.2087 delta.

## 3. Add external data (FLEURS + LRSC, both verified safe) → train
```bash
python prepare_external.py --dry-run     # text-only OOV check (can also run locally first)
python prepare_external.py               # downloads + resamples FLEURS lin/sna/lug (~30 h)
python prepare_lrsc.py --dry-run         # text-only OOV/filter check (can also run locally first)
python prepare_lrsc.py --download        # fetches Mendeley zip, keeps only LRSC/lingala (~5 h)
```
Writes `prepared/train/fleurs-{lin,sna,lug}/` + `prepared/train/lrsc-lin/`, appends both to
`manifests/train.csv`. Both are TRAIN-only; val is untouched (LRSC's own train+valid split both
fold into OUR train — it isn't WAXAL-comparable).
✅ **Check:** `No out-of-vocab characters` for FLEURS (or a short list of stray symbols to
eyeball). For LRSC, `ɔ`/`ɛ` **are expected** as OOV here — that's fine, not junk: they're real
Lingala letters absent from WAXAL's own transcripts, and fold into the vocab natively in step 4.
Per-source clip/hour counts printed by each script.

## 4. Rebuild the vocab (WAXAL + FLEURS + LRSC)
```bash
python build_vocab.py
```
✅ **Check:** `Vocab size` ≥ v1's 105 + 2 (FLEURS chars + LRSC's `ɔ`/`ɛ` fold in), `uppercase=True`,
punctuation listed. Eyeball `vocab/vocab_report.json` rare chars for FLEURS/LRSC junk → add to
`common.CHAR_MAP` + re-run if any.

## 5. Train — fresh, 10 epochs, light augmentation
```bash
python train_ctc.py --num-train-epochs 0.05 --eval-steps 10   # smoke: throughput + wiring
python train_ctc.py                                           # full: 10 epochs (~20-25 h)
```
✅ **Check (smoke):** `trainable ~965M/965M`, `Waveform augmentation ON (train only)`, then an
`EVAL combined … (board-faithful micro)` line at step 10 (NOT 1.0).
📏 **Measure throughput on the smoke** — v1's 5 epochs = ~10 h, so 10 epochs of WAXAL alone ≈
~20 h; with FLEURS + LRSC data and aug overhead, plan-mms-ctc-v2.md §3 (Lever C) estimates
**~22–25 h**. If the CPU augmentation starves the GPU, raise `--dataloader-num-workers` (or
`--no-augment` to isolate).

- Best checkpoint is selected on the **board-faithful `combined`** (not loss).
- **Expect val to look like a board number now** (~0.20s), not v1's looser 0.205 macro.
- Ablation (optional, isolates augmentation): `python train_ctc.py --no-augment --artifact-dir artifacts/mms-1b-all-noaug`.

### Warm start / resume
| Need | Command |
|---|---|
| Box died mid-run | `python train_ctc.py --resume-from-checkpoint auto` (same schedule/optimizer/step) |
| Val still improving at epoch 10 | `python train_ctc.py --model-id artifacts/mms-1b-all --learning-rate 1e-5 --artifact-dir artifacts/mms-1b-all-s2` (fresh schedule, lower LR) |

## 6. Evaluate + submit — the authoritative numbers
```bash
# smoke: val yardstick on a few clips
python infer.py --model artifacts/mms-1b-all --dtype bfloat16 --eval-set val --skip-submission

# v1 through the SAME board-faithful metric, for a clean delta:
python infer.py --model mosesbomera/waxal-mms-1b-ctc --dtype bfloat16 --eval-set val --skip-submission

# board proxy + submission.csv (corruption gaps auto-filled with a placeholder):
python infer.py --model artifacts/mms-1b-all --dtype bfloat16 --eval-set test
```
- `VAL combined` is the **yardstick** — beat v1's val. `TEST combined` is the **board proxy**
  (v1 = 0.2087). Both use raw refs vs the char-mapped model output = board-faithful.
- **Use `--dtype bfloat16`** (float32 batch-8 OOM-thrashes on 46 s clips; bf16 ~identical WER).
- `submission.csv` fills empty (corruption) rows with `a` automatically — **never** the reference.

## 7. Save the model off-box — before teardown
```bash
python push_model.py --repo-id mosesbomera/waxal-mms-1b-ctc-v2       # private; v1 untouched
zip -r logs.zip artifacts/*/logs manifests vocab clean_split_report.json
```

---

## Gotchas
| Symptom | Cause / fix |
|---|---|
| `ffmpeg`/`torchcodec` error in `prepare_external.py` | FLEURS audio decode needs ffmpeg — `apt-get install -y ffmpeg` (only the `--dry-run` is text-only) |
| `ModuleNotFoundError: soundfile` in `prepare_lrsc.py` | not always in the base image — `pip install soundfile` (§0) |
| `prepare_lrsc.py --download` can't find `LRSC/<lang>/` in the zip | Mendeley's zip layout wasn't verified against the live URL before this was written — it fails loudly with a sample of the zip's actual entries; adjust the extraction marker, or fall back to `--lrsc-root` with a manually-fetched copy |
| `VAL DRIFT vs v1` | thresholds/metadata changed → check, or `--allow-val-drift` if intentional |
| `clean_split` overwrote FLEURS/LRSC rows | you re-ran it after `prepare_external.py`/`prepare_lrsc.py` — rerun steps 3→4 (order matters) |
| vocab has junk rare chars | add them to `common.CHAR_MAP`, re-run `build_vocab.py` (LRSC's `ɔ`/`ɛ` are NOT junk — see §3/§4) |
| GPU starved during train | augmentation on CPU — raise `--dataloader-num-workers`; or `--no-augment` to check |
| OOM in train | lower `--per-device-train-batch-size` (raise `--gradient-accumulation-steps` to hold eff. batch 16) |
| OOM in `infer.py` | `--dtype bfloat16` / smaller `--batch-size`; the auto-split backstop prevents a crash |
| val ≫ board on Lingala | public-test corruption, not the model — trust the speaker-disjoint val |

## Teardown checklist
- [ ] model pushed to the Hub (step 7)
- [ ] `logs.zip` downloaded (`metrics.jsonl`, `final_eval_full_val.json`, `eval_val.json`, `run_config.json`)
- [ ] val + test board-faithful numbers recorded vs v1 (0.2087)

## Not here (deliberately)
KenLM decoding and checkpoint averaging were **proven dead ends in v1** — omitted. Per-language
adapters / seed ensemble / pseudo-labeling are **v3** (see `../report-next-levers.md`).
