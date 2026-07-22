# RUN — mms-ctc on vast.ai (A100-40GB)

Run sequence for a **fresh box**. Full rationale: [`../plan-mms-ctc.md`](../plan-mms-ctc.md).

```
folder/
  *.py           <- all scripts, FLAT (this is what the code expects)
  data/          <- Train.csv, Test.csv, SampleSubmission.csv   (upload these)
  prepared/ manifests/ vocab/ lm/ artifacts/   <- created by the steps below
```
All paths auto-resolve off the folder — **no path flags needed**. Run from inside it.

---

## 0. Setup
```bash
cd /workspace
export HF_TOKEN=hf_...          # needs access to google/WaxalNLP
python -c "import torch,transformers,datasets,bitsandbytes,jiwer; print('deps ok', torch.cuda.get_device_name(0))"
df -h .                          # need ~70 GB: ~49 GB audio + ~18 GB checkpoints
```
Upload only `*.py` + `data/`. **Don't** upload `manifests/`/`vocab/` — step 2 regenerates them identically (deterministic, seed 42).

## 1. Download the data — the slow step (~hours)
```bash
python prepare_data.py --max-examples-per-language 100   # smoke: ~1 min
python prepare_data.py                                   # full: 38,199 clips / ~212 h
```
Writes `prepared/waxal-raw/` + `data/waxal_clip_metadata.csv`.

✅ **Check:** `speaker_id looks good: 595 distinct speakers across 38199 clips`
❌ If it says speaker_id is EMPTY → stale chunks were reused; re-run with `--no-skip-existing`.

## 2. Freeze the training set — offline, ~2 s, no GPU
```bash
python clean_split.py
```
Writes `manifests/{train,val}.csv`, `vocab/vocab.json`, `lm/corpus.txt`, `clean_split_report.json`.

✅ **Check:** `✓ verified 0 speaker overlap` · lin `13,667 train / 1,530 val` · vocab `103`

## 3. Train
```bash
python train_ctc.py --num-train-epochs 0.05 --eval-steps 10   # smoke: never run end-to-end before
python train_ctc.py                                            # full FT — the primary (5 epochs)
```
✅ **Check (smoke):** `trainable ~965M/965M`, then an `EVAL combined(macro) …` line at step 10.
📏 **Use the smoke run to measure throughput** — `2,065 optimizer steps/epoch` (33,045 clips ÷ 16). Multiply by your steps/sec to get the true epoch cost *before* committing (~4–5 h/epoch expected).

**Why 5 epochs and not 8/3:** `--num-train-epochs` is **not just a ceiling** — the linear schedule decays the LR to 0 at the *end* of it. Early-stopping at epoch 3 of an 8-epoch budget leaves the LR at ~60% of peak, **never annealed** — worse than a clean 3-epoch run. So pick a budget you'll finish and let it anneal; early stopping is a safety net, not the decision-maker.

Optional control run (evidence says full-FT should win at our data size; verify once):
```bash
python train_ctc.py --train-mode adapter --artifact-dir artifacts/mms-1b-adapter
```

**Expect the val number to look WORSE than whisper-lora's 0.74.** That's correct — it's speaker-disjoint now, so it finally measures what Phase 2 scores. Don't compare the two directly.

### Warm start / resume — two different things
| Need | Command | What happens |
|---|---|---|
| **Box died / preempted** mid-run | `python train_ctc.py --resume-from-checkpoint auto` | True resume: **same** LR schedule, optimizer state, step count, RNG. Needs a `checkpoint-*/` to exist (kept by `--save-total-limit 3`). |
| **Val still improving** when the LR hit 0 | `python train_ctc.py --model-id artifacts/mms-1b-all --learning-rate 1e-5 --artifact-dir artifacts/mms-1b-all-s2` | **Warm start**, not a resume: loads the trained weights, **fresh** LR schedule + optimizer. Use a lower LR — the model is already good. Prefer this over raising `--num-train-epochs`. |

This is what makes starting at 5 epochs low-risk: an under-trained run tells you "train longer" for ~20 h and you extend from where it stopped; an over-long run that early-stops un-annealed costs the full budget *and* gives a worse model.

## 4. Average the best checkpoints — **must run on the box, before teardown**
```bash
python average_checkpoints.py        # averages top 3, scores both, keeps only if better
```
Needs the `checkpoint-*/` dirs, which exist **only here** and are pruned by `--save-total-limit` (default 3). **Cannot be done retroactively** — if you tear down first, the candidates are gone.

## 5. Save the model off-box — **before teardown**
The full model is ~3.9 GB, so push it to the Hub rather than downloading via the browser:
```bash
python push_model.py --repo-id <user>/waxal-mms-1b-ctc          # private by default
python push_model.py --path artifacts/mms-1b-all-avg --repo-id <user>/waxal-mms-1b-ctc-avg
zip -r logs.zip artifacts/*/logs manifests vocab clean_split_report.json   # small, grab via UI
```
Or push automatically at the end of training: `python train_ctc.py --push-to-hub --hub-model-id <user>/waxal-mms-1b-ctc`.

Pull it back on a fresh box with `from_pretrained("<user>/waxal-mms-1b-ctc")`. Checkpoint dirs are excluded from the upload by default (~6 GB each, useless off-box).

## 6. Inference + submission (greedy) — a LIGHT box (no data download)

Inference streams the **test** audio from HF on the fly — no `prepared/`, no `Train.csv`, no 200 GB download. A fresh box needs only: `infer.py` + `common.py` + `data/SampleSubmission.csv` + `HF_TOKEN`. The model comes from the Hub (or the local `artifacts/mms-1b-all` if same box).

```bash
# smoke first — this decode path is version-sensitive (~1 min):
python infer.py --model mosesbomera/waxal-mms-1b-ctc --dtype bfloat16 \
                --skip-submission --eval-num-samples 20
#   ✅ per-language WER ~0.3 (NOT 1.0) -> decode wired right

# full: local test eval + submission.csv
python infer.py --model mosesbomera/waxal-mms-1b-ctc --dtype bfloat16
```
- **Use `--dtype bfloat16`** — float32 at batch 8 OOM-thrashes on the 46 s clips (the auto-split backstop fires, but bf16 avoids it; ~identical WER).
- **Fill the corruption gaps before submitting.** ~7 Lingala test clips are the 1.008 s corruption (25-word ref over 1 s audio) → the model emits empty → counts as "missing". They're unwinnable; fill with a neutral placeholder (NOT the reference — using test labels = disqualification):
  ```bash
  python3 -c "import pandas as pd; d=pd.read_csv('submission.csv'); d['Target']=d['Target'].fillna('').apply(lambda x: x if str(x).strip() else 'a'); d.to_csv('submission.csv',index=False); print('empty:', int((d['Target'].astype(str).str.strip()=='').sum()))"
  ```
  ✅ `empty: 0`, then download `submission.csv` and submit.

Local full-test ≈ board: greedy scored **local combined 0.2015 (WER 0.308 / CER 0.095) → board 0.790**. Lingala is dragged by public-test corruption that Phase-2 won't have — don't tune to public-LB Lingala; the speaker-disjoint val (0.205) is the north star.

## 7. KenLM decoding (Stage 3) — TRIED, NO GAIN (keep greedy)

> **Outcome (2026-07-17): KenLM does NOT help this model — submission stays greedy.**
> Built the 4-gram LM and swept α: **0.5→2.0 gave byte-identical results** (the LM never
> overturns the acoustic top-1); **α=10 finally moved the numbers but made WER worse.** The
> full MMS-1B fine-tune is too confident (CER 0.067 on sna) for an n-gram LM to add anything —
> it already learned word statistics during fine-tuning. N-gram LMs help *weak/uncertain*
> acoustic models, not a strong in-domain FT. **Don't re-run this stage.** Steps kept below for
> the record / in case a future weaker model wants an LM.
>
> **Py3.13 install gotcha:** the PyPI `kenlm` wheel fails to build (removed C-API symbols).
> Fix: `pip install cython` → rebuild from the cloned repo (`cython --cplus -3 python/kenlm.pyx`
> then `pip install . --no-build-isolation`) so Cython regenerates a 3.13-compatible `kenlm.cpp`.
> Install the decoder without downgrading numpy: `pip install pyctcdecode --no-deps && pip install pygtrie`.

An n-gram LM fixes word-boundary errors greedy CTC can't (Lingala's WER driver). Needs a one-time `lmplz` build; then it plugs into `infer.py --kenlm`.

```bash
# a) build kenlm + install the decoder (one-time on the box)
apt-get update && apt-get install -y build-essential cmake libboost-all-dev \
    libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
git clone https://github.com/kpu/kenlm && cd kenlm && mkdir build && cd build \
    && cmake .. && make -j4 && cd /workspace
pip install kenlm pyctcdecode

# b) build the 4-gram LM from lm/corpus.txt (34k transcripts, val speakers excluded)
#    corpus is in the repo; if absent, regenerate with:  python clean_split.py
python build_lm.py --lmplz-bin kenlm/build/bin/lmplz          # -> lm/waxal.arpa

# c) measure LM vs greedy on the SAME local test (WER should drop, CER ~flat)
python infer.py --model mosesbomera/waxal-mms-1b-ctc --dtype bfloat16 \
                --kenlm lm/waxal.arpa --lm-alpha 0.5 --lm-beta 1.5 --skip-submission

# d) if better, make the LM submission and compare the board delta vs 0.790
python infer.py --model mosesbomera/waxal-mms-1b-ctc --dtype bfloat16 \
                --kenlm lm/waxal.arpa --skip-evaluation
```
- **Expect a modest gain** (WER −0.02 to −0.05): half the metric is CER, which the LM doesn't touch.
- **Tune α/β on clean val, not the corrupt test.** Defaults (0.5 / 1.5) are a sane start; proper tuning needs a val-eval mode (streams `manifests/val.csv` ids) — ask if you want it before sweeping.
- First real test is step (c) — if it errors on the vocab/labels mapping, that's the untested `pyctcdecode` path; grab the traceback.

---

## Teardown checklist
- [ ] `average_checkpoints.py` run (step 4) — checkpoints die with the box
- [ ] model pushed to the Hub (step 5)
- [ ] `logs.zip` downloaded — `metrics.jsonl`, `final_eval_full_val.json`, `run_config.json`
- [ ] `data/waxal_clip_metadata.csv` kept (it's already in the repo; only needed if it changed)

Audio (`prepared/`, ~49 GB) is **disposable** — step 1 rebuilds it.

## Already done (reports in the repo — only re-run to re-verify)
```bash
python eda.py          # distributions, threshold sweep  -> report-mms-ctc-eda.md
python verify_data.py  # corruption verdict + speaker leakage
```

## Gotchas
| Symptom | Cause / fix |
|---|---|
| `speaker_id` empty in the metadata | Stale chunks reused → `--no-skip-existing` |
| `data_dir` resolves oddly | Scripts must sit **flat** beside `data/`; never use `SCRIPT_DIR.parent` in new code — use `common.workdir()` |
| OOM in training | Lower `--per-device-train-batch-size` (raise `--gradient-accumulation-steps` to hold effective batch 16); 46 s clips are the spike |
| Disk full | `prepared/` ~49 GB + ~6 GB/checkpoint. Lower `--save-total-limit` (costs you step 4) |
| `average_checkpoints.py`: "No checkpoint-*/" | `--save-total-limit` was too low, or training was interrupted before a save |
| bitsandbytes missing | Falls back to fp32 AdamW — much more memory; install it or lower the batch |
| **OOM during `infer.py`** | Use `--dtype bfloat16` (float32 batch-8 thrashes on 46 s clips); or lower `--batch-size`. The auto-split backstop prevents a crash either way. |
| **`submission.csv` has missing rows** | The 1.008 s corruption in the public test → empty predictions. Fill with a placeholder (step 6); never with the reference text. |
| `infer.py --kenlm` errors on labels/vocab | Untested `pyctcdecode` path — check `kenlm`+`pyctcdecode` installed and the label mapping; greedy (drop `--kenlm`) always works. |
| local WER ≫ board on Lingala | Public-test corruption, not your model. Phase-2 is clean — trust the speaker-disjoint val, not public-LB Lingala. |
