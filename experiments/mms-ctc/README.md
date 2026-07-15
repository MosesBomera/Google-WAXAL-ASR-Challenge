# mms-ctc — MMS-1B (CTC) fine-tune for WAXAL

Pivot from whisper-lora (Whisper). Rationale + full spec: [`plan-mms-ctc.md`](plan-mms-ctc.md);
why we pivoted: [`report-whisper-lora.md`](../whisper-lora/report-whisper-lora.md).
Versioned by **git commits** (no `vN` folders).

**Backbone:** `facebook/mms-1b-all` · **full transformer fine-tune** (freeze only the CNN feature extractor) · CTC · **cased+punctuated vocab** · KenLM decoding.

## Stages
| Stage | What | Status |
|---|---|---|
| **1a** | Download the **entire** dataset, raw (no filtering) | ✅ built |
| **1b** | **Proper EDA** — distributions, threshold sweep, extremes | ✅ built |
| **1c** | Clean using thresholds chosen *from the EDA* + cased CTC vocab | after 1b |
| 2 | MMS-1B CTC fine-tune (full FT + regularizers + per-language WER/CER) | |
| 3 | KenLM + pyctcdecode beam decoding | |
| 4 | Recover long clips (forced alignment) + label cleaning | |
| 5 | Ensemble with Whisper-v1 | |

**Why download raw first:** the download is the expensive step (hours); cleaning decisions are cheap. Materialize once → EDA → *then* pick thresholds. Changing a threshold costs seconds, not a re-download, and we can inspect exactly what a filter would remove before committing. (We already learned the paper's `words/sec>4` filter fires **zero** times on our languages while Lingala loses ~11% to `<1.5 s` — so importing their thresholds on faith would have been a mistake.)

## Run it
On vast.ai the scripts and `data/` (with `Train.csv`) are siblings; paths auto-resolve.
```bash
export HF_TOKEN=hf_...        # to stream google/WaxalNLP

# smoke test (a few clips/lang):
python src/prepare_data.py --max-examples-per-language 100
python src/eda.py

# full:
python src/prepare_data.py        # RAW download -> ./prepared/waxal-raw   (the slow part)
python src/eda.py                 # analysis     -> ./prepared/waxal-raw/logs/eda_report.json
```

## What you get
- `prepared/waxal-raw/{train,validation}/<lang>/` — **raw** Arrow chunks (`id, audio, transcription` [cased-normalized], `language, duration`). Nothing dropped.
- `prepared/waxal-raw/logs/download_report.json` — clips + hours per (language, split), missing ids.
- `prepared/waxal-raw/logs/eda_report.json` + a printed summary:
  - **per-language distributions** — duration/words-per-sec/chars-per-sec/length percentiles + a duration histogram;
  - **threshold sweep** — exactly how many clips each candidate cut-off (`<0.5/1/1.5/2 s`, `>20/30/40/60 s`, `wps>3/4/5/6`) would remove, per language;
  - **the key diagnostic** — of the clips under 1.5 s, how many are *also* impossibly fast (i.e. genuinely corrupt) vs. legitimately short;
  - **extremes with their text** — shortest / longest / fastest / slowest, to eyeball;
  - **character inventory** (feeds the cased vocab) + rare/junk chars;
  - **duplicate transcripts**.

Thresholds get chosen from that report, then applied in Stage 1c — not imported from another paper.

## Notes
- Text keeps **case + punctuation** (unlike MMS's default lowercasing) to protect the CER half of the metric.
- Public test set stays evaluation-only — never in `prepared/`.
- Not yet run end-to-end (needs the HF stream + GPU env for later stages); compiles and the cleaning/vocab logic is unit-checked. Run the smoke test first.
