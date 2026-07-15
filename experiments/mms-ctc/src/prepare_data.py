"""mms-ctc / Stage 1a — download the ENTIRE WaxalNLP dataset for the
Train.csv ids (lin/sna/lug) and save it RAW. No filtering, no cleaning.

Captures every field we need downstream, so we never have to re-download to
answer a question:
  * audio (16 kHz mono), transcription (from Train.csv — the authoritative label),
  * **speaker_id** — enables a SPEAKER-DISJOINT validation split (the right
    Phase-2 proxy; the provided split may share speakers between train/val),
  * **gender** (normalized: the corpus mixes 'Female'/'female'/'Male'/'male'/''),
  * duration + derived n_words / words-per-sec, so cleaning thresholds can be
    chosen from a tiny metadata file instead of touching 212 h of audio.

Two outputs:
  1. <output-dir>/{train,validation}/<lang>/  — raw Arrow chunks (with speaker_id).
  2. <metadata-out>  — ONE small CSV (~4 MB) with a row per clip. Keep this
     locally: it's everything needed for cleaning decisions, speaker-disjoint
     splitting, and EDA, with no audio and no re-download.

Cleaning is deliberately NOT done here (see report-mms-ctc-eda.md);
it's a separate, cheap, re-runnable step driven by the metadata.

    python prepare_data.py                                   # full download
    python prepare_data.py --max-examples-per-language 100    # smoke test
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from collections import Counter, defaultdict

from common import (
    SCRIPT_DIR,
    data_dir,
    detect_id_key,
    fmt_duration,
    load_streaming_split,
    load_train_csv,
    login_to_hf,
    make_audio_to_array,
    normalize_text,
    percentiles,
    prepared_dir,
    quiet_hf_logging,
    setup_file_logging,
    write_json,
)

logger = logging.getLogger("prepare_data")

# Thresholds recorded as FLAGS in the metadata (not applied here) — the
# 1.008/1.045 s clips with full transcripts are corpus-corrupt (verified against
# the Hub's raw audio); see verify_data.py.
SHORT_S = 1.5
FAST_WPS = 4.0

# A saved chunk is only reusable if it has EVERY field we need. Chunks written by
# an older version of this script lack speaker_id/gender; skipping those and
# defaulting the missing fields silently produces a useless metadata file, so
# --skip-existing checks the schema and re-downloads anything stale.
REQUIRED_CHUNK_COLUMNS = {"id", "audio", "transcription", "language",
                          "duration", "speaker_id", "gender"}


def _chunk_is_current(chunk_path: str) -> tuple[bool, set]:
    """(reusable?, missing_columns) for an already-saved chunk."""
    import datasets

    try:
        ds = datasets.load_from_disk(chunk_path)
    except Exception:
        return False, REQUIRED_CHUNK_COLUMNS
    missing = REQUIRED_CHUNK_COLUMNS - set(ds.column_names)
    return (not missing), missing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-id", default="google/WaxalNLP")
    p.add_argument("--train-csv", default=str(data_dir() / "Train.csv"))
    p.add_argument("--languages", nargs="+", default=None)
    p.add_argument("--sample-rate", type=int, default=16_000)
    p.add_argument("--output-dir", default=str(prepared_dir()))
    p.add_argument("--metadata-out", default=str(SCRIPT_DIR / "metadata" / "waxal_clip_metadata.csv"),
                   help="Small per-clip metadata CSV (no audio) — download & keep this locally.")
    p.add_argument("--max-examples-per-language", type=int, default=None,
                   help="Cap clips per (language, split); unset = everything. Use e.g. 100 to smoke test.")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--progress-every", type=int, default=500)
    return p.parse_args()


def _norm_gender(g) -> str:
    """The corpus mixes 'Female'/'female'/'Male'/'male'/'' — collapse to 3 values."""
    g = str(g or "").strip().lower()
    if g.startswith("f"):
        return "female"
    if g.startswith("m"):
        return "male"
    return "unknown"


def _meta_row(_id, language, split, speaker_id, gender, dur, text) -> dict:
    n_words = len(text.split())
    wps = n_words / dur if dur > 0 else 0.0
    return {"id": _id, "language": language, "split": split,
            "speaker_id": speaker_id, "gender": gender,
            "duration_s": round(dur, 3), "n_words": n_words, "n_chars": len(text),
            "wps": round(wps, 3),
            "is_short": int(dur < SHORT_S), "is_too_fast": int(wps > FAST_WPS)}


def materialize_chunk(args, language, split, id_to_transcription, audio_to_array, chunk_path):
    """Streams one (language, split) and saves EVERY matched clip. No filtering."""
    import datasets

    wanted = set(id_to_transcription)
    target = args.max_examples_per_language or len(wanted)
    stream = load_streaming_split(args.dataset_id, language, split, args.sample_rate)
    id_key = detect_id_key(next(iter(stream)), wanted)
    stream = stream.filter(lambda ex: str(ex[id_key]) in wanted)
    if args.max_examples_per_language is not None:
        stream = stream.take(args.max_examples_per_language)

    seen_ids: set[str] = set()
    meta_rows: list[dict] = []
    total_seconds = 0.0
    start = time.monotonic()

    def generator():
        nonlocal total_seconds
        for ex in stream:
            _id = str(ex[id_key])
            arr = audio_to_array(ex["audio"])
            dur = len(arr) / args.sample_rate
            text = normalize_text(id_to_transcription[_id])
            speaker_id = str(ex.get("speaker_id", "") or "")
            gender = _norm_gender(ex.get("gender"))
            total_seconds += dur
            seen_ids.add(_id)
            meta_rows.append(_meta_row(_id, language, split, speaker_id, gender, dur, text))
            n = len(seen_ids)
            if args.progress_every and n % args.progress_every == 0:
                elapsed = time.monotonic() - start
                rate = n / elapsed if elapsed > 0 else 0.0
                eta = (target - n) / rate if rate > 0 else float("nan")
                logger.info("%s/%s: %d/%d | %s | %.1f ex/s | ETA %s", language, split, n,
                            target, fmt_duration(elapsed), rate,
                            fmt_duration(eta) if rate > 0 else "?")
            yield {"id": _id,
                   "audio": {"array": arr, "sampling_rate": args.sample_rate},
                   "transcription": text, "language": language, "duration": dur,
                   "speaker_id": speaker_id, "gender": gender}

    features = datasets.Features({
        "id": datasets.Value("string"),
        "audio": datasets.Audio(sampling_rate=args.sample_rate),
        "transcription": datasets.Value("string"),
        "language": datasets.Value("string"),
        "duration": datasets.Value("float32"),
        "speaker_id": datasets.Value("string"),
        "gender": datasets.Value("string"),
    })
    ds = datasets.Dataset.from_generator(generator, features=features)
    ds.save_to_disk(chunk_path)

    missing = wanted - seen_ids
    if missing and args.max_examples_per_language is None:
        logger.warning("%s/%s: %d Train.csv ids had no matching audio. e.g. %s",
                       language, split, len(missing), list(missing)[:5])
    logger.info("%s/%s: saved %d/%d (%.1f h) | %d speakers | took %s", language, split,
                len(seen_ids), len(wanted), total_seconds / 3600.0,
                len({r["speaker_id"] for r in meta_rows}),
                fmt_duration(time.monotonic() - start))

    cache_files = [f["filename"] for f in getattr(ds, "cache_files", []) or []]
    del ds
    gc.collect()
    for f in cache_files:
        try:
            os.remove(f)
        except OSError:
            pass
    return meta_rows


def _meta_from_existing(chunk_path: str, language: str, split: str) -> list[dict]:
    """Rebuilds metadata from an already-saved chunk (no audio decode) so the CSV
    stays complete even when --skip-existing skips the download.

    Requires a CURRENT chunk — never silently defaults a missing column, which
    would emit a metadata file that looks fine but has no speaker ids at all.
    """
    import datasets

    ds = datasets.load_from_disk(chunk_path)
    missing = REQUIRED_CHUNK_COLUMNS - set(ds.column_names)
    if missing:
        raise ValueError(
            f"{chunk_path} is stale (missing {sorted(missing)}). Re-download it "
            f"(--no-skip-existing) rather than emitting blank metadata.")
    cols = {c: ds[c] for c in ("id", "transcription", "duration", "speaker_id", "gender")}
    return [_meta_row(cols["id"][i], language, split, cols["speaker_id"][i],
                      cols["gender"][i], float(cols["duration"][i]), cols["transcription"][i])
            for i in range(len(ds))]


def _speaker_summary(meta: list[dict]) -> dict:
    """THE question for validation design: does the provided validation split share
    speakers with train? If it does, our val score is optimistic and we must build
    our own speaker-disjoint split."""
    out: dict = {}
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in meta:
        by_lang[r["language"]].append(r)
    for lang, rows in sorted(by_lang.items()):
        tr = {r["speaker_id"] for r in rows if r["split"] == "train" and r["speaker_id"]}
        va = {r["speaker_id"] for r in rows if r["split"] == "validation" and r["speaker_id"]}
        val_rows = [r for r in rows if r["split"] == "validation" and r["speaker_id"]]
        leaked = sum(1 for r in val_rows if r["speaker_id"] in tr)
        clips_per_spk = Counter(r["speaker_id"] for r in rows)
        out[lang] = {
            "n_clips": len(rows),
            "n_val_clips": len(val_rows),
            "n_speakers_total": len({r["speaker_id"] for r in rows if r["speaker_id"]}),
            "n_speakers_train": len(tr), "n_speakers_validation": len(va),
            "speakers_in_both": len(tr & va),
            "val_clips_whose_speaker_is_in_train": leaked,
            "val_speaker_leakage_pct": round(100 * leaked / max(1, len(val_rows)), 2),
            "clips_per_speaker": percentiles([float(v) for v in clips_per_spk.values()]),
            "top_speaker_share_pct": round(100 * max(clips_per_spk.values()) / max(1, len(rows)), 2),
            "gender_counts": dict(Counter(r["gender"] for r in rows)),
        }
    return out


def prepare(args: argparse.Namespace) -> None:
    import pandas as pd

    log_dir = os.path.join(args.output_dir, "logs")
    setup_file_logging(log_dir, name="prepare_data")
    write_json(os.path.join(log_dir, "prepare_config.json"), vars(args))
    login_to_hf(args.hf_token)
    audio_to_array = make_audio_to_array(args.sample_rate)

    logger.info("Reading %s ...", args.train_csv)
    train_csv = load_train_csv(args.train_csv)
    languages = args.languages or sorted(train_csv["language"].unique())
    logger.info("Languages: %s | RAW download (no filtering — cleaning is a separate step)", languages)

    meta: list[dict] = []
    run_start = time.monotonic()
    for split in sorted(train_csv["split"].unique()):
        split_rows = train_csv[train_csv["split"] == split]
        for lang in languages:
            rows = split_rows[split_rows["language"] == lang]
            id_to_transcription = dict(zip(rows["id"], rows["transcription"]))
            if not id_to_transcription:
                continue
            chunk_path = os.path.join(args.output_dir, split, lang)
            if args.skip_existing and os.path.isdir(chunk_path):
                current, missing = _chunk_is_current(chunk_path)
                if current:
                    logger.info("%s/%s: exists and is current, skipping download "
                                "(reading metadata back)", lang, split)
                    meta.extend(_meta_from_existing(chunk_path, lang, split))
                    continue
                logger.warning("%s/%s: existing chunk is STALE (missing %s) — re-downloading. "
                               "Reusing it would silently blank those fields.",
                               lang, split, sorted(missing))
            logger.info("Materializing %s/%s (%d rows) ...", lang, split, len(id_to_transcription))
            meta.extend(materialize_chunk(
                args, lang, split, id_to_transcription, audio_to_array, chunk_path))

    # ---- the small, portable metadata file ----
    if meta:
        os.makedirs(os.path.dirname(os.path.abspath(args.metadata_out)), exist_ok=True)
        df = pd.DataFrame(meta).sort_values(["language", "split", "id"])
        df.to_csv(args.metadata_out, index=False)
        size_mb = os.path.getsize(args.metadata_out) / 1e6
        logger.info("Wrote per-clip metadata: %s (%d rows, %.1f MB) — keep this locally.",
                    args.metadata_out, len(df), size_mb)
        n_spk = df["speaker_id"].astype(str).str.strip().replace("", None).nunique(dropna=True)
        if n_spk <= 1:
            logger.error("!!! speaker_id is EMPTY for every clip (%d distinct). The metadata is "
                         "unusable for a speaker-disjoint split. Re-run with --no-skip-existing.", n_spk)
        else:
            logger.info("speaker_id looks good: %d distinct speakers across %d clips.", n_spk, len(df))

        spk = _speaker_summary(meta)
        write_json(os.path.join(log_dir, "speaker_summary.json"), spk)
        logger.info("===== SPEAKER / SPLIT SUMMARY =====")
        for lang, s in spk.items():
            logger.info("  %s: %d clips | %d speakers (train %d, val %d, in BOTH %d)",
                        lang, s["n_clips"], s["n_speakers_total"], s["n_speakers_train"],
                        s["n_speakers_validation"], s["speakers_in_both"])
            logger.info("     val speaker leakage: %d/%d val clips (%.1f%%) have a speaker seen in train",
                        s["val_clips_whose_speaker_is_in_train"], s["n_val_clips"],
                        s["val_speaker_leakage_pct"])
            logger.info("     clips/speaker p50=%.0f max=%.0f | top speaker = %.1f%% of clips | gender %s",
                        s["clips_per_speaker"].get("p50", 0), s["clips_per_speaker"].get("max", 0),
                        s["top_speaker_share_pct"], s["gender_counts"])

        logger.info("===== DOWNLOAD SUMMARY =====")
        for (lang, split), g in df.groupby(["language", "split"]):
            logger.info("  %s/%s: %d clips, %.1f h", lang, split, len(g), g["duration_s"].sum() / 3600)

    logger.info("Done in %s.", fmt_duration(time.monotonic() - run_start))
    logger.info("Next: python eda.py --data-dir %s   (or analyse %s locally)",
                args.output_dir, args.metadata_out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    prepare(parse_args())
    # HF streaming leaves background connections that can throw/hang on teardown
    # ("[Errno 9] Bad file descriptor"). Everything is flushed by now.
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
