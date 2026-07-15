"""Downloads WaxalNLP audio for exactly the examples listed in Zindi's
Train.csv (lin/sna/lug), decodes + resamples to 16 kHz, and saves it to disk
ONE (language, split) chunk at a time so the RAM/page-cache footprint never
grows past a single chunk.

This is the gemma3n-baseline preparer, unchanged in spirit — audio is stored
decoded (not the chat/label structure), so train.py builds Whisper features and
labels cheaply at collate time and audio isn't duplicated on disk.

Saved layout (point train.py's --data-dir here):
    <output-dir>/
        train/{lin,sna,lug}/         Arrow shards: id, audio{array,sr}, transcription, language
        validation/{lin,sna,lug}/

Usage:
    python prepare_data.py                       # full prep, all 3 languages
    python prepare_data.py --max-examples-per-language 100   # quick smoke test
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time

from common import (
    data_dir,
    detect_id_key,
    fmt_duration,
    load_streaming_split,
    load_train_csv,
    login_to_hf,
    make_audio_to_array,
    prepared_dir,
    quiet_hf_logging,
)

logger = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-id", default="google/WaxalNLP")
    p.add_argument("--train-csv", default=str(data_dir() / "Train.csv"))
    p.add_argument("--languages", nargs="+", default=None,
                   help="Restrict to these codes (default: all in Train.csv, i.e. lin sna lug).")
    p.add_argument("--sample-rate", type=int, default=16_000)
    p.add_argument("--output-dir", default=str(prepared_dir()),
                   help="Where to save materialized audio chunks (default: ./prepared/waxal-multilingual).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-examples-per-language", type=int, default=None,
                   help="Cap matched examples per (language, split) for a smoke test.")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--progress-every", type=int, default=500)
    return p.parse_args()


def materialize_language_split(dataset_id, language, split, sample_rate, audio_to_array,
                               id_to_transcription, max_examples, progress_every=500):
    import datasets

    wanted_ids = set(id_to_transcription)
    target_count = max_examples if max_examples is not None else len(wanted_ids)
    stream = load_streaming_split(dataset_id, language, split, sample_rate)

    id_key = detect_id_key(next(iter(stream)), wanted_ids)
    stream = stream.filter(lambda ex: str(ex[id_key]) in wanted_ids)
    if max_examples is not None:
        stream = stream.take(max_examples)

    seen_ids: set[str] = set()
    total_seconds = 0.0
    start = time.monotonic()

    def generator():
        nonlocal total_seconds
        for ex in stream:
            _id = str(ex[id_key])
            arr = audio_to_array(ex["audio"])
            total_seconds += len(arr) / sample_rate
            seen_ids.add(_id)
            n = len(seen_ids)
            if progress_every and n % progress_every == 0:
                elapsed = time.monotonic() - start
                rate = n / elapsed if elapsed > 0 else 0.0
                eta = (target_count - n) / rate if rate > 0 else float("nan")
                logger.info("%s/%s: %d/%d | %s | %.1f ex/s | ETA %s", language, split, n,
                            target_count, fmt_duration(elapsed), rate,
                            fmt_duration(eta) if rate > 0 else "?")
            yield {"id": _id,
                   "audio": {"array": arr, "sampling_rate": sample_rate},
                   "transcription": id_to_transcription[_id],
                   "language": language}

    features = datasets.Features({
        "id": datasets.Value("string"),
        "audio": datasets.Audio(sampling_rate=sample_rate),
        "transcription": datasets.Value("string"),
        "language": datasets.Value("string"),
    })
    ds = datasets.Dataset.from_generator(generator, features=features)
    elapsed = time.monotonic() - start
    missing = wanted_ids - seen_ids
    if missing and max_examples is None:
        logger.warning("%s/%s: %d ids had no matching audio (id field '%s'). e.g. %s",
                       language, split, len(missing), id_key, list(missing)[:5])
    logger.info("%s/%s: matched %d/%d rows, %.1f min audio, took %s",
                language, split, len(seen_ids), len(wanted_ids), total_seconds / 60,
                fmt_duration(elapsed))
    return ds


def materialize_and_save_chunk(dataset_id, language, split, sample_rate, audio_to_array,
                               id_to_transcription, max_examples, progress_every, chunk_path):
    ds = materialize_language_split(dataset_id, language, split, sample_rate, audio_to_array,
                                    id_to_transcription, max_examples, progress_every)
    n = len(ds)
    ds.save_to_disk(chunk_path)
    logger.info("%s/%s: saved %d examples to %s", language, split, n, chunk_path)
    cache_files = [f["filename"] for f in getattr(ds, "cache_files", []) or []]
    del ds
    gc.collect()
    for f in cache_files:
        try:
            os.remove(f)
        except OSError:
            pass
    return n


def prepare(args: argparse.Namespace) -> None:
    run_start = time.monotonic()
    login_to_hf(args.hf_token)
    audio_to_array = make_audio_to_array(args.sample_rate)

    logger.info("Reading %s ...", args.train_csv)
    train_csv = load_train_csv(args.train_csv)
    languages = args.languages or sorted(train_csv["language"].unique())
    logger.info("Languages: %s", languages)

    total = 0
    for split in sorted(train_csv["split"].unique()):
        logger.info("=== Split: %s ===", split)
        split_rows = train_csv[train_csv["split"] == split]
        for lang in languages:
            lang_rows = split_rows[split_rows["language"] == lang]
            id_to_transcription = dict(zip(lang_rows["id"], lang_rows["transcription"]))
            if not id_to_transcription:
                logger.info("%s/%s: no rows, skipping", lang, split)
                continue
            chunk_path = os.path.join(args.output_dir, split, lang)
            if args.skip_existing and os.path.isdir(chunk_path):
                logger.info("%s/%s: %s exists, skipping", lang, split, chunk_path)
                continue
            logger.info("Materializing %s/%s (%d rows) ...", lang, split, len(id_to_transcription))
            total += materialize_and_save_chunk(
                args.dataset_id, lang, split, args.sample_rate, audio_to_array,
                id_to_transcription, args.max_examples_per_language, args.progress_every, chunk_path)

    logger.info("Done. %d examples saved this run. Total time: %s", total,
                fmt_duration(time.monotonic() - run_start))
    logger.info("Next: python train.py --data-dir %s", args.output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    prepare(parse_args())


if __name__ == "__main__":
    main()
