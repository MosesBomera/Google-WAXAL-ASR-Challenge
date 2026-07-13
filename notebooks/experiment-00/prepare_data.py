"""Downloads WaxalNLP audio for exactly the examples listed in Zindi's
Train.csv, decodes + resamples it, and saves it to disk ONE (language, split)
CHUNK AT A TIME so the RAM/page-cache footprint never grows past a single
chunk — instead of materializing everything and saving once at the end.

Why Train.csv instead of "take everything available per language": streaming
every language config unbounded risks exceeding available memory/disk, and
the WaxalNLP paper (2602.02734v3.pdf) shows the ASR release itself totals
~1,250 transcribed hours across 14 languages — the transcribed hours for our
three target languages are already modest (Lingala ~101.5h, Shona ~99.2h,
Luganda ~46.0h per Fig. 2a). Train.csv's ~38k (train+validation) rows are
essentially the full, authoritative Phase-1 training set for lin/sna/lug —
not an arbitrary subsample — so filtering to it both bounds resource use
*and* is the correct thing to train on.

Saved layout — one subdirectory PER (split, language), not one big
DatasetDict, so nothing needs all languages resident at once:
    <output-dir>/
        train/
            lin/   Arrow shards: id, audio {array, sampling_rate},
            lug/   ...           transcription, language
            sna/
        validation/
            lin/
            lug/
            sna/

Each chunk is saved to disk and its Python/cache references are dropped
immediately after — see materialize_and_save_chunk(). train.py loads and
concatenates these chunks lazily (memory-mapped, no full re-materialization).

Audio is stored decoded (raw float32 arrays at --sample-rate), NOT the chat
'messages' structure — that's built cheaply at collate time in train.py so
audio isn't duplicated on disk.

Usage:
    python prepare_data.py
    python prepare_data.py --train-csv ../../data/.../Train.csv --output-dir ./data/waxal-multilingual
    python prepare_data.py --max-examples-per-language 200   # smoke test
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time

from common import (
    detect_id_key,
    fmt_duration,
    load_streaming_split,
    load_train_csv,
    login_to_hf,
    make_audio_to_array,
    quiet_hf_logging,
)

logger = logging.getLogger("prepare_data")

DEFAULT_TRAIN_CSV = (
    "../../data/google-waxal-asr-challenge20260630-10570-elxebu/Train.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset-id", default="google/WaxalNLP")
    p.add_argument(
        "--train-csv", default=DEFAULT_TRAIN_CSV,
        help="Zindi's Train.csv — the authoritative (id, transcription, "
             "language, split) list to materialize audio for.",
    )
    p.add_argument(
        "--languages", nargs="+", default=None,
        help="Restrict to these language codes (default: all languages "
             "present in --train-csv, i.e. lin sna lug).",
    )
    p.add_argument("--sample-rate", type=int, default=16_000)
    p.add_argument(
        "--output-dir", default="./data/waxal-multilingual",
        help="Where to save per-(split, language) chunks. Point train.py's "
             "--data-dir at this same directory.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-examples-per-language", type=int, default=None,
        help="Cap MATCHED examples taken per (language, split), applied "
             "after id-filtering. Unset = use every row in Train.csv for "
             "that language/split. Set this for a quick smoke test, e.g. 200.",
    )
    p.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True,
        help="Skip a (split, language) chunk if its output folder already "
             "exists (default: on). Lets a killed/interrupted run resume "
             "without re-downloading finished chunks.",
    )
    p.add_argument(
        "--hf-token", default=None,
        help="Hugging Face token. Falls back to the HF_TOKEN env var, then "
             "an interactive login prompt if neither is set.",
    )
    p.add_argument(
        "--progress-every", type=int, default=500,
        help="Log elapsed time / rate / ETA every N matched examples within "
             "a (language, split). Set 0 to disable in-loop progress logs.",
    )
    return p.parse_args()


def materialize_language_split(
    dataset_id: str,
    language: str,
    split: str,
    sample_rate: int,
    audio_to_array,
    id_to_transcription: dict[str, str],
    max_examples: int | None,
    progress_every: int = 500,
):
    """Streams one (language, split), keeps only rows whose id is in
    id_to_transcription (i.e. listed in Train.csv), and materializes them
    into an in-memory-arrow-backed datasets.Dataset.

    Uses the Train.csv transcription (not the HF one) so saved labels exactly
    match what the competition provides, even if the two ever diverge.
    """
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
                remaining = target_count - n
                eta = remaining / rate if rate > 0 else float("nan")
                logger.info(
                    "%s/%s: %d/%d matched | elapsed %s | %.1f ex/s | ETA %s",
                    language, split, n, target_count, fmt_duration(elapsed),
                    rate, fmt_duration(eta) if rate > 0 else "?",
                )

            yield {
                "id": _id,
                "audio": {"array": arr, "sampling_rate": sample_rate},
                "transcription": id_to_transcription[_id],
                "language": language,
            }

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
        logger.warning(
            "%s/%s: %d Train.csv ids had no matching audio in the Hub stream "
            "(id field used: '%s'). First few missing: %s",
            language, split, len(missing), id_key, list(missing)[:5],
        )
    logger.info(
        "%s/%s: matched %d/%d Train.csv rows, %.1f min of audio, "
        "took %s (%.1f ex/s)",
        language, split, len(seen_ids), len(wanted_ids), total_seconds / 60,
        fmt_duration(elapsed), len(seen_ids) / elapsed if elapsed > 0 else 0.0,
    )
    return ds


def materialize_and_save_chunk(
    dataset_id: str,
    language: str,
    split: str,
    sample_rate: int,
    audio_to_array,
    id_to_transcription: dict[str, str],
    max_examples: int | None,
    progress_every: int,
    chunk_path: str,
) -> int:
    """Materializes one (language, split) chunk and saves it straight to
    chunk_path, then drops every reference to it (including the underlying
    HF generator cache file) before returning — so RAM/page-cache usage from
    this chunk doesn't carry over into the next one.

    Returns the number of examples saved.
    """
    ds = materialize_language_split(
        dataset_id, language, split, sample_rate, audio_to_array,
        id_to_transcription, max_examples, progress_every,
    )
    n = len(ds)

    save_start = time.monotonic()
    ds.save_to_disk(chunk_path)
    logger.info(
        "%s/%s: saved %d examples to %s (took %s)",
        language, split, n, chunk_path, fmt_duration(time.monotonic() - save_start),
    )

    # Drop the intermediate HF-cache-backed Arrow file(s) from_generator wrote
    # (separate from chunk_path, which is now the authoritative saved copy) so
    # its memory-mapped pages stop being resident/charged to this process.
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

    total_examples = 0
    for split in sorted(train_csv["split"].unique()):
        split_start = time.monotonic()
        logger.info("=== Split: %s ===", split)
        split_rows = train_csv[train_csv["split"] == split]
        split_examples = 0
        for lang in languages:
            lang_rows = split_rows[split_rows["language"] == lang]
            id_to_transcription = dict(zip(lang_rows["id"], lang_rows["transcription"]))
            if not id_to_transcription:
                logger.info("%s/%s: no rows in Train.csv, skipping", lang, split)
                continue

            chunk_path = os.path.join(args.output_dir, split, lang)
            if args.skip_existing and os.path.isdir(chunk_path):
                logger.info(
                    "%s/%s: %s already exists, skipping (--no-skip-existing to redo)",
                    lang, split, chunk_path,
                )
                continue

            logger.info(
                "Materializing %s / %s (%d rows in Train.csv) ...",
                lang, split, len(id_to_transcription),
            )
            n = materialize_and_save_chunk(
                args.dataset_id, lang, split, args.sample_rate,
                audio_to_array, id_to_transcription, args.max_examples_per_language,
                args.progress_every, chunk_path,
            )
            split_examples += n

        logger.info(
            "Split '%s': %d examples saved this run, took %s",
            split, split_examples, fmt_duration(time.monotonic() - split_start),
        )
        total_examples += split_examples

    logger.info("✅ Done. %d examples saved this run.", total_examples)
    logger.info("Total time: %s", fmt_duration(time.monotonic() - run_start))
    logger.info("Next: python train.py --data-dir %s", args.output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
