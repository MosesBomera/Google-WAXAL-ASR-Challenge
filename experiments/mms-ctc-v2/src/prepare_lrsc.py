"""mms-ctc-v2 / Stage 1e-bis — fold LRSC (Mendeley Lingala Read Speech Corpus) into training.

LRSC — 4.3h of read Lingala speech from 32 Congolese adults, Mendeley Data
(data.mendeley.com/datasets/28x8tc9n9k/1, CC-BY-4.0) — was VERIFIED disjoint from WAXAL by
check_lrsc.py (0 test/train/val transcript overlap, 0 duration-confirmed duplicate recordings, 0
speaker overlap, 100% net-new across all 2,940 clips). It ships as fairseq-style manifests, not a
HF dataset, so ingestion is local-disk: this script reuses check_lrsc.py's exact id/text/audio-path
loader (`stream_records`) rather than duplicating it.

Like prepare_external.py (FLEURS), it emits the SAME prepared/ schema train_ctc.py already loads
generically (Features({id, audio:Audio(16k), transcription, language, duration:f32, speaker_id,
gender}) -> prepared/train/lrsc-<lang>/) and appends to manifests/train.csv ONLY — LRSC's own
"valid" split folds into OUR train too (it isn't WAXAL-comparable; val stays 100% WAXAL
speaker-disjoint, same iron rule as FLEURS).

Differences from FLEURS, each checked against the real data before writing this script:
  * no resampling — every one of the 2,940 wavs is already 16kHz mono PCM16 (verified directly);
  * REAL per-clip speaker_id (`lrsc-<lang>-<name>`, 31 distinct participants encoded in the clip
    id) instead of one synthetic id per language — directly helps Lingala, our most
    speaker-poor language;
  * applies common.clean_verdict (the same duration/wps hygiene filter WAXAL's own prep uses) —
    drops ~15 too-short + 2 too-fast clips out of 2,940 (~0.6%), reported not silently dropped;
  * VOCAB COVERAGE guard (same purpose as FLEURS'): LRSC transcripts use the real Lingala
    open-vowel letters ɔ/ɛ, which WAXAL's own transcripts never use (confirmed: zero occurrences
    in Train.csv) — so these become NEW native vocab characters once build_vocab.py reruns AFTER
    this script (see plan-mms-ctc-v2.md §5), not [UNK] and not a lossy CHAR_MAP substitution.

Getting the raw files onto the training box (two ways, same downstream loader either way):
  * --download        fetch the Mendeley zip directly on the box and extract ONLY
                       LRSC/<lang-folder>/** (the zip also bundles the unlabeled CSRC corpus,
                       ~2.2 of its ~2.8GB — out of scope here, discarded after extraction).
  * --lrsc-root PATH   point at an already-present local folder (e.g. scp'd/rsync'd from a
                       machine that already has it, or pulled from a private HF mirror you push
                       yourself) — same layout check_lrsc.py already reads.

    python prepare_lrsc.py --dry-run                    # report hours + drops + OOV; write nothing
    python prepare_lrsc.py --download                   # fetch from Mendeley, ingest lin
    python prepare_lrsc.py --lrsc-root /path/to/LRSC     # ingest from an already-local copy
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter

from check_lrsc import _dur_from_audio, _lrsc_root_default, stream_records
from common import clean_verdict, normalize_text, prepared_dir, quiet_hf_logging, workdir, write_json
from prepare_external import _count_oov, append_manifest, vocab_chars

logger = logging.getLogger("prepare_lrsc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # locating the data — same flags as check_lrsc.py, feeding the same loader
    p.add_argument("--lrsc-root", default=_lrsc_root_default(),
                   help="Path to .../LRSC (parent of --lang-folder). Ignored if --download.")
    p.add_argument("--lang-folder", default="lingala", help="e.g. lingala.")
    p.add_argument("--splits", nargs="+", default=["train", "valid"],
                   help="LRSC splits to fold in — BOTH go to OUR train (see docstring).")
    p.add_argument("--waxal-languages", nargs="+", default=["lin"],
                   help="OUR language code for the ingested rows (first entry used) — must match "
                        "a WAXAL code so per-language metrics bucket right.")
    # getting the data onto this box
    p.add_argument("--download", action="store_true",
                   help="Fetch the Mendeley zip and extract only LRSC/<lang-folder>/** (discards "
                        "the rest + the zip afterward). Use on a fresh box with no local copy.")
    p.add_argument("--mendeley-url", default="https://data.mendeley.com/public-api/zip/28x8tc9n9k/download/1")
    p.add_argument("--download-dir", default=str(workdir() / "data" / "lrsc-download"))
    p.add_argument("--force-download", action="store_true", help="Re-fetch even if already extracted.")
    # output — matches prepare_external.py
    p.add_argument("--audio-dir", default=str(prepared_dir()), help="Where prepared/ chunks live.")
    p.add_argument("--manifest-dir", default=str(workdir() / "manifests"))
    p.add_argument("--vocab", default=str(workdir() / "vocab" / "vocab.json"))
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--dir-prefix", default="lrsc")
    p.add_argument("--dry-run", action="store_true",
                   help="Report hours + filter drops + OOV chars; write nothing.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# getting the raw files onto disk
# ---------------------------------------------------------------------------

def _download(url: str, dest: str) -> None:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "mms-ctc-v2/prepare_lrsc"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:  # noqa: S310 (fixed https URL)
        total = int(resp.headers.get("Content-Length", 0))
        read, next_log, chunk = 0, 50_000_000, 1 << 20
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            read += len(buf)
            if read >= next_log:
                pct = f" ({100 * read / total:.0f}%)" if total else ""
                logger.info("  …%.0f MB downloaded%s", read / 1e6, pct)
                next_log += 50_000_000
    logger.info("Downloaded %.0f MB -> %s", read / 1e6, dest)


def _extract_subtree(zip_path: str, lang_folder: str, dest_root: str) -> None:
    """Extract only entries under .../LRSC/<lang_folder>/ — matched by substring, since Mendeley
    may nest the zip under a versioned wrapper folder we haven't inspected. Everything else
    (notably the unlabeled CSRC corpus) is left zipped and never touches disk."""
    import zipfile

    marker = f"LRSC/{lang_folder}/"
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if marker in n and not n.endswith("/")]
        if not names:
            raise SystemExit(f"No entries containing {marker!r} in the zip. "
                             f"Sample entries: {zf.namelist()[:10]}")
        for n in names:
            rel = n[n.index("LRSC/"):]
            out_path = os.path.join(dest_root, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with zf.open(n) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
        logger.info("Extracted %d files -> %s", len(names), os.path.join(dest_root, "LRSC", lang_folder))


def ensure_local_root(args: argparse.Namespace) -> str:
    """Local dir whose <lang_folder>/<split>/{transcript.txt,audio/} check_lrsc.stream_records
    can read — either the already-populated --lrsc-root, or (--download) freshly fetched."""
    if not args.download:
        return args.lrsc_root
    lang_dir = os.path.join(args.download_dir, "LRSC", args.lang_folder)
    if os.path.isdir(lang_dir) and not args.force_download:
        logger.info("Reusing already-extracted %s (--force-download to refetch).", lang_dir)
        return os.path.join(args.download_dir, "LRSC")
    os.makedirs(args.download_dir, exist_ok=True)
    zip_path = os.path.join(args.download_dir, "mendeley_28x8tc9n9k.zip")
    logger.info("Downloading Mendeley zip (~2.8GB full package; only LRSC/%s/** is kept) from %s",
               args.lang_folder, args.mendeley_url)
    _download(args.mendeley_url, zip_path)
    logger.info("Extracting LRSC/%s/** ...", args.lang_folder)
    _extract_subtree(zip_path, args.lang_folder, args.download_dir)
    os.remove(zip_path)
    logger.info("Zip removed; kept only the extracted LRSC/%s subtree.", args.lang_folder)
    return os.path.join(args.download_dir, "LRSC")


# ---------------------------------------------------------------------------
# scan (--dry-run: header-only duration, no full decode) / ingest (full)
# ---------------------------------------------------------------------------

def scan_language(args: argparse.Namespace, lang: str, vchars: set[str]):
    """--dry-run path: transcript + WAV-header duration only (soundfile.info, no full decode).
    Reports would-be hours + clean_verdict drops + OOV chars; writes nothing."""
    root = ensure_local_root(args)
    oov: Counter = Counter()
    drops: Counter = Counter()
    n = kept = 0
    total_s = 0.0
    for _split, _rec_id, text, apath in stream_records(root, args.lang_folder, args.splits):
        text = normalize_text(text)
        n += 1
        if not text.strip():
            drops["empty"] += 1
            continue
        dur = _dur_from_audio(apath)
        if dur is None:
            drops["unreadable_audio"] += 1
            continue
        reason, _stats = clean_verdict(text, dur)
        if reason:
            drops[reason] += 1
            continue
        _count_oov(text, vchars, oov)
        total_s += dur
        kept += 1
    hours = total_s / 3600
    logger.info("[%s] %d scanned, %d kept, %.2f h | dropped: %s | OOV: %d distinct%s",
               lang, n, kept, hours, dict(drops), len(oov),
               (" -> " + repr(dict(oov.most_common(20)))) if oov else " (all in vocab ✓)")
    return None, [], oov, hours, drops, kept


def ingest_language(args: argparse.Namespace, lang: str, vchars: set[str]):
    """Full path: reads every LRSC wav locally (soundfile, no ffmpeg), applies clean_verdict,
    materializes into prepared/ + returns manifest rows."""
    import datasets
    import soundfile as sf

    root = ensure_local_root(args)
    sr = args.sample_rate
    records: list[dict] = []
    oov: Counter = Counter()
    drops: Counter = Counter()
    idx = 0
    for _split, rec_id, text, apath in stream_records(root, args.lang_folder, args.splits):
        text = normalize_text(text)
        if not text.strip():
            drops["empty"] += 1
            continue
        try:
            arr, file_sr = sf.read(apath, dtype="float32")
        except Exception as e:  # noqa: BLE001 — corrupt/missing file, skip and count it
            logger.warning("Unreadable audio %s (%s) — skipped.", apath, e)
            drops["unreadable_audio"] += 1
            continue
        if arr.ndim != 1 or file_sr != sr:
            raise SystemExit(f"{apath}: shape={arr.shape} sr={file_sr} — LRSC was verified "
                             f"uniformly mono/{sr}Hz; this file breaks that assumption, investigate "
                             f"before continuing rather than silently resampling/downmixing.")
        dur = len(arr) / sr
        reason, _stats = clean_verdict(text, dur)
        if reason:
            drops[reason] += 1
            continue
        _count_oov(text, vchars, oov)
        speaker = rec_id.split("_", 1)[0] if rec_id else "unknown"
        records.append({
            "id": f"{args.dir_prefix}-{lang}-{idx}",
            "audio": {"array": arr, "sampling_rate": sr},
            "transcription": text,
            "language": lang,
            "duration": float(dur),
            "speaker_id": f"{args.dir_prefix}-{lang}-{speaker}",   # REAL speaker, train-only prefix
            "gender": "unknown",                                    # LRSC has no gender metadata
        })
        idx += 1

    if not records:
        raise SystemExit(f"No usable LRSC/{args.lang_folder} records after filtering — check "
                         f"--lrsc-root/--download.")

    target = datasets.Features({
        "id": datasets.Value("string"),
        "audio": datasets.Audio(sampling_rate=sr),
        "transcription": datasets.Value("string"),
        "language": datasets.Value("string"),
        "duration": datasets.Value("float32"),
        "speaker_id": datasets.Value("string"),
        "gender": datasets.Value("string"),
    })
    ds = datasets.Dataset.from_list(records, features=target)
    hours = sum(ds["duration"]) / 3600

    rows = [{"id": r["id"], "language": lang, "split": "train", "speaker_id": r["speaker_id"],
             "gender": "unknown", "duration_s": round(r["duration"], 3),
             "n_words": len(r["transcription"].split()),
             "wps": round(len(r["transcription"].split()) / r["duration"], 3) if r["duration"] > 0 else 0.0}
            for r in records]

    logger.info("[%s] %d clips, %.2f h | dropped: %s | OOV chars: %d distinct%s",
               lang, len(ds), hours, dict(drops), len(oov),
               (" -> " + repr(dict(oov.most_common(20)))) if oov else " (all in vocab ✓)")

    out = os.path.join(args.audio_dir, "train", f"{args.dir_prefix}-{lang}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.save_to_disk(out)
    logger.info("[%s] saved -> %s", lang, out)
    return ds, rows, oov, hours, drops, len(rows)


# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()

    lang = args.waxal_languages[0]
    vchars = vocab_chars(args.vocab)
    logger.info("=" * 70)
    logger.info("Folding LRSC/%s into WAXAL %s train | vocab-so-far: %d emittable characters",
               args.lang_folder, lang, len(vchars))
    logger.info("=" * 70)

    scan = scan_language if args.dry_run else ingest_language
    _ds, rows, oov, hours, drops, n_clips = scan(args, lang, vchars)

    if oov:
        logger.warning("OUT-OF-VOCAB characters in LRSC/%s text: %d distinct -> %s",
                       args.lang_folder, len(oov), dict(oov.most_common(30)))
        logger.warning("These become NEW native vocab characters once build_vocab.py reruns AFTER "
                       "this script (plan-mms-ctc-v2.md §5) — eyeball for junk vs. real graphemes "
                       "before adding anything to common.CHAR_MAP.")
    else:
        logger.info("No out-of-vocab characters — LRSC text fits the vocab-so-far cleanly ✓")

    report = {"lang_folder": args.lang_folder, "language": lang, "dry_run": args.dry_run,
              "clips": n_clips, "hours": round(hours, 2), "dropped": dict(drops),
              "oov_distinct": len(oov), "oov_top": dict(oov.most_common(30))}
    write_path = os.path.join(args.manifest_dir, "lrsc_report.json")
    write_json(write_path, report)
    logger.info("Report -> %s", write_path)

    if args.dry_run:
        logger.info("DRY RUN — no chunks written, manifest untouched. Would-be: %d clips, %.2f h.",
                    n_clips, hours)
        return
    append_manifest(args.manifest_dir, rows)
    logger.info("Done. Next: build_vocab.py (picks up ɔ/ɛ etc. natively), then train_ctc.py.")


if __name__ == "__main__":
    main()
