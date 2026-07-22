"""mms-ctc-v2 / Stage 1d-ter — is LRSC (Mendeley, Lingala Read Speech Corpus) the SAME
data as WAXAL, or genuinely new?

LRSC ships on disk as fairseq-style manifests, not a HF `datasets` dir: for each split
(train/valid) there's a `transcript.txt` ("<id> <transcript>", one per line) and matching
`audio/<id>.wav` files. That's neither a HF Hub repo (check_afrivoice.py's model) nor a
`load_from_disk`-compatible arrow dir (what verify_external_data.py's --from-disk expects),
so it gets its own tiny local-disk loader. It reuses the SAME content-based checks as
verify_external_data.py / check_afrivoice.py (which it imports for the WAXAL side + verdict):

  * transcript overlap vs WAXAL train / val / TEST  (TEST hit = breach material),
  * NET-NEW %  (transcripts matching nothing in WAXAL — the "is it different?" number),
  * duration-CONFIRMED duplicates  (same transcript + same audio length = the SAME
    recording) — files are local, so duration comes straight from soundfile, no network,
  * speaker overlap  (reported corroboration only): LRSC ids encode a participant-name
    prefix ("betty_221011-125449_lin_359_elicit_0" -> speaker "betty"); unlikely to match
    WAXAL's obfuscated speaker_ids, but checked anyway + a self-check on confirmed dups.

Reading test labels is a compliance check, never training on them. See
[[waxal-public-test-labels-rule]].

    pip install soundfile
    python check_lrsc.py --lang-folder lingala --waxal-languages lin --output-dir ../logs/lrsc-lin
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from common import data_dir, login_to_hf, quiet_hf_logging, workdir, write_json
from verify_external_data import build_waxal_refs, loose_key, verdict

logger = logging.getLogger("check_lrsc")


def _lrsc_root_default() -> str:
    return str(data_dir().parent / "Speech Recognition Datasets for Congolese Languages" / "LRSC")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lrsc-root", default=_lrsc_root_default(), help="Path to .../LRSC (parent of --lang-folder).")
    p.add_argument("--lang-folder", default="lingala", help="e.g. lingala.")
    p.add_argument("--splits", nargs="+", default=["train", "valid"])
    p.add_argument("--waxal-languages", nargs="+", required=True, help="WAXAL lang to compare, e.g. lin.")
    # WAXAL-ref args (consumed by verify_external_data.build_waxal_refs)
    p.add_argument("--train-csv", default=str(data_dir() / "Train.csv"))
    p.add_argument("--val-csv", default=str(workdir() / "manifests" / "val.csv"))
    p.add_argument("--waxal-metadata", default=str(data_dir() / "waxal_clip_metadata.csv"))
    p.add_argument("--waxal-dataset-id", default="google/WaxalNLP")
    p.add_argument("--check-test", action=argparse.BooleanOptionalAction, default=True)
    # duplicate confirmation
    p.add_argument("--confirm-audio-dur", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--duration-tol", type=float, default=0.05, help="Seconds tolerance for a duration match.")
    p.add_argument("--max-records", type=int, default=0, help="Cap LRSC records scanned (0 = all).")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--peek", action="store_true", help="Print the first record and exit.")
    p.add_argument("--output-dir", default=None, help="Default: <workdir>/logs")
    p.add_argument("--hf-token", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# LRSC manifest streaming (local disk)
# ---------------------------------------------------------------------------

def stream_records(lrsc_root: str, lang_folder: str, splits):
    """Yield (split, id, text, audio_path) from <lrsc_root>/<lang_folder>/<split>/transcript.txt."""
    for split in splits:
        split_dir = os.path.join(lrsc_root, lang_folder, split)
        tpath = os.path.join(split_dir, "transcript.txt")
        audio_dir = os.path.join(split_dir, "audio")
        if not os.path.isfile(tpath):
            logger.warning("No transcript.txt for split %r at %s — skipped.", split, tpath)
            continue
        with open(tpath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                rec_id, text = parts
                yield split, rec_id, text, os.path.join(audio_dir, f"{rec_id}.wav")


def _dur_from_audio(path: str) -> float | None:
    if not os.path.isfile(path):
        return None
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()
    out_dir = args.output_dir or os.path.join(str(workdir()), "logs")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=" * 78)
    logger.info("Is LRSC/%s the same data as WAXAL (%s), or new?",
                args.lang_folder, "+".join(args.waxal_languages))
    logger.info("=" * 78)

    if args.peek:
        rec = next(stream_records(args.lrsc_root, args.lang_folder, args.splits), None)
        if rec is None:
            raise SystemExit(f"No records found under {args.lrsc_root}/{args.lang_folder}")
        split, rec_id, text, apath = rec
        logger.info("First record: split=%r id=%r text=%r audio=%r (exists=%s)",
                    split, rec_id, text[:80], apath, os.path.isfile(apath))
        return

    login_to_hf(args.hf_token)     # only needed for --check-test streaming WAXAL test from the Hub
    refs = build_waxal_refs(args)  # transcripts (train/val/test), speakers, text_dur, text_spk

    b = {"n": 0, "train_hits": 0, "val_hits": 0, "test_hits": 0, "no_match": 0, "test_hit_examples": [],
         "dup_clips": 0, "dup_confirmed": 0, "dup_unconfirmed": 0, "dup_spk_agree": 0, "dup_spk_disagree": 0,
         "val_speaker_clips": 0}
    ext_speakers, overlap_speakers = set(), set()
    t0 = time.time()

    for split, rec_id, text, apath in stream_records(args.lrsc_root, args.lang_folder, args.splits):
        b["n"] += 1
        k = loose_key(text)
        spk = rec_id.split("_", 1)[0] if rec_id else ""
        if k:
            if k in refs["test"]:
                b["test_hits"] += 1
                if len(b["test_hit_examples"]) < 5:
                    b["test_hit_examples"].append(text[:100])
            if k in refs["train"]:
                b["train_hits"] += 1
            if k in refs["val"]:
                b["val_hits"] += 1
            if k not in refs["test"] and k not in refs["train"] and k not in refs["val"]:
                b["no_match"] += 1
        if spk:
            ext_speakers.add(spk)
            if spk in refs.get("speakers", ()):
                overlap_speakers.add(spk)
            if spk in refs.get("val_speakers", ()):
                b["val_speaker_clips"] += 1
        # duplicate = same transcript, confirmed by duration
        if k and k in refs.get("text_dur", {}):
            b["dup_clips"] += 1
            if args.confirm_audio_dur:
                edur = _dur_from_audio(apath)
                if edur is not None:
                    if any(abs(edur - wd) <= args.duration_tol for wd in refs["text_dur"][k]):
                        b["dup_confirmed"] += 1
                        if spk and spk in refs.get("text_spk", {}).get(k, ()):
                            b["dup_spk_agree"] += 1
                        elif spk:
                            b["dup_spk_disagree"] += 1
                    else:
                        b["dup_unconfirmed"] += 1
        if b["n"] % args.log_every == 0:
            logger.info("  …%d records (%.0f/s) | test=%d train=%d net-new=%d | dups=%d (confirmed %d)",
                        b["n"], b["n"] / max(time.time() - t0, 1e-9), b["test_hits"], b["train_hits"],
                        b["no_match"], b["dup_clips"], b["dup_confirmed"])
        if args.max_records and b["n"] >= args.max_records:
            break

    b["n_ext_speakers"], b["n_overlap_speakers"] = len(ext_speakers), len(overlap_speakers)
    b["overlap_speakers_sample"] = sorted(overlap_speakers)[:10]
    src = f"LRSC/{args.lang_folder}"
    scan = {"total": b["n"], "per_source": {src: b}}
    verdicts = verdict(scan, {}, rerelease_frac=0.02)

    logger.info("\n" + "=" * 78 + "\nVERDICT\n" + "=" * 78)
    v = verdicts[src]
    logger.info("[%s]  n=%d  test=%d  train=%d (%.1f%%)  net-new=%d (%.1f%%)  dups=%d (confirmed %d)  "
                "speakers %d/%d overlap", src, v["n"], v["test_hits"], v["train_hits"],
                100 * v["train_overlap_frac"], v["no_match"], 100 * v["net_new_frac"], v["dup_clips"],
                v["dup_confirmed"], v["n_overlap_speakers"], v["n_ext_speakers"])
    logger.info("  => %s", v["verdict"])
    for ex in v["test_hit_examples"]:
        logger.info("     test-match example: %r", ex)

    path = os.path.join(out_dir, "check_lrsc.json")
    write_json(path, {"lrsc_lang_folder": args.lang_folder, "waxal_languages": args.waxal_languages,
                      "records_scanned": b["n"], "verdict": verdicts[src]})
    logger.info("\nSaved -> %s", path)


if __name__ == "__main__":
    main()
