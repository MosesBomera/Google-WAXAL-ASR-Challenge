"""mms-ctc-v2 / Stage 1d-bis — is AfriVoice (lin/sna) the SAME data as WAXAL, or genuinely new?

AfriVoice is NOT a standard `datasets`-loadable set: its transcripts live in per-language
`manifest_*.json` files (line-separated JSON) read via HfFileSystem. So this dedicated tool
streams those manifests and applies the SAME content-based checks as verify_external_data.py
(which it reuses for the WAXAL side + the verdict):

  * transcript overlap vs WAXAL train / val / TEST  (TEST hit = breach material),
  * NET-NEW %  (transcripts matching nothing in WAXAL — the "is it different?" number),
  * duration-CONFIRMED duplicates  (same transcript + same audio length = the SAME recording),
    speaker-id-INDEPENDENT (their UIDs may be per-dataset),
  * speaker overlap  (reported corroboration) + a self-check: do speaker_ids AGREE on clips we
    independently confirm as duplicates? agree => ids global; disagree => per-dataset.

Duration comes from a `duration` field if the manifest has one, else from the audio file's WAV
header via soundfile (`pip install soundfile`, no ffmpeg — AfriVoice is WAV). Reading test labels
is a compliance check, never training on them. See [[waxal-public-test-labels-rule]].

    pip install soundfile huggingface_hub
    python check_afrivoice.py --lang-folder Lingala --waxal-languages lin --output-dir ../logs/afrivoice-lin
    python check_afrivoice.py --lang-folder Shona   --waxal-languages sna --output-dir ../logs/afrivoice-sna
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict

from common import data_dir, login_to_hf, quiet_hf_logging, workdir, write_json
from verify_external_data import build_waxal_refs, loose_key, verdict

logger = logging.getLogger("check_afrivoice")

TEXT_FIELDS = ("transcription", "text", "sentence", "transcript", "normalized_text")
SPEAKER_FIELDS = ("speaker_id", "speaker", "client_id", "client", "spk")
AUDIO_FIELDS = ("audio_path", "path", "audio", "wav", "file", "audio_filepath")
DUR_FIELDS = ("duration", "duration_s", "length", "audio_duration", "dur", "seconds")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--afrivoice-repo", default="datasets/DigitalUmuganda/Afrivoice",
                   help="HfFileSystem path (note the 'datasets/' prefix).")
    p.add_argument("--lang-folder", required=True, help="e.g. Lingala or Shona.")
    p.add_argument("--waxal-languages", nargs="+", required=True, help="WAXAL lang to compare, e.g. lin or sna.")
    # WAXAL-ref args (consumed by verify_external_data.build_waxal_refs)
    p.add_argument("--train-csv", default=str(data_dir() / "Train.csv"))
    p.add_argument("--val-csv", default=str(workdir() / "manifests" / "val.csv"))
    p.add_argument("--waxal-metadata", default=str(data_dir() / "waxal_clip_metadata.csv"))
    p.add_argument("--waxal-dataset-id", default="google/WaxalNLP")
    p.add_argument("--check-test", action=argparse.BooleanOptionalAction, default=True)
    # duplicate confirmation
    p.add_argument("--confirm-audio-dur", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--duration-tol", type=float, default=0.05, help="Seconds tolerance for a duration match.")
    p.add_argument("--max-audio-confirm", type=int, default=1500,
                   help="Cap WAV-header fetches for duplicate candidates (bounds network round-trips).")
    p.add_argument("--max-records", type=int, default=0, help="Cap AfriVoice records scanned (0 = all).")
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--peek", action="store_true", help="Print the first record's keys and exit.")
    p.add_argument("--output-dir", default=None, help="Default: <workdir>/logs")
    p.add_argument("--hf-token", default=None)
    return p.parse_args()


def _first(record: dict, cands) -> str | None:
    for c in cands:
        if c in record and record[c] not in (None, ""):
            return c
    lower = {k.lower(): k for k in record}
    return next((lower[c] for c in cands if c in lower), None)


# ---------------------------------------------------------------------------
# AfriVoice manifest streaming (HfFileSystem)
# ---------------------------------------------------------------------------

def list_manifests(fs, repo: str, lang_folder: str) -> list[str]:
    base = f"{repo}/{lang_folder}"
    try:
        files = fs.ls(base, detail=False)
        manifests = [f for f in files if "manifest_" in os.path.basename(f) and f.endswith(".json")]
    except Exception as e:  # noqa: BLE001
        logger.warning("ls %s failed (%s) — trying recursive glob.", base, e)
        manifests = []
    if not manifests:
        manifests = fs.glob(f"{base}/**/manifest_*.json")
    return sorted(manifests)


def stream_records(fs, manifest_files):
    for mp in manifest_files:
        with fs.open(mp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


# ---------------------------------------------------------------------------
# duration (manifest field, else WAV header over HfFileSystem)
# ---------------------------------------------------------------------------

def _dur_from_audio(fs, path: str) -> float | None:
    try:
        import soundfile as sf
        with fs.open(path, "rb") as fobj:
            info = sf.info(fobj)            # reads the header only (seeks), not the whole file
        return info.frames / float(info.samplerate)
    except Exception:  # noqa: BLE001
        return None


class AudioResolver:
    """Finds where a manifest's audio_path lives in the repo, then caches the working pattern."""
    def __init__(self, fs, repo, lang_folder):
        self.fs, self.repo, self.lang = fs, repo, lang_folder
        self.pattern = None
        self.dead = False

    def duration(self, audio_path) -> float | None:
        if self.dead or not audio_path:
            return None
        ap = str(audio_path)
        base = os.path.basename(ap)
        cands = ([self.pattern.format(ap=ap, base=base)] if self.pattern else
                 [f"{self.repo}/{self.lang}/{ap}", f"{self.repo}/{ap}",
                  f"{self.repo}/{self.lang}/audio/{base}", f"{self.repo}/{self.lang}/{base}"])
        pats = ["{repo}/{lang}/{ap}", "{repo}/{ap}", "{repo}/{lang}/audio/{base}", "{repo}/{lang}/{base}"]
        for i, c in enumerate(cands):
            try:
                if self.fs.exists(c):
                    d = _dur_from_audio(self.fs, c)
                    if d is not None and self.pattern is None:
                        self.pattern = pats[i] if len(cands) > 1 else self.pattern
                    return d
            except Exception:  # noqa: BLE001
                continue
        if self.pattern is None:      # never resolved on the first candidate → give up on audio
            self.dead = True
            logger.warning("Could not resolve audio paths (e.g. %r) — duration confirmation DISABLED; "
                           "relying on transcript overlap only.", ap)
        return None


# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()
    out_dir = args.output_dir or os.path.join(str(workdir()), "logs")
    os.makedirs(out_dir, exist_ok=True)
    login_to_hf(args.hf_token)
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=args.hf_token or os.environ.get("HF_TOKEN"))

    logger.info("=" * 78)
    logger.info("Is AfriVoice/%s the same data as WAXAL (%s), or new?",
                args.lang_folder, "+".join(args.waxal_languages))
    logger.info("=" * 78)

    manifests = list_manifests(fs, args.afrivoice_repo, args.lang_folder)
    if not manifests:
        raise SystemExit(f"No manifest_*.json under {args.afrivoice_repo}/{args.lang_folder}")
    logger.info("Found %d manifest files for %s.", len(manifests), args.lang_folder)

    if args.peek:
        rec = next(stream_records(fs, manifests))
        logger.info("First record keys: %s", sorted(rec.keys()))
        logger.info("Sample: %s", {k: (str(v)[:60]) for k, v in rec.items()})
        return

    refs = build_waxal_refs(args)          # transcripts (train/val/test), speakers, text_dur, text_spk
    resolver = AudioResolver(fs, args.afrivoice_repo, args.lang_folder)

    b = {"n": 0, "train_hits": 0, "val_hits": 0, "test_hits": 0, "no_match": 0, "test_hit_examples": [],
         "dup_clips": 0, "dup_confirmed": 0, "dup_unconfirmed": 0, "dup_spk_agree": 0, "dup_spk_disagree": 0,
         "val_speaker_clips": 0}
    ext_speakers, overlap_speakers = set(), set()
    text_field = spk_field = audio_field = dur_field = None
    n_confirm = 0
    t0 = time.time()

    for rec in stream_records(fs, manifests):
        if text_field is None:
            text_field = _first(rec, TEXT_FIELDS)
            spk_field = _first(rec, SPEAKER_FIELDS)
            audio_field = _first(rec, AUDIO_FIELDS)
            dur_field = _first(rec, DUR_FIELDS)
            logger.info("Manifest schema: transcript=%r speaker=%r audio=%r duration=%r",
                        text_field, spk_field, audio_field, dur_field)
            if text_field is None:
                raise SystemExit(f"No transcript field in manifest (looked for {TEXT_FIELDS}); keys={sorted(rec)}")
        b["n"] += 1
        k = loose_key(rec.get(text_field, ""))
        spk = str(rec.get(spk_field) or "") if spk_field else ""
        if k:
            if k in refs["test"]:
                b["test_hits"] += 1
                if len(b["test_hit_examples"]) < 5:
                    b["test_hit_examples"].append(str(rec.get(text_field))[:100])
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
                edur = float(rec[dur_field]) if dur_field and rec.get(dur_field) not in (None, "") else None
                if edur is None and n_confirm < args.max_audio_confirm and audio_field:
                    edur = resolver.duration(rec.get(audio_field))
                    n_confirm += 1
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
    src = f"AfriVoice/{args.lang_folder}"
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

    path = os.path.join(out_dir, "check_afrivoice.json")
    write_json(path, {"afrivoice_lang": args.lang_folder, "waxal_languages": args.waxal_languages,
                      "manifests": len(manifests), "audio_confirmations": n_confirm, "verdict": verdicts[src]})
    logger.info("\nSaved -> %s", path)


if __name__ == "__main__":
    main()
