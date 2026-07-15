"""mms-ctc / Stage 1b — two targeted data checks (TRAIN/VALIDATION ONLY).

The public test set is NEVER touched here: the organizers state the Phase-2 test
set (the one that decides awards) is clean, so any corruption in the public test
split is an artifact we neither can nor need to act on.

CHECK 1 — "corruption": are the ~1,400 clips whose audio is exactly 1.01 s while
their transcript is full-length (11-44 words/sec) genuinely corrupt IN THE CORPUS,
or is our own decode/resample truncating them? Deleting 1,400 clips on a wrong
assumption would be expensive, so we verify against the RAW, un-resampled audio
straight from the Hub and check whether the 1 s actually contains speech.

CHECK 2 — "metadata": does WaxalNLP expose a speaker/session id? If it does we can
build a SPEAKER-DISJOINT validation split, a far better Phase-2 proxy than a random
one (the open question carried over from whisper-lora). Train.csv has no speaker
column, so the dataset schema is the only place to look.

    python verify_data.py                      # both checks
    python verify_data.py --check corruption
    python verify_data.py --check metadata
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter

from common import (
    data_dir,
    load_train_csv,
    login_to_hf,
    normalize_text,
    prepared_dir,
    quiet_hf_logging,
    write_json,
)

logger = logging.getLogger("verify_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", choices=["corruption", "metadata", "all"], default="all")
    p.add_argument("--data-dir", default=str(prepared_dir()))
    p.add_argument("--train-csv", default=str(data_dir() / "Train.csv"))
    p.add_argument("--dataset-id", default="google/WaxalNLP")
    p.add_argument("--languages", nargs="+", default=["lin", "sna", "lug"])
    p.add_argument("--split", default="train", help="train|validation ONLY — never test.")
    p.add_argument("--short-threshold", type=float, default=1.5)
    p.add_argument("--n-samples", type=int, default=5,
                   help="How many suspect clips per language to re-fetch raw from the Hub.")
    p.add_argument("--metadata-sample", type=int, default=200,
                   help="Clips per language to sample for speaker stats (if a speaker field exists).")
    p.add_argument("--output-dir", default=None, help="Default: <data-dir>/logs")
    p.add_argument("--hf-token", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _raw_audio_info(audio):
    """Returns (array, sampling_rate) with NO resampling — the audio exactly as the
    Hub serves it, so we can tell corpus truncation from our own processing."""
    import numpy as np
    from collections.abc import Mapping

    if isinstance(audio, Mapping):
        return np.asarray(audio["array"], dtype=np.float32), int(audio.get("sampling_rate", 0))
    if hasattr(audio, "get_all_samples"):
        s = audio.get_all_samples()
        arr = np.asarray(s.data, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        return arr.flatten(), int(getattr(s, "sample_rate", 0))
    return np.asarray(audio, dtype=np.float32), 0


def _rms(arr) -> float:
    import numpy as np
    return float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0


def _find_suspect_ids(data_dir: str, split: str, languages: list[str], thresh: float):
    """Suspect clips from OUR prepared raw data: duration < thresh."""
    import datasets

    out: dict[str, list[dict]] = {}
    for lang in languages:
        p = os.path.join(data_dir, split, lang)
        if not os.path.isdir(p):
            continue
        ds = datasets.load_from_disk(p)
        ids, durs, texts = ds["id"], ds["duration"], ds["transcription"]
        rows = [{"id": i, "our_duration_s": float(d), "n_words": len(t.split()), "text": t}
                for i, d, t in zip(ids, durs, texts) if float(d) < thresh]
        out[lang] = rows
    return out


# ---------------------------------------------------------------------------
# CHECK 1 — corruption
# ---------------------------------------------------------------------------

def check_corruption(args, train_csv, out_dir) -> dict:
    import datasets

    logger.info("=" * 70)
    logger.info("CHECK 1 — is the 1.01 s truncation in the CORPUS, or our decode bug?")
    logger.info("=" * 70)

    suspects = _find_suspect_ids(args.data_dir, args.split, args.languages, args.short_threshold)
    id_to_text = dict(zip(train_csv["id"], train_csv["transcription"]))

    report: dict = {"split": args.split, "short_threshold_s": args.short_threshold, "per_language": {}}

    for lang, rows in suspects.items():
        n = len(rows)
        durs = Counter(round(r["our_duration_s"], 3) for r in rows)
        logger.info("\n--- %s: %d clips < %.1f s in our prepared data", lang, n, args.short_threshold)
        if n == 0:
            logger.info("    (none — nothing to verify)")
            report["per_language"][lang] = {"n_suspect": 0}
            continue
        logger.info("    distinct durations among them: %s",
                    dict(sorted(durs.items())[:6]) if len(durs) <= 6 else f"{len(durs)} distinct")

        want = {r["id"] for r in rows[: args.n_samples * 40]}
        found: list[dict] = []
        # Stream WITHOUT cast_column -> native sample rate, no resampling by us.
        stream = datasets.load_dataset(args.dataset_id, name=f"{lang}_asr",
                                       split=args.split, streaming=True)
        for ex in stream:
            _id = str(ex.get("id", ""))
            if _id not in want:
                continue
            arr, sr = _raw_audio_info(ex["audio"])
            hub_dur = len(arr) / sr if sr else 0.0
            ours = next(r for r in rows if r["id"] == _id)
            hub_text = next((str(ex[k]) for k in ("transcription", "text", "sentence")
                             if k in ex and ex[k]), "")
            found.append({
                "id": _id, "hub_native_sr": sr, "hub_n_samples": int(arr.size),
                "hub_duration_s": round(hub_dur, 3), "our_duration_s": round(ours["our_duration_s"], 3),
                "rms_energy": round(_rms(arr), 5), "n_words_train_csv": len(normalize_text(id_to_text.get(_id, "")).split()),
                "hub_text_matches_train_csv": normalize_text(hub_text) == normalize_text(id_to_text.get(_id, "")),
                "text_preview": normalize_text(id_to_text.get(_id, ""))[:90],
            })
            if len(found) >= args.n_samples:
                break

        for f in found:
            logger.info("    %s | Hub raw: %d samples @ %d Hz = %.3fs | ours: %.3fs | RMS %.4f | "
                        "%d words | hub_text==Train.csv: %s",
                        f["id"], f["hub_n_samples"], f["hub_native_sr"], f["hub_duration_s"],
                        f["our_duration_s"], f["rms_energy"], f["n_words_train_csv"],
                        f["hub_text_matches_train_csv"])
            logger.info("        text: %r", f["text_preview"])

        verdict = _verdict(found)
        logger.info("    ==> VERDICT: %s", verdict)
        report["per_language"][lang] = {
            "n_suspect": n, "distinct_durations": len(durs),
            "samples_checked": found, "verdict": verdict,
        }

    write_json(os.path.join(out_dir, "verify_corruption.json"), report)
    logger.info("\nSaved -> %s", os.path.join(out_dir, "verify_corruption.json"))
    return report


def _verdict(found: list[dict]) -> str:
    if not found:
        return "INCONCLUSIVE — could not re-fetch any suspect clip from the Hub"
    hub_short = sum(1 for f in found if f["hub_duration_s"] < 2.0)
    silent = sum(1 for f in found if f["rms_energy"] < 1e-4)
    if hub_short == len(found):
        base = ("CORPUS CORRUPTION — the Hub itself serves ~1 s of audio for a full-length "
                "transcript; our decode is faithful. Safe to drop these clips.")
        if silent == len(found):
            return base + " NOTE: the 1 s is essentially SILENT (audio lost entirely)."
        return base + " The 1 s contains real speech (audio truncated mid-utterance)."
    if hub_short == 0:
        return ("OUR BUG — the Hub serves full-length audio but our pipeline stored ~1 s. "
                "DO NOT drop these clips; fix prepare_data.py first.")
    return f"MIXED — {hub_short}/{len(found)} short on the Hub. Investigate before dropping."


# ---------------------------------------------------------------------------
# CHECK 2 — speaker / session metadata
# ---------------------------------------------------------------------------

SPEAKER_HINTS = ("speaker", "spk", "session", "recording", "gender", "age", "client", "user")


def check_metadata(args, out_dir) -> dict:
    import datasets

    logger.info("=" * 70)
    logger.info("CHECK 2 — does WaxalNLP expose speaker/session metadata?")
    logger.info("   (Train.csv has none; a speaker-disjoint val split needs it)")
    logger.info("=" * 70)

    report: dict = {"per_language": {}}
    for lang in args.languages:
        stream = datasets.load_dataset(args.dataset_id, name=f"{lang}_asr",
                                       split=args.split, streaming=True)
        first = next(iter(stream))
        fields = sorted(first.keys())
        features = {k: str(v) for k, v in (stream.features or {}).items()} if stream.features else {}
        candidates = [f for f in fields if any(h in f.lower() for h in SPEAKER_HINTS)]

        logger.info("\n--- %s", lang)
        logger.info("    fields: %s", fields)
        logger.info("    speaker-ish candidates: %s", candidates or "NONE")
        preview = {k: (str(v)[:60] if k != "audio" else "<audio>") for k, v in first.items()}
        logger.info("    example: %s", preview)

        stats = {}
        if candidates:
            # Cheap sample to gauge speaker diversity (avoids a full re-download).
            vals: dict[str, Counter] = {c: Counter() for c in candidates}
            for i, ex in enumerate(stream):
                if i >= args.metadata_sample:
                    break
                for c in candidates:
                    vals[c][str(ex.get(c))] += 1
            for c in candidates:
                stats[c] = {"distinct_in_sample": len(vals[c]),
                            "sample_size": min(args.metadata_sample, sum(vals[c].values())),
                            "most_common": vals[c].most_common(5)}
                logger.info("    %s: %d distinct in %d sampled clips | top: %s",
                            c, stats[c]["distinct_in_sample"], stats[c]["sample_size"],
                            stats[c]["most_common"])
        report["per_language"][lang] = {"fields": fields, "features": features,
                                        "speaker_candidates": candidates, "stats": stats}

    any_speaker = any(r["speaker_candidates"] for r in report["per_language"].values())
    report["verdict"] = (
        "SPEAKER METADATA AVAILABLE — we can build a speaker-disjoint validation split "
        "(re-run prepare_data.py saving the field, or join by id)."
        if any_speaker else
        "NO SPEAKER METADATA — a speaker-disjoint split is not possible from this dataset; "
        "fall back to the provided train/validation split and treat val as optimistic."
    )
    logger.info("\n==> VERDICT: %s", report["verdict"])
    write_json(os.path.join(out_dir, "verify_metadata.json"), report)
    logger.info("Saved -> %s", os.path.join(out_dir, "verify_metadata.json"))
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()
    assert args.split != "test", "This script never touches the test split."
    out_dir = args.output_dir or os.path.join(args.data_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    login_to_hf(args.hf_token)

    if args.check in ("corruption", "all"):
        check_corruption(args, load_train_csv(args.train_csv), out_dir)
    if args.check in ("metadata", "all"):
        check_metadata(args, out_dir)

    logging.shutdown()
    os._exit(0)   # HF streaming leaves connections that can hang on teardown


if __name__ == "__main__":
    main()
