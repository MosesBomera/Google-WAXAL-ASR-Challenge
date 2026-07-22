"""mms-ctc-v2 / Stage 1c — freeze the WAXAL training set (manifests only).

Turns the raw download + EDA findings into the exact, inspectable split Stage 2 trains
on. Runs ENTIRELY OFFLINE from the local metadata CSV — no GPU, no audio, no re-download.

Two outputs:
  1. manifests/{train,val}.csv  — the frozen SPEAKER-DISJOINT split
  2. clean_split_report.json    — per-language before/after + split audit

Differences from v1 mms-ctc:
  * VOCAB is no longer built here — it's built by build_vocab.py AFTER prepare_external.py
    AND prepare_lrsc.py, so it covers WAXAL **+ FLEURS + LRSC**. (Vocab is deferred; this
    stage is WAXAL-only.)
  * The KenLM corpus is gone (KenLM was proven a dead end in v1).
  * NEW guardrail: the val id-set is verified byte-identical to v1's, so v2's val stays a
    fair comparison to the 0.205 baseline (clean_split is deterministic at seed 42).

    python clean_split.py
    python clean_split.py --no-verify-val        # skip the v1 comparability check
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from collections import Counter, defaultdict

from common import (
    data_dir,
    load_train_csv,
    normalize_for_ctc,
    percentiles,
    quiet_hf_logging,
    setup_file_logging,
    workdir,
    write_json,
)

logger = logging.getLogger("clean_split")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", default=str(data_dir() / "waxal_clip_metadata.csv"))
    p.add_argument("--train-csv", default=str(data_dir() / "Train.csv"),
                   help="Authoritative transcripts (the metadata CSV carries no text).")
    p.add_argument("--out-dir", default=str(workdir()))
    # cleaning — values chosen from OUR EDA (report-mms-ctc-eda.md)
    p.add_argument("--min-duration-sec", type=float, default=1.5,
                   help="Drops the verified 1.008/1.045 s corpus corruption.")
    p.add_argument("--max-words-per-sec", type=float, default=4.0)
    p.add_argument("--max-duration-sec", type=float, default=None,
                   help="Default None = keep everything (longest real clip 46.04 s; CTC has no cap).")
    # split
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-speakers-per-lang", type=int, default=15)
    p.add_argument("--val-max-speaker-frac", type=float, default=0.05)
    p.add_argument("--val-max-speaker-share-of-val", type=float, default=0.15)
    p.add_argument("--min-gender-share", type=float, default=0.05)
    # v2 guardrail
    p.add_argument("--verify-val-against",
                   default=str(workdir().parent / "mms-ctc" / "manifests" / "val.csv"),
                   help="v1 val.csv to compare against for comparability (empty to disable).")
    p.add_argument("--verify-val", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-val-drift", action="store_true",
                   help="Downgrade a val-id mismatch from error to warning.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. load + clean
# ---------------------------------------------------------------------------

def load_clips(metadata_path: str, train_csv_path: str) -> list[dict]:
    import pandas as pd

    meta = pd.read_csv(metadata_path, dtype={"speaker_id": str, "id": str})
    text = load_train_csv(train_csv_path)[["id", "transcription"]]
    df = meta.merge(text, on="id", how="left")
    missing = int(df["transcription"].isna().sum())
    if missing:
        logger.warning("%d clips had no transcript in Train.csv", missing)
    df["transcription"] = df["transcription"].fillna("")
    return df.to_dict("records")


def clean(clips: list[dict], args) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    drops: dict = defaultdict(lambda: Counter())
    for c in clips:
        lang = c["language"]
        text = normalize_for_ctc(c["transcription"])
        dur = float(c["duration_s"])
        n_words = len(text.split())
        wps = n_words / dur if dur > 0 else 0.0

        if n_words == 0:
            drops[lang]["empty"] += 1
            continue
        if dur < args.min_duration_sec:
            drops[lang]["corrupt_short"] += 1
            continue
        if wps > args.max_words_per_sec:
            drops[lang]["corrupt_fast"] += 1
            continue
        if args.max_duration_sec is not None and dur > args.max_duration_sec:
            drops[lang]["too_long"] += 1
            continue
        kept.append({**c, "transcription": text, "n_words": n_words,
                     "wps": round(wps, 3), "duration_s": dur})
    return kept, {l: dict(v) for l, v in drops.items()}


# ---------------------------------------------------------------------------
# 2. speaker-disjoint split (gender-proportional; unchanged from v1)
# ---------------------------------------------------------------------------

def speaker_disjoint_split(kept: list[dict], args) -> tuple[set, dict]:
    rng = random.Random(args.seed)
    val_speakers: set = set()
    audit: dict = {}

    by_lang: dict[str, list[dict]] = defaultdict(list)
    for c in kept:
        by_lang[c["language"]].append(c)

    for lang, rows in sorted(by_lang.items()):
        per_spk: dict[str, list[dict]] = defaultdict(list)
        for c in rows:
            per_spk[c["speaker_id"]].append(c)
        target = int(len(rows) * args.val_frac)
        max_spk_clips = max(1, min(int(len(rows) * args.val_max_speaker_frac),
                                   int(target * args.val_max_speaker_share_of_val)))
        ideal = max(1.0, target / max(1, args.val_speakers_per_lang))

        spk_gender = {s: Counter(c["gender"] for c in cl).most_common(1)[0][0]
                      for s, cl in per_spk.items()}
        lang_gender = Counter(c["gender"] for c in rows)
        gender_target = {g: target * n / len(rows) for g, n in lang_gender.items()
                         if n / len(rows) >= args.min_gender_share}

        eligible = [s for s in per_spk
                    if len(per_spk[s]) <= max_spk_clips and spk_gender[s] in gender_target]
        buckets: dict[str, list[str]] = defaultdict(list)
        for s in eligible:
            buckets[spk_gender[s]].append(s)
        for g in buckets:
            buckets[g].sort(key=lambda s: (abs(len(per_spk[s]) - ideal), rng.random()))

        chosen: list[str] = []
        n_clips = 0
        got: Counter = Counter()
        while n_clips < target:
            avail = [g for g in buckets if buckets[g]]
            if not avail or len(per_spk) - len(chosen) <= 1:
                break
            g = min(avail, key=lambda g: (got[g] / gender_target[g])
                    if gender_target.get(g, 0) > 0 else float("inf"))
            spk = buckets[g].pop(0)
            chosen.append(spk)
            n_clips += len(per_spk[spk])
            got[g] += len(per_spk[spk])

        val_speakers.update(chosen)
        val_rows = [c for s in chosen for c in per_spk[s]]
        biggest = max((len(per_spk[s]) for s in chosen), default=0)
        audit[lang] = {
            "total_clips": len(rows), "total_speakers": len(per_spk),
            "val_speakers": len(chosen), "val_clips": len(val_rows),
            "val_clip_pct": round(100 * len(val_rows) / max(1, len(rows)), 2),
            "train_speakers": len(per_spk) - len(chosen),
            "train_clips": len(rows) - len(val_rows),
            "val_hours": round(sum(c["duration_s"] for c in val_rows) / 3600, 2),
            "train_hours": round(sum(c["duration_s"] for c in rows) / 3600
                                 - sum(c["duration_s"] for c in val_rows) / 3600, 2),
            "largest_val_speaker_share_of_val_pct": round(100 * biggest / max(1, len(val_rows)), 2),
            "val_gender_pct": {g: round(100 * n / max(1, len(val_rows)), 1)
                               for g, n in sorted(Counter(c["gender"] for c in val_rows).items())},
            "language_gender_pct": {g: round(100 * n / max(1, len(rows)), 1)
                                    for g, n in sorted(Counter(c["gender"] for c in rows).items())},
        }
    return val_speakers, audit


# ---------------------------------------------------------------------------

def verify_val_against_v1(val_ids: set, ref_path: str, allow_drift: bool) -> None:
    """The val set is our yardstick vs the 0.205 baseline — it MUST be byte-identical to v1's.
    clean_split is deterministic (seed 42), so any drift means a threshold/metadata changed."""
    import pandas as pd

    if not ref_path or not os.path.isfile(ref_path):
        logger.warning("v1 val reference not found (%s) — skipping comparability check. "
                       "Confirm the val set matches v1 before trusting the 0.205 delta.", ref_path)
        return
    ref_ids = set(pd.read_csv(ref_path, dtype={"id": str})["id"])
    if val_ids == ref_ids:
        logger.info("  ✓ val id-set matches v1 exactly (%d clips) — yardstick preserved.", len(val_ids))
        return
    added, removed = val_ids - ref_ids, ref_ids - val_ids
    msg = (f"VAL DRIFT vs v1: +{len(added)} / -{len(removed)} clips "
           f"(new {len(val_ids)} vs v1 {len(ref_ids)}). e.g. +{sorted(added)[:3]} -{sorted(removed)[:3]}. "
           "The 0.205 baseline is no longer comparable.")
    if allow_drift:
        logger.warning(msg + "  (--allow-val-drift set; continuing.)")
    else:
        raise SystemExit(msg + "  Re-check thresholds/metadata, or pass --allow-val-drift.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()
    out = args.out_dir
    setup_file_logging(os.path.join(out, "logs"), name="clean_split")
    import pandas as pd

    logger.info("Loading %s ...", args.metadata)
    clips = load_clips(args.metadata, args.train_csv)
    logger.info("Loaded %d clips (offline — no audio, no GPU).", len(clips))

    kept, drops = clean(clips, args)
    logger.info("===== CLEANING =====")
    before = Counter(c["language"] for c in clips)
    after = Counter(c["language"] for c in kept)
    hours_before, hours_after = defaultdict(float), defaultdict(float)
    for c in clips:
        hours_before[c["language"]] += float(c["duration_s"]) / 3600
    for c in kept:
        hours_after[c["language"]] += c["duration_s"] / 3600
    clean_report = {}
    for lang in sorted(before):
        d = drops.get(lang, {})
        clean_report[lang] = {
            "clips_before": before[lang], "clips_after": after[lang],
            "clips_dropped": before[lang] - after[lang],
            "clip_drop_pct": round(100 * (before[lang] - after[lang]) / before[lang], 2),
            "hours_before": round(hours_before[lang], 2), "hours_after": round(hours_after[lang], 2),
            "hours_lost_pct": round(100 * (hours_before[lang] - hours_after[lang]) / max(1e-9, hours_before[lang]), 2),
            "drops": d,
        }
        r = clean_report[lang]
        logger.info("  %s: %d -> %d clips (-%.2f%%) | %.1f -> %.1f h (-%.2f%% audio) | %s",
                    lang, r["clips_before"], r["clips_after"], r["clip_drop_pct"],
                    r["hours_before"], r["hours_after"], r["hours_lost_pct"], d)

    val_speakers, split_audit = speaker_disjoint_split(kept, args)
    for c in kept:
        c["split"] = "val" if c["speaker_id"] in val_speakers else "train"

    logger.info("===== SPEAKER-DISJOINT SPLIT =====")
    for lang, a in split_audit.items():
        logger.info("  %s: train %d clips/%d spk (%.1f h) | val %d clips/%d spk (%.1f h, %.1f%%)",
                    lang, a["train_clips"], a["train_speakers"], a["train_hours"],
                    a["val_clips"], a["val_speakers"], a["val_hours"], a["val_clip_pct"])

    tr_spk = {c["speaker_id"] for c in kept if c["split"] == "train"}
    va_spk = {c["speaker_id"] for c in kept if c["split"] == "val"}
    overlap = tr_spk & va_spk
    assert not overlap, f"speaker leakage: {sorted(overlap)[:5]}"
    logger.info("  ✓ verified 0 speaker overlap (train %d spk, val %d spk)", len(tr_spk), len(va_spk))

    # manifests
    cols = ["id", "language", "split", "speaker_id", "gender", "duration_s", "n_words", "wps"]
    df = pd.DataFrame(kept)[cols].sort_values(["language", "split", "id"])
    os.makedirs(os.path.join(out, "manifests"), exist_ok=True)
    val_ids: set = set()
    for name in ("train", "val"):
        p = os.path.join(out, "manifests", f"{name}.csv")
        sub = df[df["split"] == name]
        sub.to_csv(p, index=False)
        if name == "val":
            val_ids = set(sub["id"])
        logger.info("Wrote %s (%d clips)", p, len(sub))

    if args.verify_val:
        verify_val_against_v1(val_ids, args.verify_val_against, args.allow_val_drift)

    report = {"config": vars(args), "cleaning": clean_report, "split": split_audit,
              "split_is_speaker_disjoint": True,
              "totals": {"clips_before": len(clips), "clips_after": len(kept),
                         "train_clips": int((df["split"] == "train").sum()),
                         "val_clips": int((df["split"] == "val").sum()),
                         "clips_per_speaker_after": percentiles(
                             [float(v) for v in Counter(c["speaker_id"] for c in kept).values()])}}
    write_json(os.path.join(out, "clean_split_report.json"), report)
    logger.info("Report -> %s", os.path.join(out, "clean_split_report.json"))
    logger.info("Stage 1c done. Next: prepare_external.py (FLEURS) -> prepare_lrsc.py (LRSC) "
                "-> build_vocab.py -> train_ctc.py.")


if __name__ == "__main__":
    main()
