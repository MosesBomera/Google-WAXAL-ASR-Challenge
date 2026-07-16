"""experiment mms-ctc / Stage 1c — freeze the training set.

Turns the raw download + EDA findings into the exact, inspectable set Stage 2
trains on. Runs **entirely offline from the local metadata CSV** — no GPU, no
audio, no re-download — so the split can be built, reviewed and committed before
a box is ever spun up.

Four outputs (see plan-mms-ctc.md §6):
  1. manifests/{train,val}.csv  — the frozen SPEAKER-DISJOINT split
  2. vocab/vocab.json           — cased+punctuated CTC vocab (+ normalization map)
  3. lm/corpus.txt              — KenLM text (Stage 3), val speakers excluded
  4. clean_split_report.json    — per-language before/after + split audit

Why speaker-disjoint (report-mms-ctc-eda.md, Finding 7): the PROVIDED split is
~100% speaker-leaked (lin 99.95%, sna 99.77%, lug 99.25%), so it scores memorised
voices while Phase 2 is unseen speakers. We ignore `split` from the corpus and
build our own by `speaker_id`. Cleaning runs FIRST — otherwise we'd hold out
speakers whose clips all vanish (15 lin / 9 sna speakers are ~100% corrupt).

    python clean_split.py                    # defaults are the EDA-chosen values
    python clean_split.py --val-frac 0.12 --seed 7
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
    p.add_argument("--out-dir", default=str(workdir()),
                   help="Writes manifests/, vocab/, lm/ here (the experiment root — the flat "
                        "folder on vast.ai, or experiments/mms-ctc/ in the repo).")
    # cleaning — values chosen from OUR EDA, not imported (see report §'Recommended cleaning config')
    p.add_argument("--min-duration-sec", type=float, default=1.5,
                   help="Drops the verified 1.008/1.045 s corpus corruption.")
    p.add_argument("--max-words-per-sec", type=float, default=4.0,
                   help="Belt-and-braces on the same corruption.")
    p.add_argument("--max-duration-sec", type=float, default=None,
                   help="Default None = keep everything (longest real clip is 46.04 s; CTC has no cap).")
    # split
    p.add_argument("--val-frac", type=float, default=0.10, help="Target share of clips held out per language.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-speakers-per-lang", type=int, default=15,
                   help="Desired number of held-out voices per language. Drives which SIZE of "
                        "speaker we pick, balancing a stable val estimate against keeping train "
                        "speaker-diverse (Lingala has only ~91 voices to begin with). Measured "
                        "trade-off at a fixed ~10%% clip budget — lin train speakers: "
                        "10->74, 15->70, 25->42, 40->34. Default 15 keeps 77%% of Lingala's "
                        "voices in train while still holding out 15 for the estimate.")
    p.add_argument("--val-max-speaker-frac", type=float, default=0.05,
                   help="A val speaker may not exceed this share of the language's clips — keeps "
                        "Lingala's 1,737-clip speaker (10.7%) out of val and in train.")
    p.add_argument("--val-max-speaker-share-of-val", type=float, default=0.15,
                   help="No single voice may exceed this share of the VAL set (the language-relative "
                        "cap is too loose to protect a small val). Only Lingala binds here, and it "
                        "trades against gender: 0.10 caps the top voice at 9%% but skews val to "
                        "33F/67M (population 54/46); 0.15 allows 14.8%% and gets 59F/41M. We take "
                        "0.15 — a systematic gender bias is worse than one noisy speaker.")
    p.add_argument("--min-gender-share", type=float, default=0.05,
                   help="Gender buckets below this share of a language are excluded from val and "
                        "kept in train. Speaker-level picks can't hit a tiny budget: Luganda's "
                        "'unknown' gender is 0.9%% of clips, but one 55-clip speaker overshot it "
                        "to 8.4%% of val.")
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
# 2. speaker-disjoint split
# ---------------------------------------------------------------------------

def speaker_disjoint_split(kept: list[dict], args) -> tuple[set, dict]:
    """Chooses VAL speakers per language, balancing two things that pull apart:

      * val needs ENOUGH voices to be a stable unseen-speaker estimate;
      * train needs to KEEP its voices — speaker diversity is what drives
        generalisation, and Lingala only has ~85 to start with (Finding 7b).

    Naive smallest-first maximises val voices but guts train (it left Lingala with
    34 of 85). Largest-first risks one speaker owning val. So we aim for a target
    NUMBER of val speakers and pick those closest to the size that hits the clip
    budget with exactly that many — i.e. median-ish speakers — while excluding any
    speaker too large to sit in val at all (--val-max-speaker-frac).

    GENDER: picking purely by size let gender land lopsided (lin val was 80% male,
    sna 78% female) — with only ~15 val voices per language that happens by chance,
    and it biases the estimate toward one voice type. So we fill each gender bucket
    in proportion to that language's own clip-level gender mix, making val a
    representative sample rather than a coin flip.
    """
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
        # Two caps, take the tighter: keep mega-speakers in train (language-relative)
        # AND stop any one voice dominating the estimate (val-relative).
        max_spk_clips = max(1, min(int(len(rows) * args.val_max_speaker_frac),
                                   int(target * args.val_max_speaker_share_of_val)))
        ideal = max(1.0, target / max(1, args.val_speakers_per_lang))

        # Each speaker gets one gender (majority over their clips — the corpus has
        # some per-clip label noise).
        spk_gender = {s: Counter(c["gender"] for c in cl).most_common(1)[0][0]
                      for s, cl in per_spk.items()}
        # Per-gender val clip budgets, proportional to THIS language's mix. Genders
        # too rare to hit with whole speakers are excluded from val (kept in train).
        lang_gender = Counter(c["gender"] for c in rows)
        gender_target = {g: target * n / len(rows) for g, n in lang_gender.items()
                         if n / len(rows) >= args.min_gender_share}

        eligible = [s for s in per_spk
                    if len(per_spk[s]) <= max_spk_clips and spk_gender[s] in gender_target]
        # Prefer speakers nearest the size that hits `target` with ~`val_speakers_per_lang` of them.
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
            if not avail or len(per_spk) - len(chosen) <= 1:   # never strand train with 0 speakers
                break
            # Take from whichever gender is furthest below its share of the val budget.
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
            "largest_val_speaker_share_of_val_pct":
                round(100 * biggest / max(1, len(val_rows)), 2),
            "val_gender": dict(Counter(c["gender"] for c in val_rows)),
            # Does val's gender mix match the language's? (the whole point of the
            # gender-proportional fill — a lopsided val biases the estimate)
            "val_gender_pct": {g: round(100 * n / max(1, len(val_rows)), 1)
                               for g, n in sorted(Counter(c["gender"] for c in val_rows).items())},
            "language_gender_pct": {g: round(100 * n / max(1, len(rows)), 1)
                                    for g, n in sorted(Counter(c["gender"] for c in rows).items())},
        }
    return val_speakers, audit


# ---------------------------------------------------------------------------
# 3. vocab + 4. LM corpus
# ---------------------------------------------------------------------------

def build_vocab(train_rows: list[dict]) -> tuple[dict, dict]:
    counts: Counter = Counter()
    per_lang: dict[str, set] = defaultdict(set)
    for c in train_rows:
        counts.update(c["transcription"])
        per_lang[c["language"]].update(c["transcription"])

    chars = sorted(ch for ch in counts if ch != " ")
    vocab = {ch: i for i, ch in enumerate(chars)}
    vocab["|"] = len(vocab)        # word delimiter (space)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)    # doubles as the CTC blank

    rare = sorted(((ch, n) for ch, n in counts.items() if n < 20 and ch != " "),
                  key=lambda x: x[1])
    report = {
        "vocab_size": len(vocab), "n_chars_from_data": len(chars),
        "per_language_distinct": {l: len(s) for l, s in sorted(per_lang.items())},
        "has_uppercase": any(ch.isupper() for ch in counts),
        "punctuation": sorted(ch for ch in counts if not ch.isalnum() and not ch.isspace()),
        "rare_under_20": [{"char": ch, "codepoint": f"U+{ord(ch):04X}", "count": n}
                          for ch, n in rare],
    }
    return vocab, report


def build_lm_corpus(all_clips: list[dict], val_speakers: set) -> list[str]:
    """KenLM text = every valid transcript whose speaker is NOT held out.

    Includes the corrupt-audio clips' text: their transcripts are perfectly good
    sentences (only the audio is truncated), so ~1,400 free sentences we'd
    otherwise delete. Excluding val speakers keeps validation honest — an LM that
    had memorised val sentences would flatter the Stage-3 numbers.
    """
    lines = []
    for c in all_clips:
        if c["speaker_id"] in val_speakers:
            continue
        t = normalize_for_ctc(c["transcription"])
        if t:
            lines.append(t)
    return lines


# ---------------------------------------------------------------------------

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

    # --- clean ---
    kept, drops = clean(clips, args)
    logger.info("===== CLEANING (thresholds from our EDA) =====")
    before = Counter(c["language"] for c in clips)
    after = Counter(c["language"] for c in kept)
    hours_before = defaultdict(float)
    hours_after = defaultdict(float)
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
        logger.info("  %s: %d -> %d clips (-%.2f%%) | %.1f -> %.1f h (-%.2f%% of audio) | %s",
                    lang, r["clips_before"], r["clips_after"], r["clip_drop_pct"],
                    r["hours_before"], r["hours_after"], r["hours_lost_pct"], d)

    # --- speaker-disjoint split ---
    val_speakers, split_audit = speaker_disjoint_split(kept, args)
    for c in kept:
        c["split"] = "val" if c["speaker_id"] in val_speakers else "train"

    logger.info("===== SPEAKER-DISJOINT SPLIT (ours — the provided one is ~100%% leaked) =====")
    for lang, a in split_audit.items():
        logger.info("  %s: train %d clips/%d spk (%.1f h) | val %d clips/%d spk (%.1f h, %.1f%%)",
                    lang, a["train_clips"], a["train_speakers"], a["train_hours"],
                    a["val_clips"], a["val_speakers"], a["val_hours"], a["val_clip_pct"])
        logger.info("     largest val speaker = %.1f%% of val | gender val %s vs language %s",
                    a["largest_val_speaker_share_of_val_pct"],
                    a["val_gender_pct"], a["language_gender_pct"])

    # --- audit: the split MUST be speaker-disjoint ---
    tr_spk = {c["speaker_id"] for c in kept if c["split"] == "train"}
    va_spk = {c["speaker_id"] for c in kept if c["split"] == "val"}
    overlap = tr_spk & va_spk
    assert not overlap, f"speaker leakage: {sorted(overlap)[:5]}"
    logger.info("  ✓ verified 0 speaker overlap (train %d spk, val %d spk)", len(tr_spk), len(va_spk))

    # --- vocab (from TRAIN only) ---
    train_rows = [c for c in kept if c["split"] == "train"]
    vocab, vocab_report = build_vocab(train_rows)
    write_json(os.path.join(out, "vocab", "vocab.json"), vocab)
    logger.info("===== VOCAB ===== size %d (%d data chars + | [UNK] [PAD]) | uppercase=%s",
                vocab_report["vocab_size"], vocab_report["n_chars_from_data"],
                vocab_report["has_uppercase"])
    logger.info("  punctuation: %s", "".join(vocab_report["punctuation"]))
    if vocab_report["rare_under_20"]:
        logger.info("  %d rare chars remain (<20x) after normalization: %s",
                    len(vocab_report["rare_under_20"]),
                    [r["char"] for r in vocab_report["rare_under_20"]])

    # --- LM corpus ---
    lm_lines = build_lm_corpus(clips, val_speakers)
    lm_path = os.path.join(out, "lm", "corpus.txt")
    os.makedirs(os.path.dirname(lm_path), exist_ok=True)
    with open(lm_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lm_lines) + "\n")
    n_words = sum(len(l.split()) for l in lm_lines)
    logger.info("===== LM CORPUS ===== %d lines, %d words -> %s (val speakers excluded)",
                len(lm_lines), n_words, lm_path)

    # --- manifests ---
    cols = ["id", "language", "split", "speaker_id", "gender", "duration_s", "n_words", "wps"]
    df = pd.DataFrame(kept)[cols].sort_values(["language", "split", "id"])
    os.makedirs(os.path.join(out, "manifests"), exist_ok=True)
    for name in ("train", "val"):
        p = os.path.join(out, "manifests", f"{name}.csv")
        sub = df[df["split"] == name]
        sub.to_csv(p, index=False)
        logger.info("Wrote %s (%d clips)", p, len(sub))

    report = {"config": vars(args), "cleaning": clean_report, "split": split_audit,
              "split_is_speaker_disjoint": True,
              "vocab": vocab_report,
              "lm_corpus": {"lines": len(lm_lines), "words": n_words, "path": lm_path},
              "totals": {"clips_before": len(clips), "clips_after": len(kept),
                         "train_clips": len(train_rows), "val_clips": len(kept) - len(train_rows),
                         "clips_per_speaker_after": percentiles(
                             [float(v) for v in Counter(c["speaker_id"] for c in kept).values()])}}
    write_json(os.path.join(out, "clean_split_report.json"), report)
    logger.info("Report -> %s", os.path.join(out, "clean_split_report.json"))
    logger.info("Stage 1c done. Next: Stage 2 — MMS-1B CTC fine-tune on manifests/train.csv.")


if __name__ == "__main__":
    main()
