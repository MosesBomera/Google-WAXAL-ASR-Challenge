"""mms-ctc / Stage 1b — proper EDA on the RAW downloaded WAXAL data.

Answers the questions that should decide our cleaning thresholds, instead of
importing another paper's numbers on faith:

  1. What do the duration / words-per-second / transcript-length distributions
     actually look like, PER LANGUAGE?
  2. How many clips would each candidate threshold remove? (a sweep, so we see
     the cost before committing)
  3. Are the extremes actually corrupt? — we print the shortest / fastest /
     longest / slowest clips WITH their text so we can eyeball them.
  4. What's the character inventory (for the cased CTC vocab), and are there
     junk/rare characters?
  5. Are there duplicate or suspiciously repeated transcripts?

Reads only the id/transcription/language/duration columns — audio is never
decoded, so this is fast and cheap to re-run.

    python eda.py                       # reads ./prepared/waxal-raw
    python eda.py --split train --top-k 15
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter, defaultdict

from common import percentiles, prepared_dir, quiet_hf_logging, write_json

logger = logging.getLogger("eda")

# Candidate thresholds to sweep (chosen to bracket the paper's values so we can
# see whether they transfer to OUR languages).
MIN_DUR_CANDIDATES = [0.5, 1.0, 1.5, 2.0]
MAX_DUR_CANDIDATES = [20.0, 30.0, 40.0, 60.0]
MAX_WPS_CANDIDATES = [3.0, 4.0, 5.0, 6.0]
LOW_WPS_CANDIDATES = [0.2, 0.3, 0.5]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(prepared_dir()))
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--output-dir", default=None, help="Default: <data-dir>/logs")
    p.add_argument("--top-k", type=int, default=10, help="How many extreme examples to show.")
    return p.parse_args()


def load_meta(data_dir: str, splits: list[str]) -> list[dict]:
    """Loads id/transcription/language/duration WITHOUT decoding audio."""
    import datasets

    rows: list[dict] = []
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            logger.warning("No such split dir: %s", split_dir)
            continue
        for name in sorted(os.listdir(split_dir)):
            p = os.path.join(split_dir, name)
            if not os.path.isdir(p):
                continue
            ds = datasets.load_from_disk(p)
            cols = {c: ds[c] for c in ("id", "transcription", "language", "duration") if c in ds.column_names}
            n = len(ds)
            for i in range(n):
                text = cols["transcription"][i]
                dur = float(cols["duration"][i]) if "duration" in cols else 0.0
                nw = len(text.split())
                rows.append({
                    "id": cols["id"][i], "split": split,
                    "language": cols["language"][i] if "language" in cols else name,
                    "text": text, "duration": dur, "n_words": nw,
                    "n_chars": len(text),
                    "wps": nw / dur if dur > 0 else 0.0,
                    "cps": len(text) / dur if dur > 0 else 0.0,
                })
    return rows


def _hist(values: list[float], edges: list[float]) -> dict:
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        out[f"{lo:g}-{hi:g}"] = sum(1 for v in values if lo <= v < hi)
    out[f">={edges[-1]:g}"] = sum(1 for v in values if v >= edges[-1])
    return out


def analyse(rows: list[dict], top_k: int) -> dict:
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r)

    report: dict = {"total_clips": len(rows), "per_language": {}, "threshold_sweep": {},
                    "extremes": {}, "characters": {}, "duplicates": {}}

    # ---- distributions ----
    for lang, rs in sorted(by_lang.items()):
        durs = [r["duration"] for r in rs]
        report["per_language"][lang] = {
            "n_clips": len(rs),
            "hours": round(sum(durs) / 3600, 2),
            "duration_s": percentiles(durs),
            "words_per_sec": percentiles([r["wps"] for r in rs]),
            "chars_per_sec": percentiles([r["cps"] for r in rs]),
            "n_words": percentiles([float(r["n_words"]) for r in rs]),
            "n_chars": percentiles([float(r["n_chars"]) for r in rs]),
            "duration_histogram": _hist(durs, [0, 1, 1.5, 2, 5, 10, 20, 30, 40, 60]),
        }

    # ---- threshold sweep: what would each filter cost, per language? ----
    for lang, rs in sorted(by_lang.items()):
        n = len(rs)
        sweep = {"n_clips": n}
        sweep["min_duration"] = {f"<{t}s": _pct(sum(1 for r in rs if r["duration"] < t), n)
                                 for t in MIN_DUR_CANDIDATES}
        sweep["max_duration"] = {f">{t}s": _pct(sum(1 for r in rs if r["duration"] > t), n)
                                 for t in MAX_DUR_CANDIDATES}
        sweep["max_words_per_sec"] = {f">{t}": _pct(sum(1 for r in rs if r["wps"] > t), n)
                                      for t in MAX_WPS_CANDIDATES}
        sweep["low_words_per_sec"] = {f"<{t}": _pct(sum(1 for r in rs if 0 < r["wps"] < t), n)
                                      for t in LOW_WPS_CANDIDATES}
        sweep["empty_text"] = _pct(sum(1 for r in rs if r["n_words"] == 0), n)
        # overlap: are the short clips ALSO the impossibly-fast ones (i.e. corrupt)?
        short = [r for r in rs if r["duration"] < 1.5]
        sweep["short_clips_that_are_also_fast"] = {
            "n_short_lt_1.5s": len(short),
            "of_which_wps_gt_4": sum(1 for r in short if r["wps"] > 4),
            "of_which_wps_gt_6": sum(1 for r in short if r["wps"] > 6),
            "median_words_in_short_clips": _median([float(r["n_words"]) for r in short]),
        }
        report["threshold_sweep"][lang] = sweep

    # ---- extremes: eyeball whether they're really corrupt ----
    for lang, rs in sorted(by_lang.items()):
        def top(key, reverse):
            picked = sorted(rs, key=lambda r: r[key], reverse=reverse)[:top_k]
            return [{"id": r["id"], "duration_s": round(r["duration"], 2), "n_words": r["n_words"],
                     "wps": round(r["wps"], 2), "text": r["text"][:180]} for r in picked]
        report["extremes"][lang] = {
            "shortest": top("duration", False),
            "longest": top("duration", True),
            "fastest_wps": top("wps", True),
            "slowest_wps": top("wps", False),
        }

    # ---- character inventory (feeds the cased CTC vocab) ----
    per_lang_chars: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        per_lang_chars[r["language"]].update(r["text"])
    all_chars: Counter = Counter()
    for c in per_lang_chars.values():
        all_chars.update(c)
    rare = sorted([(ch, n) for ch, n in all_chars.items() if n < 20 and ch != " "], key=lambda x: x[1])
    report["characters"] = {
        "total_distinct": len(all_chars),
        "per_language_distinct": {l: len(c) for l, c in sorted(per_lang_chars.items())},
        "rare_under_20": [{"char": ch, "codepoint": f"U+{ord(ch):04X}", "count": n} for ch, n in rare[:100]],
        "has_uppercase": any(ch.isupper() for ch in all_chars),
        "punctuation": sorted({ch for ch in all_chars if not ch.isalnum() and not ch.isspace()}),
    }

    # ---- duplicates / degenerate text ----
    text_counts = Counter(r["text"] for r in rows)
    dups = [(t, n) for t, n in text_counts.items() if n > 1]
    report["duplicates"] = {
        "n_duplicate_texts": len(dups),
        "n_clips_with_duplicated_text": sum(n for _, n in dups),
        "most_repeated": [{"count": n, "text": t[:120]} for t, n in
                          sorted(dups, key=lambda x: -x[1])[:10]],
    }
    return report


def _pct(k: int, n: int) -> dict:
    return {"n": k, "pct": round(100 * k / max(1, n), 2)}


def _median(vals: list[float]):
    if not vals:
        return None
    s = sorted(vals)
    return s[len(s) // 2]


def _log_report(report: dict, top_k: int) -> None:
    logger.info("===== PER-LANGUAGE DISTRIBUTIONS =====")
    for lang, d in report["per_language"].items():
        dur, wps = d["duration_s"], d["words_per_sec"]
        logger.info("  %s: %d clips, %.1f h | dur p50 %.1fs p90 %.1fs p99 %.1fs max %.1fs "
                    "| wps p50 %.2f p99 %.2f | words p50 %.0f",
                    lang, d["n_clips"], d["hours"], dur["p50"], dur["p90"], dur["p99"],
                    dur["max"], wps["p50"], wps["p99"], d["n_words"]["p50"])
        logger.info("     duration histogram: %s", d["duration_histogram"])

    logger.info("===== THRESHOLD SWEEP (what each filter would cost) =====")
    for lang, s in report["threshold_sweep"].items():
        logger.info("  %s (n=%d)", lang, s["n_clips"])
        logger.info("     drop if dur <  : %s", {k: f"{v['n']} ({v['pct']}%)" for k, v in s["min_duration"].items()})
        logger.info("     drop if dur >  : %s", {k: f"{v['n']} ({v['pct']}%)" for k, v in s["max_duration"].items()})
        logger.info("     drop if wps >  : %s", {k: f"{v['n']} ({v['pct']}%)" for k, v in s["max_words_per_sec"].items()})
        logger.info("     flag  if wps <  : %s", {k: f"{v['n']} ({v['pct']}%)" for k, v in s["low_words_per_sec"].items()})
        o = s["short_clips_that_are_also_fast"]
        logger.info("     KEY: of %d clips <1.5s, %d have wps>4 (%d have wps>6); median words in them: %s",
                    o["n_short_lt_1.5s"], o["of_which_wps_gt_4"], o["of_which_wps_gt_6"],
                    o["median_words_in_short_clips"])

    logger.info("===== EXTREMES (eyeball for corruption) =====")
    for lang, ex in report["extremes"].items():
        logger.info("  --- %s: shortest ---", lang)
        for r in ex["shortest"][:top_k]:
            logger.info("     %.2fs %2dw wps=%.1f | %r", r["duration_s"], r["n_words"], r["wps"], r["text"][:90])
        logger.info("  --- %s: fastest (wps) ---", lang)
        for r in ex["fastest_wps"][:top_k]:
            logger.info("     %.2fs %2dw wps=%.1f | %r", r["duration_s"], r["n_words"], r["wps"], r["text"][:90])

    c = report["characters"]
    logger.info("===== CHARACTERS ===== distinct=%d per-lang=%s uppercase=%s",
                c["total_distinct"], c["per_language_distinct"], c["has_uppercase"])
    logger.info("  punctuation: %s", "".join(c["punctuation"]))
    if c["rare_under_20"]:
        logger.info("  %d rare chars (<20x) — check for junk: %s", len(c["rare_under_20"]),
                    [r["char"] for r in c["rare_under_20"][:30]])
    d = report["duplicates"]
    logger.info("===== DUPLICATES ===== %d distinct texts repeat, covering %d clips",
                d["n_duplicate_texts"], d["n_clips_with_duplicated_text"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()
    out_dir = args.output_dir or os.path.join(args.data_dir, "logs")

    logger.info("Loading metadata (no audio decode) from %s ...", args.data_dir)
    rows = load_meta(args.data_dir, args.splits)
    if not rows:
        logger.error("No data found under %s — run prepare_data.py first.", args.data_dir)
        return
    logger.info("Loaded %d clips across splits %s.", len(rows), args.splits)

    report = analyse(rows, args.top_k)
    _log_report(report, args.top_k)
    path = os.path.join(out_dir, "eda_report.json")
    write_json(path, report)
    logger.info("Full report -> %s", path)
    logger.info("Read the sweep + extremes, THEN pick thresholds for clean_data.py.")


if __name__ == "__main__":
    main()
