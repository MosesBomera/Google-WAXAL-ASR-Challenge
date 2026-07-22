"""mms-ctc-v2 / Stage 0 — what normalization does the WAXAL leaderboard use?

Decides a v2 design fork: our v1 vocab is CASED + PUNCTUATED (to protect the CER half). But
Sunbird's Kinyarwanda system trained/scored on LOWERCASE UNPUNCTUATED text — hinting that *some*
Bantu-ASR leaderboards normalize case/punct before scoring. If WAXAL's does too, cased+punct is
wasted head capacity and we should switch targets to lowercase-unpunct. If it doesn't, cased+punct
is justified. We settle it with evidence, not instinct.

The lever: we already know the board's OWN numbers on v1's cased submission —
    board combined 0.790182  →  WER 0.315922 / CER 0.103714   (= 0.5·WER + 0.5·CER error)
…yet our local cased eval reported WER 0.308 / CER 0.095. The gap is a NORMALIZATION difference.
So we recompute WER/CER on the SAME submission vs the SAME references under several normalization
schemes and see which one REPRODUCES the board's 0.315922 / 0.103714. That scheme is the board's.

  minimal (cased+punct) reproduces the board  → board is case/punct-SENSITIVE → KEEP cased+punct.
  lowercase+strip-punct reproduces the board  → board NORMALIZES        → SWITCH to lowercase-unpunct.

Reading the test transcripts here is a scoring-reproduction check, not training on them.

    pip install jiwer
    python scorer_probe.py            # streams refs once, caches, prints the scheme table + verdict
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import unicodedata

logger = logging.getLogger("scorer_probe")

# The board's own metrics on v1's cased submission (from the leaderboard).
BOARD = {"wer": 0.315922124, "cer": 0.103714041}
LANGS = ["lin", "sna", "lug"]
TEXT_FIELDS = ("transcription", "text", "sentence", "raw_transcription")

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _ws(s: str) -> str:
    return " ".join(str(s).split())


SCHEMES = {
    "minimal (cased+punct)": lambda s: _ws(unicodedata.normalize("NFKC", str(s))),
    "lowercase (keep punct)": lambda s: _ws(unicodedata.normalize("NFKC", str(s)).lower()),
    "strip-punct (keep case)": lambda s: _ws(_PUNCT.sub(" ", unicodedata.normalize("NFKC", str(s)))),
    "lowercase + strip-punct": lambda s: _ws(_PUNCT.sub(" ", unicodedata.normalize("NFKC", str(s)).lower())),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--submission",
                   default=os.path.normpath(os.path.join(_here, "..", "..", "mms-ctc",
                                                          "training-artifacts", "submission.csv")))
    p.add_argument("--dataset-id", default="google/WaxalNLP")
    p.add_argument("--languages", nargs="+", default=LANGS)
    p.add_argument("--refs-cache", default=os.path.join(_here, "..", "logs", "test_refs.csv"),
                   help="Cache of id,language,reference streamed from the Hub (skips re-streaming).")
    p.add_argument("--hf-token", default=None)
    return p.parse_args()


def _first_field(example: dict, candidates):
    for c in candidates:
        if c in example and example[c] not in (None, ""):
            return c
    lower = {k.lower(): k for k in example}
    return next((lower[c] for c in candidates if c in lower), None)


def _no_audio_decode(ds):
    import datasets
    for col, feat in dict(getattr(ds, "features", None) or {}).items():
        if isinstance(feat, datasets.Audio):
            ds = ds.cast_column(col, datasets.Audio(decode=False))
    return ds


def load_references(args) -> dict:
    """id -> reference transcript, for the WAXAL test split (text-only stream; cached)."""
    import pandas as pd

    if os.path.isfile(args.refs_cache):
        df = pd.read_csv(args.refs_cache, dtype=str).fillna("")
        logger.info("Loaded %d cached references from %s", len(df), args.refs_cache)
        return dict(zip(df["id"], df["reference"]))

    import datasets
    if args.hf_token:
        from huggingface_hub import login
        login(args.hf_token)

    rows = []
    for lang in args.languages:
        stream = _no_audio_decode(datasets.load_dataset(args.dataset_id, name=f"{lang}_asr",
                                                        split="test", streaming=True))
        n = 0
        for ex in stream:
            tf = _first_field(ex, TEXT_FIELDS)
            _id = str(ex.get("id", ""))
            if tf and _id:
                rows.append({"id": _id, "language": lang, "reference": str(ex[tf])})
                n += 1
                if n % 1000 == 0:
                    logger.info("  …%s test: %d refs", lang, n)
        logger.info("%s test: %d references", lang, n)

    os.makedirs(os.path.dirname(os.path.abspath(args.refs_cache)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.refs_cache, index=False)
    logger.info("Cached %d references -> %s", len(rows), args.refs_cache)
    return {r["id"]: r["reference"] for r in rows}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        import jiwer
    except ImportError:
        raise SystemExit("jiwer not installed — run: pip install jiwer")
    import pandas as pd

    args = parse_args()
    sub = pd.read_csv(args.submission, dtype=str).fillna("")
    id_col = "ID" if "ID" in sub.columns else sub.columns[0]
    txt_col = "Target" if "Target" in sub.columns else sub.columns[1]
    preds = dict(zip(sub[id_col], sub[txt_col]))
    logger.info("Submission: %d predictions (%s)", len(preds), args.submission)

    refs = load_references(args)

    # align by id, bucket by language (from the id prefix lin_/sna_/lug_)
    by_lang: dict[str, dict[str, list]] = {l: {"ref": [], "hyp": []} for l in args.languages}
    missing = 0
    for _id, hyp in preds.items():
        ref = refs.get(_id)
        if ref is None:
            missing += 1
            continue
        lang = _id.split("_", 1)[0]
        if lang in by_lang:
            by_lang[lang]["ref"].append(ref)
            by_lang[lang]["hyp"].append(hyp)
    if missing:
        logger.warning("%d submission ids had no reference (skipped)", missing)
    for l in args.languages:
        logger.info("  %s: %d aligned pairs", l, len(by_lang[l]["ref"]))

    # --- score every normalization scheme, macro over languages + micro pooled ---
    def score(refs_l, hyps_l, norm):
        r = [norm(x) for x in refs_l]
        h = [norm(x) for x in hyps_l]
        keep = [(a, b) for a, b in zip(r, h) if a.strip() != ""]   # jiwer needs non-empty refs
        r, h = [a for a, _ in keep], [b for _, b in keep]
        return jiwer.wer(r, h), jiwer.cer(r, h)

    logger.info("\n%-26s | %-22s | %-22s", "scheme", "MACRO (mean of langs)", "MICRO (pooled)")
    logger.info("%-26s | %-22s | %-22s", "", "WER    CER   combined", "WER    CER   combined")
    logger.info("-" * 78)

    results = {}
    all_ref = [x for l in args.languages for x in by_lang[l]["ref"]]
    all_hyp = [x for l in args.languages for x in by_lang[l]["hyp"]]
    for name, norm in SCHEMES.items():
        per = {l: score(by_lang[l]["ref"], by_lang[l]["hyp"], norm) for l in args.languages}
        mac_w = sum(w for w, _ in per.values()) / len(per)
        mac_c = sum(c for _, c in per.values()) / len(per)
        mic_w, mic_c = score(all_ref, all_hyp, norm)
        results[name] = {"macro": (mac_w, mac_c), "micro": (mic_w, mic_c), "per": per}
        logger.info("%-26s | %.4f %.4f %.4f | %.4f %.4f %.4f", name,
                    mac_w, mac_c, 0.5 * (mac_w + mac_c), mic_w, mic_c, 0.5 * (mic_w + mic_c))

    # --- which scheme×aggregation reproduces the board? ---
    logger.info("\nBoard target: WER %.4f  CER %.4f  combined %.4f",
                BOARD["wer"], BOARD["cer"], 0.5 * (BOARD["wer"] + BOARD["cer"]))
    best = None
    for name, r in results.items():
        for agg in ("macro", "micro"):
            w, c = r[agg]
            dist = abs(w - BOARD["wer"]) + abs(c - BOARD["cer"])
            if best is None or dist < best[0]:
                best = (dist, name, agg, w, c)
    dist, name, agg, w, c = best
    logger.info("Closest to board: **%s** [%s]  (WER %.4f CER %.4f, L1 dist %.4f)", name, agg, w, c, dist)

    verdict = ("KEEP cased+punct — the board is case/punct-SENSITIVE."
               if name.startswith("minimal") else
               "SWITCH to lowercase-unpunct — the board normalizes it away."
               if name.startswith("lowercase + strip") else
               f"MIXED — board looks like '{name}'; inspect before deciding the vocab.")
    logger.info("VERDICT: %s", verdict)
    if dist > 0.01:
        logger.warning("Closest scheme is still %.4f off the board — board may use extra rules "
                       "(e.g. number/word expansion); treat the verdict as indicative.", dist)


if __name__ == "__main__":
    main()
