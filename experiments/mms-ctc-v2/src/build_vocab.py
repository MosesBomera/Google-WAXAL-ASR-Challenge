"""mms-ctc-v2 / Stage 1f — build the CASED CTC vocab from the TRAIN manifest (WAXAL + FLEURS + LRSC).

Runs AFTER prepare_external.py AND prepare_lrsc.py so the vocab covers FLEURS characters and
LRSC's real Lingala open-vowel letters ɔ/ɛ too (no silent [UNK] → CER rot). Unlike MMS's default
(lowercase + strip punct), we KEEP case + punctuation — the board scores CER case- AND
punct-sensitive (scorer_probe.py), so cased+punct directly earns CER.

Correctness details vs v1:
  * built from the actual TRAIN split (manifests/train.csv ids), not "whatever sits in
    prepared/train/" — our speaker re-split moves clips between corpus dirs and drops corrupt ones;
  * transcripts pass through `normalize_for_ctc` (the SAME normalization used for CTC labels), so
    the vocab is exactly the character set the model will emit.

Output (HF Wav2Vec2CTCTokenizer format):
  vocab.json  — {char: id}, space -> "|" (delimiter), plus "[UNK]" and "[PAD]" (PAD = CTC blank).
  vocab_report.json — size, per-language sets, rare chars to eyeball.

    python build_vocab.py            # reads manifests/train.csv + prepared/{train,validation}/*/
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter

from common import normalize_for_ctc, prepared_dir, quiet_hf_logging, workdir, write_json

logger = logging.getLogger("build_vocab")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(prepared_dir()),
                   help="Prepared chunks (WAXAL + FLEURS). Both corpus dirs are scanned.")
    p.add_argument("--manifest", default=str(workdir() / "manifests" / "train.csv"))
    p.add_argument("--output-dir", default=str(workdir() / "vocab"))
    p.add_argument("--min-count", type=int, default=1,
                   help="Chars below this count are dropped to [UNK] (1 = keep all; junk is already "
                        "handled by normalize_for_ctc's CHAR_MAP).")
    return p.parse_args()


def load_train_transcripts(data_dir: str, train_ids: set) -> tuple[dict, dict]:
    """id -> normalized transcript, for the manifest's TRAIN ids only. Scans both corpus
    dirs because the speaker-disjoint re-split moves clips around; external data lives in
    train/fleurs-* (FLEURS) and train/lrsc-* (LRSC)."""
    import datasets

    id2text: dict[str, str] = {}
    for corpus_split in ("train", "validation"):
        d = os.path.join(data_dir, corpus_split)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if not os.path.isdir(p):
                continue
            ds = datasets.load_from_disk(p)                       # lazy; reads only id+transcription
            for _id, t in zip(ds["id"], ds["transcription"]):
                if _id in train_ids:
                    id2text[_id] = normalize_for_ctc(t)
    return id2text, {}


def build(args: argparse.Namespace) -> None:
    import pandas as pd

    os.makedirs(args.output_dir, exist_ok=True)
    man = pd.read_csv(args.manifest, dtype={"id": str})
    train_ids = set(man["id"])
    id2lang = dict(zip(man["id"], man["language"]))
    logger.info("Train manifest: %d ids. Reading transcripts from %s ...", len(train_ids), args.data_dir)

    id2text, _ = load_train_transcripts(args.data_dir, train_ids)
    found = len(id2text)
    if found < len(train_ids):
        logger.warning("%d/%d train ids had no chunk on disk (run prepare_data/prepare_external first?)",
                       len(train_ids) - found, len(train_ids))

    char_counts: Counter = Counter()
    per_lang_chars: dict[str, set] = {}
    for _id, t in id2text.items():
        char_counts.update(t)
        per_lang_chars.setdefault(id2lang.get(_id, "?"), set()).update(t)

    kept = {c for c, n in char_counts.items() if n >= args.min_count}
    kept.discard(" ")
    vocab_chars = sorted(kept)
    vocab = {c: i for i, c in enumerate(vocab_chars)}
    vocab["|"] = len(vocab)          # word delimiter (space)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)      # doubles as the CTC blank
    write_json(os.path.join(args.output_dir, "vocab.json"), vocab)

    rare = sorted([(c, n) for c, n in char_counts.items() if n < 20 and c != " "], key=lambda x: x[1])
    report = {
        "vocab_size": len(vocab), "n_chars_from_data": len(vocab_chars),
        "n_train_transcripts": found, "specials": ["|", "[UNK]", "[PAD]"],
        "has_uppercase": any(ch.isupper() for ch in char_counts),
        "punctuation": sorted(ch for ch in char_counts if not ch.isalnum() and not ch.isspace()),
        "per_language_char_counts": {l: len(s) for l, s in sorted(per_lang_chars.items())},
        "chars_only_in_one_language": _exclusive_chars(per_lang_chars),
        "rare_chars_under_20": [{"char": c, "count": n, "codepoint": f"U+{ord(c):04X}"} for c, n in rare[:80]],
    }
    write_json(os.path.join(args.output_dir, "vocab_report.json"), report)

    logger.info("Vocab size %d (%d data chars + | [UNK] [PAD]) | uppercase=%s",
                len(vocab), len(vocab_chars), report["has_uppercase"])
    logger.info("  punctuation: %s", "".join(report["punctuation"]))
    logger.info("  per-language chars: %s", report["per_language_char_counts"])
    if rare:
        logger.info("  %d rare chars (<20x) — eyeball vocab_report.json for FLEURS junk (stray "
                    "symbols/digits) to add to CHAR_MAP.", len(rare))
    logger.info("Next: train_ctc.py")


def _exclusive_chars(per_lang_chars: dict[str, set]) -> dict:
    out = {}
    langs = list(per_lang_chars)
    for l in langs:
        others = set().union(*(per_lang_chars[o] for o in langs if o != l)) if len(langs) > 1 else set()
        excl = sorted(per_lang_chars[l] - others - {" "})
        if excl:
            out[l] = excl
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    build(parse_args())


if __name__ == "__main__":
    main()
