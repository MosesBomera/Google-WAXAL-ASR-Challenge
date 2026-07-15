"""mms-ctc / Stage 1 — build the CASED CTC character vocabulary from the
cleaned training transcripts.

Unlike MMS's default pipeline (lowercase + strip punctuation), we KEEP case and
punctuation so the CTC head can emit text that matches WAXAL's cased, punctuated
references — protecting the CER half of the metric regardless of how the
leaderboard normalizes.

Output (HF Wav2Vec2CTCTokenizer format):
  * vocab.json  — {char: id}, with space -> "|" (word delimiter), plus "[UNK]"
    and "[PAD]" (PAD doubles as the CTC blank).
  * vocab_report.json — size, per-language character sets, and any rare/suspect
    characters to eyeball before training.

    python build_vocab.py                       # reads ./prepared/waxal-clean
    python build_vocab.py --data-dir ... --output-dir ...
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter

from common import prepared_dir, quiet_hf_logging, write_json

logger = logging.getLogger("build_vocab")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(prepared_dir()),
                   help="Cleaned chunks from prepare_data.py (uses the 'train' split).")
    p.add_argument("--output-dir", default=None, help="Default: <data-dir>/vocab")
    p.add_argument("--split", default="train")
    p.add_argument("--min-count", type=int, default=1,
                   help="Chars appearing fewer than this many times are routed to [UNK] "
                        "(default 1 = keep everything; raise to prune noise chars).")
    return p.parse_args()


def load_transcripts(data_dir: str, split: str):
    import datasets
    split_dir = os.path.join(data_dir, split)
    texts, langs = [], []
    for name in sorted(os.listdir(split_dir)):
        p = os.path.join(split_dir, name)
        if not os.path.isdir(p):
            continue
        ds = datasets.load_from_disk(p)
        texts.extend(ds["transcription"])
        langs.extend([name] * len(ds))
    return texts, langs


def build(args: argparse.Namespace) -> None:
    out_dir = args.output_dir or os.path.join(args.data_dir, "vocab")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Loading %s transcripts from %s ...", args.split, args.data_dir)
    texts, langs = load_transcripts(args.data_dir, args.split)
    logger.info("Loaded %d transcripts.", len(texts))

    char_counts: Counter = Counter()
    per_lang_chars: dict[str, set] = {}
    for t, l in zip(texts, langs):
        char_counts.update(t)
        per_lang_chars.setdefault(l, set()).update(t)

    # Keep chars meeting min-count; " " becomes the word delimiter "|".
    kept = {c for c, n in char_counts.items() if n >= args.min_count}
    kept.discard(" ")
    vocab_chars = sorted(kept)

    vocab = {c: i for i, c in enumerate(vocab_chars)}
    vocab["|"] = len(vocab)          # word delimiter (space)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)      # doubles as the CTC blank

    write_json(os.path.join(out_dir, "vocab.json"), vocab)

    rare = sorted([(c, n) for c, n in char_counts.items() if n < 20 and c != " "],
                  key=lambda x: x[1])
    report = {
        "vocab_size": len(vocab),
        "n_chars_from_data": len(vocab_chars),
        "specials": ["|", "[UNK]", "[PAD]"],
        "per_language_char_counts": {l: len(s) for l, s in sorted(per_lang_chars.items())},
        "chars_only_in_one_language": _exclusive_chars(per_lang_chars),
        "rare_chars_under_20": [{"char": c, "count": n, "codepoint": f"U+{ord(c):04X}"}
                                 for c, n in rare[:80]],
    }
    write_json(os.path.join(out_dir, "vocab_report.json"), report)

    logger.info("Vocab size %d (%d data chars + 3 specials) -> %s",
                len(vocab), len(vocab_chars), os.path.join(out_dir, "vocab.json"))
    logger.info("Per-language char counts: %s", report["per_language_char_counts"])
    if rare:
        logger.info("%d rare chars (<20 occurrences) — eyeball vocab_report.json for noise "
                    "(stray symbols, wrong-script chars).", len(rare))
    logger.info("Next: Stage 2 — MMS-1B CTC fine-tune with vocab %s", os.path.join(out_dir, "vocab.json"))


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
