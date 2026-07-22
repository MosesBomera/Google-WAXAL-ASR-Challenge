"""experiment mms-ctc / Stage 3a — build a KenLM n-gram LM from lm/corpus.txt.

An n-gram LM scores how word-like a transcription is; folded into CTC beam search
(infer.py --kenlm) it fixes the word-boundary errors that dominate Lingala's WER
and that a frame-independent greedy CTC decode cannot. See the teaching notes.

The corpus (Stage 1c) is the WAXAL training transcripts with VAL speakers excluded
— so the LM never memorises validation sentences, keeping the val/board honest.

Needs the KenLM `lmplz` binary (build once on the box):
    apt-get update && apt-get install -y build-essential cmake libboost-all-dev \
        libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
    git clone https://github.com/kpu/kenlm && cd kenlm && mkdir -p build && cd build \
        && cmake .. && make -j4              # -> kenlm/build/bin/{lmplz,build_binary}
    pip install kenlm pyctcdecode            # query module + CTC beam decoder

    python build_lm.py                                    # 4-gram -> lm/waxal.arpa
    python build_lm.py --order 5 --binary --lmplz-bin kenlm/build/bin/lmplz
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess

from common import quiet_hf_logging, workdir

logger = logging.getLogger("build_lm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=str(workdir() / "lm" / "corpus.txt"))
    p.add_argument("--order", type=int, default=4, help="n-gram order (4 is a good default here).")
    p.add_argument("--output", default=str(workdir() / "lm" / "waxal.arpa"))
    p.add_argument("--binary", action="store_true",
                   help="Also produce a compact .bin (faster to load) via build_binary.")
    p.add_argument("--lmplz-bin", default="lmplz", help="Path to the kenlm lmplz binary.")
    p.add_argument("--build-binary-bin", default="build_binary")
    p.add_argument("--discount-fallback", action=argparse.BooleanOptionalAction, default=True,
                   help="Needed for small/sparse corpora — Kneser-Ney can fail on unseen high-order "
                        "count-of-counts; fallback keeps lmplz from erroring. Ours is ~900k words.")
    return p.parse_args()


def _resolve(binary: str) -> str | None:
    if os.path.isfile(binary) and os.access(binary, os.X_OK):
        return binary
    return shutil.which(binary)


def corpus_stats(path: str) -> tuple[int, int]:
    lines = words = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            lines += 1
            words += len(line.split())
    return lines, words


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()

    if not os.path.isfile(args.corpus):
        raise SystemExit(f"No corpus at {args.corpus} — run clean_split.py (Stage 1c) first.")
    lmplz = _resolve(args.lmplz_bin)
    if not lmplz:
        raise SystemExit(
            f"kenlm 'lmplz' not found ({args.lmplz_bin}). Build it:\n"
            "  git clone https://github.com/kpu/kenlm && cd kenlm && mkdir build && cd build "
            "&& cmake .. && make -j4\n"
            "then pass --lmplz-bin kenlm/build/bin/lmplz")

    lines, words = corpus_stats(args.corpus)
    logger.info("Corpus %s: %d lines, %d words", args.corpus, lines, words)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    cmd = [lmplz, "-o", str(args.order), "--text", args.corpus, "--arpa", args.output]
    if args.discount_fallback:
        cmd.append("--discount_fallback")
    logger.info("Building %d-gram LM: %s", args.order, " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Wrote %s (%.1f MB)", args.output, os.path.getsize(args.output) / 1e6)

    if args.binary:
        bb = _resolve(args.build_binary_bin)
        if not bb:
            logger.warning("build_binary not found — skipping .bin. ARPA works fine with pyctcdecode.")
        else:
            out_bin = os.path.splitext(args.output)[0] + ".bin"
            subprocess.run([bb, args.output, out_bin], check=True)
            logger.info("Wrote %s (%.1f MB)", out_bin, os.path.getsize(out_bin) / 1e6)

    logger.info("Next: python infer.py --kenlm %s --lm-alpha 0.5 --lm-beta 1.5", args.output)


if __name__ == "__main__":
    main()
