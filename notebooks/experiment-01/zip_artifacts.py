"""Zip the best-N checkpoints (+ processor + logs) from a finished training run,
so you can download everything you need BEFORE destroying the instance.

Ranks checkpoint-*/ by eval 'combined' (from logs/trainer_log_history.json) and
packs the top-N adapter weights (NOT the bulky optimizer state), the root best
model, and all logs into one small zip.

    python zip_artifacts.py                         # top 3 from ./artifacts/whisper-large-v3-lora
    python zip_artifacts.py --top-n 5 --output /workspace/v1_top5.zip
"""

from __future__ import annotations

import argparse
import logging

from common import artifacts_dir, quiet_hf_logging, rank_checkpoints_by_combined, zip_best_checkpoints

logger = logging.getLogger("zip_artifacts")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact-dir", default=str(artifacts_dir()))
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--output", default=None, help="Default: <artifact-dir>_best<N>.zip")
    args = p.parse_args()

    ranking = rank_checkpoints_by_combined(args.artifact_dir)
    logger.info("Checkpoints found (best first):")
    for step, _, combined in ranking:
        logger.info("  checkpoint-%d  combined=%s", step, "n/a" if combined == float("inf") else f"{combined:.4f}")

    out, selected = zip_best_checkpoints(args.artifact_dir, args.top_n, args.output)
    logger.info("Zipped top %d checkpoints -> %s", len(selected), out)
    for step, combined in selected:
        logger.info("  included checkpoint-%d (combined=%s)", step,
                    "n/a" if combined == float("inf") else f"{combined:.4f}")
    logger.info("Download %s before destroying the instance.", out)


if __name__ == "__main__":
    main()
