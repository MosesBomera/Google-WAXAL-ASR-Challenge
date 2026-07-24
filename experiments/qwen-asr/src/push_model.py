"""Push a trained Qwen-ASR LoRA checkpoint to the Hugging Face Hub so it survives instance
teardown / disk pressure. Private by default — competition work stays yours.

Unlike the mms-ctc-v2 push (which EXCLUDES checkpoint dirs), here the checkpoint IS the model
(a LoRA adapter + copied base config/tokenizer from MakeEveryCheckpointInferableCallback), so we
upload the folder wholesale. The base Qwen/Qwen3-ASR-1.7B is re-downloadable from the Hub, so the
adapter is all we need to keep.

    python push_model.py --repo-id mosesbomera/waxal-qwen3-asr-gate   # pushes checkpoint-2065
    python push_model.py --path artifacts/qwen-gate/checkpoint-2065 --repo-id <you>/<repo>
    python push_model.py --path artifacts/qwen-gate/logs --repo-id <you>/waxal-qwen-logs --repo-type dataset

Pull it back on a fresh box:  PeftModel.from_pretrained(base_model, "<repo-id>")
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from common import quiet_hf_logging

logger = logging.getLogger("push_model")

SCRIPT_DIR = Path(__file__).resolve().parent

# Training-resume state — big and useless for inference/preservation. Excluded by default;
# optimizer.pt alone is ~1.4 GB. Pass --include-optimizer only if you intend to resume THIS run.
RESUME_STATE = ["optimizer.pt", "rng_state.pth", "scheduler.pt"]


def _default_ckpt() -> Path:
    """Best/last gate checkpoint if present, else the gate output dir."""
    exp = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
    gate = exp / "artifacts" / "qwen-gate"
    if gate.is_dir():
        ckpts = sorted((d for d in gate.iterdir() if d.name.startswith("checkpoint-")),
                       key=lambda d: int(d.name.split("-")[1]))
        if ckpts:
            return ckpts[-1]   # last = best (loss/combined converged at end here)
    return gate


def push_folder(path: str, repo_id: str, *, private: bool = True, token: str | None = None,
                repo_type: str = "model", commit_message: str = "upload qwen-asr checkpoint",
                include_optimizer: bool = False) -> str:
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

    ignore = None if include_optimizer else RESUME_STATE
    files = [os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs
             if include_optimizer or f not in RESUME_STATE]
    size = sum(os.path.getsize(f) for f in files) / 1e9
    logger.info("Uploading %s (%d files, %.2f GB%s) -> %s [private=%s]",
                path, len(files), size, "" if include_optimizer else ", resume-state excluded",
                repo_id, private)

    api.upload_folder(folder_path=path, repo_id=repo_id, repo_type=repo_type,
                      ignore_patterns=ignore, commit_message=commit_message)
    url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
    logger.info("Done -> %s", url)
    return url


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", default=str(_default_ckpt()), help="Checkpoint folder to upload.")
    p.add_argument("--repo-id", required=True, help="e.g. mosesbomera/waxal-qwen3-asr-gate")
    p.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-optimizer", action="store_true",
                   help="Also push optimizer.pt/rng/scheduler (~1.4 GB) to resume THIS run later.")
    p.add_argument("--commit-message", default="upload qwen-asr checkpoint")
    p.add_argument("--hf-token", default=None)
    args = p.parse_args()

    if not os.path.isdir(args.path):
        raise SystemExit(f"No such folder: {args.path}")
    push_folder(args.path, args.repo_id, private=args.private, token=args.hf_token,
                repo_type=args.repo_type, commit_message=args.commit_message,
                include_optimizer=args.include_optimizer)


if __name__ == "__main__":
    main()
