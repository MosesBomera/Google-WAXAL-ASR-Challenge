"""Push a trained model (or logs) to the Hugging Face Hub so it survives instance
teardown — the practical answer to "the full model is GBs, where do I keep it?".

Why the Hub rather than downloading through the browser:
  * a full-FT MMS-1B is ~3.9 GB — the Hub uploads/downloads it far faster and
    resumably, and you already have HF_TOKEN set for the dataset;
  * a new box pulls it straight back with `from_pretrained("<user>/<repo>")`;
  * **private by default** — competition work stays yours. (Kaggle Datasets also
    works but needs separate credentials and isn't model-shaped; the Hub is the
    natural fit here.)

Checkpoint dirs are EXCLUDED by default: they're ~6 GB each and only useful on the
box itself for averaging. Use --include-checkpoints if you really want them.

    python push_model.py --repo-id myuser/waxal-mms-1b-ctc
    python push_model.py --path artifacts/mms-1b-all/logs --repo-id myuser/waxal-logs --repo-type dataset
    python push_model.py --repo-id myuser/waxal-mms-1b-ctc --no-private     # make it public
"""

from __future__ import annotations

import argparse
import logging
import os

from common import artifacts_dir, quiet_hf_logging

logger = logging.getLogger("push_model")

# Big, box-only, or not worth shipping.
DEFAULT_IGNORE = ["checkpoint-*/*", "checkpoint-*", "runs/*", "*.pyc", "__pycache__/*"]


def push_folder(path: str, repo_id: str, *, private: bool = True, token: str | None = None,
                repo_type: str = "model", include_checkpoints: bool = False,
                commit_message: str = "upload from vast.ai") -> str:
    """Creates the repo if needed and uploads `path`. Returns the repo URL."""
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

    ignore = None if include_checkpoints else DEFAULT_IGNORE
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(path) for f in fs) / 1e9
    logger.info("Uploading %s (%.1f GB on disk%s) -> %s/%s [private=%s]",
                path, size, "" if include_checkpoints else ", checkpoints excluded",
                repo_type + "s" if repo_type != "model" else "models", repo_id, private)

    api.upload_folder(folder_path=path, repo_id=repo_id, repo_type=repo_type,
                      ignore_patterns=ignore, commit_message=commit_message)
    url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
    logger.info("Done -> %s", url)
    logger.info("Pull it back on a fresh box with:  from_pretrained(%r)", repo_id)
    return url


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", default=str(artifacts_dir()), help="Folder to upload.")
    p.add_argument("--repo-id", required=True, help="e.g. myuser/waxal-mms-1b-ctc")
    p.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-checkpoints", action="store_true",
                   help="Also upload checkpoint-*/ (~6 GB each) — usually pointless off-box.")
    p.add_argument("--commit-message", default="upload from vast.ai")
    p.add_argument("--hf-token", default=None)
    args = p.parse_args()

    if not os.path.isdir(args.path):
        raise SystemExit(f"No such folder: {args.path}")
    push_folder(args.path, args.repo_id, private=args.private, token=args.hf_token,
                repo_type=args.repo_type, include_checkpoints=args.include_checkpoints,
                commit_message=args.commit_message)


if __name__ == "__main__":
    main()
