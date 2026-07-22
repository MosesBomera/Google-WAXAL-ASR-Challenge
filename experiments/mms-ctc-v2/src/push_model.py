"""Push a trained model (or logs) to the Hugging Face Hub so it survives instance teardown.

Private by default — competition work stays yours. Checkpoint dirs are EXCLUDED by default
(~6 GB each, only useful on the box). A fresh box pulls it back with from_pretrained("<repo>").

    python push_model.py --repo-id myuser/waxal-mms-1b-ctc-v2
    python push_model.py --path artifacts/mms-1b-all/logs --repo-id myuser/waxal-v2-logs --repo-type dataset
    python push_model.py --repo-id myuser/waxal-mms-1b-ctc-v2 --no-private
"""

from __future__ import annotations

import argparse
import logging
import os

from common import artifacts_dir, quiet_hf_logging

logger = logging.getLogger("push_model")

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
    p.add_argument("--repo-id", required=True, help="e.g. myuser/waxal-mms-1b-ctc-v2")
    p.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-checkpoints", action="store_true")
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
