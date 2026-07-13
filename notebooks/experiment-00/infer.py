"""Loads the LoRA adapter saved by train.py, applies it to the base Gemma 3n
model, and:
  1. (optional) reports WER/CER against the WaxalNLP test split's ground
     truth, as a local sanity check — NOT the Zindi leaderboard score, but
     correlates with it. Safe/normal to do locally; only *submitting* the
     public test-set ground truth as your predictions would break the rules.
  2. (optional) generates submission.csv for the Zindi leaderboard.

No gradient updates happen here — this is inference only. --model-id and the
prompt flags must match whatever train.py was run with, since the adapter
was fit against that exact prompt template.

Usage:
    python infer.py --artifact-dir ./gemma3n-asr-multilingual
    python infer.py --artifact-dir ./gemma3n-asr-multilingual --skip-evaluation
    python infer.py --artifact-dir ./gemma3n-asr-multilingual --skip-submission
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from common import (
    detect_id_key,
    fmt_duration,
    load_streaming_split,
    login_to_hf,
    make_audio_to_array,
    make_formatters,
    quiet_hf_logging,
)

logger = logging.getLogger("infer")

DEFAULT_SAMPLE_SUBMISSION = (
    "../../data/google-waxal-asr-challenge20260630-10570-elxebu/SampleSubmission.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model / adapter — must match train.py
    p.add_argument(
        "--model-id", default="google/gemma-3n-E2B-it",
        help="Base model id. Must match what train.py fine-tuned.",
    )
    p.add_argument(
        "--artifact-dir", default="./gemma3n-asr-multilingual",
        help="LoRA adapter + processor saved by train.py's --artifact-dir.",
    )

    # Dataset
    p.add_argument("--dataset-id", default="google/WaxalNLP")
    p.add_argument(
        "--languages", nargs="+", default=["lin", "sna", "lug"],
        help="Languages to evaluate/transcribe (default: lin sna lug).",
    )

    # Prompts — MUST match train.py, since the adapter was fit against these
    p.add_argument(
        "--system-message",
        default="You are an assistant that transcribes speech accurately.",
    )
    p.add_argument("--user-message", default="Please transcribe this audio.")

    p.add_argument("--sample-rate", type=int, default=16_000)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--inference-batch-size", type=int, default=8,
        help="Batch size for model.generate() calls.",
    )
    p.add_argument("--progress-every", type=int, default=200)

    # Evaluation (local WER/CER sanity check)
    p.add_argument(
        "--skip-evaluation", action="store_true",
        help="Skip the WER/CER sanity check against WaxalNLP's test split.",
    )
    p.add_argument(
        "--eval-num-samples", type=int, default=200,
        help="Examples per language to evaluate WER/CER on.",
    )

    # Submission generation
    p.add_argument(
        "--skip-submission", action="store_true",
        help="Skip generating submission.csv.",
    )
    p.add_argument(
        "--sample-submission", default=DEFAULT_SAMPLE_SUBMISSION,
        help="Zindi's SampleSubmission.csv — defines the exact IDs/order "
             "the output submission.csv must contain.",
    )
    p.add_argument(
        "--submission-output", default="./submission.csv",
        help="Where to write the generated submission.",
    )

    p.add_argument(
        "--hf-token", default=None,
        help="Hugging Face token. Falls back to the HF_TOKEN env var, then "
             "an interactive login prompt if neither is set.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model + adapter
# ---------------------------------------------------------------------------

def load_model_and_adapter(model_id: str, artifact_dir: str):
    import timm  # noqa: F401  (required by Gemma3n; import before transformers)
    import torch
    import transformers
    import peft

    try:
        processor = transformers.AutoProcessor.from_pretrained(artifact_dir)
    except OSError:
        logger.warning(
            "No processor found in %s, falling back to %s", artifact_dir, model_id
        )
        processor = transformers.AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"

    base_model = transformers.Gemma3nForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = peft.PeftModel.from_pretrained(base_model, artifact_dir)
    model.eval()
    return model, processor


def transcribe_batch(
    batch: dict[str, list[Any]],
    model,
    processor,
    device,
    audio_to_array,
    max_new_tokens: int,
) -> list[str]:
    """Runs inference on a batch and returns decoded transcriptions."""
    import torch

    messages_list = [msgs[:-1] for msgs in batch["messages"]]  # drop assistant turn
    audio_list = [audio_to_array(a) for a in batch["audio"]]

    text_prompts = processor.tokenizer.apply_chat_template(
        messages_list, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text_prompts, audio=audio_list, return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs.input_ids.shape[1]
    decoded = processor.tokenizer.batch_decode(
        outputs[:, input_len:], skip_special_tokens=True
    )
    return [t.strip() for t in decoded]


# ---------------------------------------------------------------------------
# Evaluation (local WER/CER sanity check)
# ---------------------------------------------------------------------------

def evaluate_language(
    lang: str,
    args: argparse.Namespace,
    model,
    processor,
    device,
    audio_to_array,
    format_batch,
) -> dict[str, float]:
    import jiwer

    ds = load_streaming_split(
        args.dataset_id, lang, "test", args.sample_rate
    ).map(format_batch, batched=True, batch_size=32)

    references: list[str] = []
    predictions: list[str] = []
    start = time.monotonic()
    for batch in ds.take(args.eval_num_samples).batch(batch_size=args.inference_batch_size):
        preds = transcribe_batch(
            batch, model, processor, device, audio_to_array, args.max_new_tokens
        )
        references.extend(batch["transcription"])
        predictions.extend(preds)

    refs_lower = [r.lower() for r in references]
    preds_lower = [p.lower() for p in predictions]
    metrics = {
        "wer": jiwer.wer(refs_lower, preds_lower),
        "cer": jiwer.cer(refs_lower, preds_lower),
        "n": len(references),
    }
    logger.info(
        "%s: WER %.2f%% | CER %.2f%% | n=%d | took %s",
        lang, metrics["wer"] * 100, metrics["cer"] * 100, metrics["n"],
        fmt_duration(time.monotonic() - start),
    )
    return metrics


def run_evaluation(args: argparse.Namespace, model, processor, device, audio_to_array, format_batch) -> None:
    logger.info("=== Evaluating WER/CER on WaxalNLP test split (local sanity check) ===")
    all_metrics = {lang: evaluate_language(
        lang, args, model, processor, device, audio_to_array, format_batch
    ) for lang in args.languages}

    mean_wer = sum(m["wer"] for m in all_metrics.values()) / len(all_metrics)
    mean_cer = sum(m["cer"] for m in all_metrics.values()) / len(all_metrics)
    logger.info(
        "Mean over %s: WER %.2f%% | CER %.2f%% | leaderboard-style score "
        "(0.5*WER + 0.5*CER): %.4f",
        args.languages, mean_wer * 100, mean_cer * 100, 0.5 * mean_wer + 0.5 * mean_cer,
    )


# ---------------------------------------------------------------------------
# Submission generation
# ---------------------------------------------------------------------------

def generate_submission(
    args: argparse.Namespace, model, processor, device, audio_to_array, format_batch
) -> None:
    import pandas as pd

    logger.info("=== Generating submission ===")
    sample_sub = pd.read_csv(args.sample_submission)
    wanted_ids = set(sample_sub["ID"].astype(str))
    logger.info("Sample submission requires %d predictions.", len(wanted_ids))

    predictions: dict[str, str] = {}
    id_key: str | None = None
    start = time.monotonic()

    for lang in args.languages:
        ds = load_streaming_split(
            args.dataset_id, lang, "test", args.sample_rate
        ).map(format_batch, batched=True, batch_size=32)

        if id_key is None:
            id_key = detect_id_key(next(iter(ds)), wanted_ids)
            logger.info("Using dataset id field: '%s'", id_key)

        logger.info("Transcribing %s test split ...", lang)
        for batch in ds.batch(batch_size=args.inference_batch_size):
            ids = [str(x) for x in batch[id_key]]
            # Only spend compute on clips the submission actually needs.
            if not any(i in wanted_ids and i not in predictions for i in ids):
                continue
            preds = transcribe_batch(
                batch, model, processor, device, audio_to_array, args.max_new_tokens
            )
            for _id, pred in zip(ids, preds):
                if _id in wanted_ids:
                    predictions[_id] = pred

            n = len(predictions)
            if args.progress_every and n % args.progress_every < args.inference_batch_size:
                elapsed = time.monotonic() - start
                rate = n / elapsed if elapsed > 0 else 0.0
                remaining = len(wanted_ids) - n
                eta = remaining / rate if rate > 0 else float("nan")
                logger.info(
                    "%d/%d matched | elapsed %s | %.2f ex/s | ETA %s",
                    n, len(wanted_ids), fmt_duration(elapsed), rate,
                    fmt_duration(eta) if rate > 0 else "?",
                )

    sample_sub["Target"] = sample_sub["ID"].astype(str).map(predictions).fillna("")
    missing = int((sample_sub["Target"] == "").sum())
    logger.info(
        "Filled %d/%d rows; %d missing.",
        len(sample_sub) - missing, len(sample_sub), missing,
    )
    if missing:
        logger.warning(
            "Some rows are empty — check the id field / test-split coverage "
            "before submitting."
        )

    sample_sub.to_csv(args.submission_output, index=False)
    logger.info("Wrote %s (took %s)", args.submission_output, fmt_duration(time.monotonic() - start))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    import torch

    login_to_hf(args.hf_token)

    logger.info("Loading base model %s + adapter from %s ...", args.model_id, args.artifact_dir)
    model, processor = load_model_and_adapter(args.model_id, args.artifact_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Model ready — device: %s, dtype: %s", device, model.dtype)

    audio_to_array = make_audio_to_array(args.sample_rate)
    _, format_batch = make_formatters(audio_to_array, args.system_message, args.user_message)

    if not args.skip_evaluation:
        run_evaluation(args, model, processor, device, audio_to_array, format_batch)
    else:
        logger.info("Skipping evaluation (--skip-evaluation).")

    if not args.skip_submission:
        generate_submission(args, model, processor, device, audio_to_array, format_batch)
    else:
        logger.info("Skipping submission generation (--skip-submission).")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
