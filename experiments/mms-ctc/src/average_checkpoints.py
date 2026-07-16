"""Average the top-K checkpoints by validation `combined`, then keep the result
ONLY if it actually beats the single best checkpoint.

Why: averaging the weights of several good checkpoints lands in a flatter region
of the loss surface, which usually generalises better at **zero extra inference
cost** — and generalisation to unseen speakers is exactly what our speaker-disjoint
val (and Phase 2) measures. Typical gain is small but real (~0.005-0.015 combined).

Run this straight after train_ctc.py, on the same box, BEFORE teardown: it needs
the `checkpoint-*/` dirs, which only exist there (~6 GB each) and are pruned by
--save-total-limit as training rolls on. Averaging cannot be done retroactively.

It is gated on purpose: averaging doesn't always help, and shipping a worse model
because the technique is fashionable would be silly. We measure, then decide.

    python average_checkpoints.py                  # top 3, evaluate, keep if better
    python average_checkpoints.py --top-k 4 --no-evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re

from common import artifacts_dir, fmt_duration, per_language_metrics, quiet_hf_logging, write_json

logger = logging.getLogger("average_checkpoints")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact-dir", default=str(artifacts_dir()))
    p.add_argument("--out-dir", default=None, help="Default: <artifact-dir>-avg")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True,
                   help="Score the average on the full speaker-disjoint val set and only keep it "
                        "if it beats the single best checkpoint.")
    p.add_argument("--per-device-eval-batch-size", type=int, default=4)
    return p.parse_args()


def rank_checkpoints(artifact_dir: str) -> list[tuple[int, str, float]]:
    """[(step, path, combined)] best-first, read from the trainer's own log history."""
    hist_path = os.path.join(artifact_dir, "logs", "trainer_log_history.json")
    step_score: dict[int, float] = {}
    if os.path.exists(hist_path):
        for r in json.load(open(hist_path)):
            if "eval_combined" in r and r.get("step") is not None:
                step_score[int(r["step"])] = float(r["eval_combined"])
    out = []
    for d in sorted(os.listdir(artifact_dir)):
        m = re.fullmatch(r"checkpoint-(\d+)", d)
        p = os.path.join(artifact_dir, d)
        if m and os.path.isdir(p):
            out.append((int(m.group(1)), p, step_score.get(int(m.group(1)), float("inf"))))
    out.sort(key=lambda t: t[2])
    return out


def average_state_dicts(paths: list[str]):
    """Uniform average of the model weights. Non-float buffers are taken from the
    first checkpoint (averaging ints/masks is meaningless)."""
    import torch
    from safetensors.torch import load_file

    def load(p):
        st = os.path.join(p, "model.safetensors")
        if os.path.exists(st):
            return load_file(st)
        return torch.load(os.path.join(p, "pytorch_model.bin"), map_location="cpu")

    avg = None
    for i, p in enumerate(paths):
        sd = load(p)
        if avg is None:
            avg = {k: (v.clone().float() if v.is_floating_point() else v.clone())
                   for k, v in sd.items()}
            continue
        for k, v in sd.items():
            if k in avg and avg[k].is_floating_point():
                avg[k] += v.float()
        del sd
    n = len(paths)
    for k, v in avg.items():
        if v.is_floating_point():
            avg[k] /= n
    return avg


def evaluate(model_dir: str, args) -> dict:
    """Full speaker-disjoint val, reusing the exact Stage-2 data path + metrics."""
    import numpy as np
    import torch
    from transformers import Wav2Vec2ForCTC

    import train_ctc as T

    targs = T.parse_args.__wrapped__() if hasattr(T.parse_args, "__wrapped__") else None
    import sys
    sys.argv = ["train_ctc.py"]
    a = T.parse_args()
    a.per_device_eval_batch_size = args.per_device_eval_batch_size

    processor = T.build_processor(a)
    val = T.load_split(a, "val")
    langs = list(val["language"])
    model = Wav2Vec2ForCTC.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    collator = T.CTCCollator(processor=processor, sample_rate=a.sample_rate)

    hyps, refs = [], []
    bs = args.per_device_eval_batch_size
    pid = processor.tokenizer.pad_token_id
    for i in range(0, len(val), bs):
        batch = collator([val[j] for j in range(i, min(i + bs, len(val)))])
        with torch.no_grad():
            logits = model(input_values=batch["input_values"].to(device),
                           attention_mask=batch.get("attention_mask").to(device)
                           if batch.get("attention_mask") is not None else None).logits
        ids = torch.argmax(logits, dim=-1).cpu().numpy()
        hyps.extend(processor.batch_decode(ids))
        lab = batch["labels"].numpy()
        refs.extend(processor.batch_decode(np.where(lab == -100, pid, lab), group_tokens=False))
    return per_language_metrics(refs, hyps, langs[:len(refs)])


def main() -> None:
    import time

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()
    out_dir = args.out_dir or f"{args.artifact_dir.rstrip('/')}-avg"

    ranked = rank_checkpoints(args.artifact_dir)
    if not ranked:
        raise SystemExit(f"No checkpoint-*/ under {args.artifact_dir}. Was --save-total-limit too "
                         f"low, or did you already tear the box down? Averaging needs them.")
    logger.info("Checkpoints (best first by eval combined):")
    for step, _, sc in ranked:
        logger.info("  checkpoint-%-6d combined=%s", step, "n/a" if sc == float("inf") else f"{sc:.4f}")

    picked = ranked[:args.top_k]
    if len(picked) < 2:
        raise SystemExit("Need >=2 checkpoints to average.")
    logger.info("Averaging top %d: %s", len(picked), [f"checkpoint-{s}" for s, _, _ in picked])

    start = time.monotonic()
    avg = average_state_dicts([p for _, p, _ in picked])

    # Save as a normal model dir (config + processor copied from the best checkpoint).
    import shutil

    import torch
    os.makedirs(out_dir, exist_ok=True)
    best_dir = picked[0][1]
    for f in ("config.json", "preprocessor_config.json", "tokenizer_config.json",
              "vocab.json", "special_tokens_map.json", "added_tokens.json"):
        for src in (os.path.join(best_dir, f), os.path.join(args.artifact_dir, f)):
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, f))
                break
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in avg.items()},
              os.path.join(out_dir, "model.safetensors"))
    logger.info("Averaged model -> %s (%s)", out_dir, fmt_duration(time.monotonic() - start))

    if not args.evaluate:
        logger.info("Skipping evaluation (--no-evaluate). Verify before you ship it.")
        return

    logger.info("Scoring the AVERAGE on the full speaker-disjoint val ...")
    avg_m = evaluate(out_dir, args)
    logger.info("Scoring the SINGLE BEST (checkpoint-%d) for comparison ...", picked[0][0])
    best_m = evaluate(best_dir, args)

    delta = avg_m["combined"] - best_m["combined"]
    verdict = "KEEP the average" if delta < 0 else "DISCARD the average — single best wins"
    logger.info("single best : combined %.4f | WER %.3f CER %.3f", best_m["combined"],
                best_m["wer_macro"], best_m["cer_macro"])
    logger.info("averaged    : combined %.4f | WER %.3f CER %.3f", avg_m["combined"],
                avg_m["wer_macro"], avg_m["cer_macro"])
    logger.info("delta %+.4f  ==> %s", delta, verdict)
    write_json(os.path.join(out_dir, "average_report.json"),
               {"checkpoints": [{"step": s, "combined": c} for s, _, c in picked],
                "single_best": best_m, "averaged": avg_m, "delta_combined": delta,
                "verdict": verdict})


if __name__ == "__main__":
    main()
