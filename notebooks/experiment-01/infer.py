"""experiment-01 / v1 — inference for the Whisper-large-v3 LoRA adapter.

Two independent jobs (either can be skipped):
  1. EVALUATION (default on): stream the WaxalNLP test split, transcribe it, and
     report WER/CER PER LANGUAGE + macro/micro. This uses the publicly available
     test labels ONLY to measure — never to train, never submitted. (docs/plans §10)
  2. SUBMISSION: transcribe the ids in SampleSubmission.csv and write submission.csv.

Decoding matches Phase-2 conditions: audio only, the model AUTO-DETECTS language
(no id routing), with Whisper hallucination guards (no-repeat n-grams + a
compression-ratio check with a beam-search fallback for degenerate outputs).

    python infer.py --artifact-dir ./artifacts/whisper-large-v3-lora
    python infer.py --artifact-dir ... --skip-submission        # eval only
    python infer.py --artifact-dir ... --skip-evaluation        # submission only
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import time
from pathlib import Path

from common import (
    append_jsonl,
    artifacts_dir,
    data_dir,
    detect_id_key,
    fmt_duration,
    format_metrics,
    load_streaming_split,
    login_to_hf,
    normalize_text,
    per_language_metrics,
    quiet_hf_logging,
    setup_file_logging,
    write_json,
)

logger = logging.getLogger("infer")
_HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="openai/whisper-large-v3")
    p.add_argument("--artifact-dir", default=str(artifacts_dir()))
    p.add_argument("--dataset-id", default="google/WaxalNLP")
    p.add_argument("--languages", nargs="+", default=["lin", "sna", "lug"])
    p.add_argument("--sample-rate", type=int, default=16_000)

    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--num-beams", type=int, default=1,
                   help="Beam width for ALL clips (1 = greedy). Try 5 for a small WER gain.")
    p.add_argument("--inference-batch-size", type=int, default=16)
    p.add_argument("--compression-ratio-threshold", type=float, default=2.4,
                   help="Above this, treat output as a hallucinated repetition and re-decode with beams.")
    p.add_argument("--chunk-length-s", type=float, default=30.0,
                   help="Window length (s, <=30) for clips longer than Whisper's 30 s cap.")
    p.add_argument("--chunk-overlap-s", type=float, default=0.0,
                   help="Overlap between windows (0 = contiguous 'transcribe every 30 s' concatenation).")
    p.add_argument("--progress-every", type=int, default=200)

    # Evaluation
    p.add_argument("--skip-evaluation", action="store_true")
    p.add_argument("--eval-num-samples", type=int, default=400, help="Per-language eval cap (0 = all).")

    # Submission
    p.add_argument("--skip-submission", action="store_true")
    p.add_argument("--sample-submission", default=str(data_dir() / "SampleSubmission.csv"))
    p.add_argument("--submission-output", default=str(_HERE / "submission.csv"))

    p.add_argument("--hf-token", default=None)
    return p.parse_args()


def compression_ratio(text: str) -> float:
    """gzip compression ratio; > ~2.4 signals a looping/hallucinated transcript."""
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / max(1, len(gzip.compress(data)))


def load_model_and_processor(model_id: str, artifact_dir: str):
    import torch
    import peft
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    try:
        processor = WhisperProcessor.from_pretrained(artifact_dir)
    except OSError:
        logger.warning("No processor in %s; falling back to %s", artifact_dir, model_id)
        processor = WhisperProcessor.from_pretrained(model_id)

    base = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model = peft.PeftModel.from_pretrained(base, artifact_dir)
    model.eval()
    # Auto-detect language at generate time (audio-only, matches Phase 2).
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = None
    model.generation_config.task = "transcribe"
    model.config.forced_decoder_ids = None
    return model, processor


def _generate(model, processor, audio_arrays, device, sample_rate, max_new_tokens, num_beams):
    import torch

    feats = processor.feature_extractor(
        audio_arrays, sampling_rate=sample_rate, return_tensors="pt"
    ).input_features.to(device=device, dtype=model.dtype)
    with torch.no_grad():
        out = model.generate(input_features=feats, max_new_tokens=max_new_tokens,
                             num_beams=num_beams, no_repeat_ngram_size=3)
    return processor.tokenizer.batch_decode(out, skip_special_tokens=True)


def _window_bounds(n_samples, win, stride):
    """Yield (start, end) sample offsets covering [0, n_samples) in windows of
    `win` advancing by `stride`. Non-overlapping when stride == win."""
    start = 0
    while start < n_samples:
        yield start, min(start + win, n_samples)
        if start + win >= n_samples:
            break
        start += stride


def _transcribe_long(model, processor, audio, device, args):
    """Whisper caps at 30 s, so a longer clip is split into successive windows,
    each transcribed, and the pieces concatenated (docs/plans §8)."""
    sr = args.sample_rate
    win = int(args.chunk_length_s * sr)
    stride = max(1, int((args.chunk_length_s - args.chunk_overlap_s) * sr))
    windows = [audio[s:e] for s, e in _window_bounds(len(audio), win, stride)]
    pieces = _generate(model, processor, windows, device, sr, args.max_new_tokens, num_beams=args.num_beams)
    return normalize_text(" ".join(p.strip() for p in pieces if p.strip()))


def transcribe_batch(model, processor, audio_arrays, device, args):
    """Transcribe a batch: clips <=30 s go through in one batched pass; clips
    >30 s are windowed and concatenated (_transcribe_long). Then any short clip
    that looks hallucinated (high compression ratio) is re-decoded with beams."""
    max_len = int(args.chunk_length_s * args.sample_rate)
    results: list[str] = [""] * len(audio_arrays)
    short = [i for i, a in enumerate(audio_arrays) if len(a) <= max_len]
    long_ = [i for i, a in enumerate(audio_arrays) if len(a) > max_len]

    if short:
        texts = _generate(model, processor, [audio_arrays[i] for i in short], device,
                          args.sample_rate, args.max_new_tokens, num_beams=args.num_beams)
        for j, i in enumerate(short):
            results[i] = normalize_text(texts[j])
    for i in long_:
        results[i] = _transcribe_long(model, processor, audio_arrays[i], device, args)

    # Hallucination fallback for the single-pass (short) clips.
    bad = [i for i in short if compression_ratio(results[i]) > args.compression_ratio_threshold]
    if bad:
        redo = _generate(model, processor, [audio_arrays[i] for i in bad], device,
                         args.sample_rate, args.max_new_tokens, num_beams=5)
        for j, i in enumerate(bad):
            results[i] = normalize_text(redo[j])
        logger.info("Re-decoded %d degenerate clip(s) with beam search.", len(bad))
    return results


# ---------------------------------------------------------------------------
# Evaluation (eval-only use of the public test labels)
# ---------------------------------------------------------------------------

def run_evaluation(args, model, processor, device, audio_to_array, log_dir):
    logger.info("=== Evaluating WER/CER on WaxalNLP test split (per language) ===")
    all_refs, all_hyps, all_langs = [], [], []
    for lang in args.languages:
        ds = load_streaming_split(args.dataset_id, lang, "test", args.sample_rate)
        if args.eval_num_samples:
            ds = ds.take(args.eval_num_samples)
        refs, hyps = [], []
        start = time.monotonic()
        batch_audio, batch_ref = [], []

        def flush():
            if not batch_audio:
                return
            hyps.extend(transcribe_batch(model, processor, batch_audio, device, args))
            refs.extend(batch_ref)
            batch_audio.clear()
            batch_ref.clear()

        for ex in ds:
            batch_audio.append(audio_to_array(ex["audio"]))
            batch_ref.append(normalize_text(ex["transcription"]))
            if len(batch_audio) >= args.inference_batch_size:
                flush()
        flush()

        all_refs += refs
        all_hyps += hyps
        all_langs += [lang] * len(refs)
        m = per_language_metrics(refs, hyps, [lang] * len(refs))
        logger.info("%s: WER %.3f | CER %.3f | n=%d | %s",
                    lang, m.get(f"wer_{lang}", m["wer"]), m.get(f"cer_{lang}", m["cer"]),
                    len(refs), fmt_duration(time.monotonic() - start))

    metrics = per_language_metrics(all_refs, all_hyps, all_langs)
    logger.info("TEST  %s", format_metrics(metrics))
    write_json(os.path.join(log_dir, "test_eval.json"), metrics)
    append_jsonl(os.path.join(log_dir, "metrics.jsonl"), {"event": "test_eval", "time": time.time(), **metrics})
    return metrics


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def generate_submission(args, model, processor, device, audio_to_array):
    import pandas as pd

    logger.info("=== Generating submission ===")
    sample_sub = pd.read_csv(args.sample_submission)
    wanted_ids = set(sample_sub["ID"].astype(str))
    logger.info("Sample submission requires %d predictions.", len(wanted_ids))

    predictions: dict[str, str] = {}
    start = time.monotonic()
    for lang in args.languages:
        ds = load_streaming_split(args.dataset_id, lang, "test", args.sample_rate)
        id_key = detect_id_key(next(iter(ds)), wanted_ids)
        batch_ids, batch_audio = [], []

        def flush():
            if not batch_audio:
                return
            texts = transcribe_batch(model, processor, batch_audio, device, args)
            for _id, txt in zip(batch_ids, texts):
                if _id in wanted_ids:
                    predictions[_id] = txt
            batch_ids.clear()
            batch_audio.clear()

        for ex in ds:
            _id = str(ex[id_key])
            if _id not in wanted_ids:
                continue
            batch_ids.append(_id)
            batch_audio.append(audio_to_array(ex["audio"]))
            if len(batch_audio) >= args.inference_batch_size:
                flush()
                if args.progress_every and len(predictions) % args.progress_every < args.inference_batch_size:
                    logger.info("%d/%d done | %s", len(predictions), len(wanted_ids),
                                fmt_duration(time.monotonic() - start))
        flush()

    sample_sub["Target"] = sample_sub["ID"].astype(str).map(predictions).fillna("")
    missing = int((sample_sub["Target"] == "").sum())
    logger.info("Filled %d/%d rows; %d missing.", len(sample_sub) - missing, len(sample_sub), missing)
    if missing:
        logger.warning("Some rows empty — must be 0 before submitting.")
    sample_sub.to_csv(args.submission_output, index=False)
    logger.info("Wrote %s (took %s)", args.submission_output, fmt_duration(time.monotonic() - start))


def run(args: argparse.Namespace) -> None:
    import torch

    log_dir = os.path.join(args.artifact_dir, "logs")
    setup_file_logging(log_dir, name="infer")
    login_to_hf(args.hf_token)

    logger.info("Loading base %s + adapter %s ...", args.model_id, args.artifact_dir)
    model, processor = load_model_and_processor(args.model_id, args.artifact_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    from common import make_audio_to_array
    audio_to_array = make_audio_to_array(args.sample_rate)

    if not args.skip_evaluation:
        run_evaluation(args, model, processor, device, audio_to_array, log_dir)
    if not args.skip_submission:
        generate_submission(args, model, processor, device, audio_to_array)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    run(parse_args())


if __name__ == "__main__":
    main()
