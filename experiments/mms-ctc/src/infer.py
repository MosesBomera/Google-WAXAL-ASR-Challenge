"""experiment mms-ctc / inference — local eval on the test split + submission.csv.

Uses DIRECT per-clip CTC decode (load model -> logits -> argmax -> collapse), the
same path `average_checkpoints.py` used to measure combined 0.205 — NOT the
training script's `trainer.predict` path, which over-counted WER by ~0.013 via
its cross-batch padding. This is exactly what a submission does, so the local
number here should track the board.

Greedy decoding; the KenLM word-boundary rescorer is Stage 3 (it plugs into the
same transcription loop). CTC has no 30 s cap, so clips are decoded whole; a
safety net chunks anything absurdly long to avoid an OOM crash mid-submission.

    # eval on the public test labels (EVALUATION ONLY) + write submission:
    python infer.py --model mosesbomera/waxal-mms-1b-ctc
    python infer.py --model artifacts/mms-1b-all --skip-submission     # just the local number
    python infer.py --model mosesbomera/waxal-mms-1b-ctc --skip-evaluation   # just submission.csv
"""

from __future__ import annotations

import argparse
import logging
import time

from common import (
    artifacts_dir,
    data_dir,
    detect_id_key,
    fmt_duration,
    load_streaming_split,
    login_to_hf,
    make_audio_to_array,
    normalize_for_ctc,
    per_language_metrics,
    quiet_hf_logging,
    setup_file_logging,
    workdir,
    write_json,
)

logger = logging.getLogger("infer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(artifacts_dir()),
                   help="Local dir (default: artifacts/mms-1b-all, where train_ctc.py saved it) OR "
                        "an HF Hub repo id like mosesbomera/waxal-mms-1b-ctc. Must contain the "
                        "model + processor (cased vocab).")
    p.add_argument("--dataset-id", default="google/WaxalNLP")
    p.add_argument("--languages", nargs="+", default=["lin", "sna", "lug"])
    p.add_argument("--sample-rate", type=int, default=16_000)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                   help="float32 reproduces the 0.205 eval exactly; bf16/fp16 are ~2x faster, ~same WER.")
    p.add_argument("--batch-size", type=int, default=8)

    # KenLM beam-search decoding (Stage 3) — the WER lever. Greedy if --kenlm unset.
    p.add_argument("--kenlm", default=None, help="Path to a KenLM .arpa/.bin (from build_lm.py). "
                                                 "Switches greedy -> beam search with the LM.")
    p.add_argument("--lm-alpha", type=float, default=0.5, help="LM weight (higher = trust the LM more).")
    p.add_argument("--lm-beta", type=float, default=1.5, help="Word-insertion bonus.")
    p.add_argument("--beam-width", type=int, default=100)

    # long-clip safety net (rarely fires: train max was 46 s, and CTC has no cap)
    p.add_argument("--max-chunk-sec", type=float, default=60.0,
                   help="Clips longer than this are split into windows to avoid OOM. 0 = never chunk.")
    p.add_argument("--chunk-window-sec", type=float, default=40.0)

    # evaluation (public test labels — EVALUATION ONLY, never trained on)
    p.add_argument("--skip-evaluation", action="store_true")
    p.add_argument("--eval-num-samples", type=int, default=0, help="Per-language cap; 0 = all.")

    # submission
    p.add_argument("--skip-submission", action="store_true")
    p.add_argument("--sample-submission", default=str(data_dir() / "SampleSubmission.csv"))
    p.add_argument("--submission-output", default=str(workdir() / "submission.csv"))

    p.add_argument("--hf-token", default=None)
    return p.parse_args()


def load_model_and_processor(model_ref: str, dtype_str: str):
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]
    processor = Wav2Vec2Processor.from_pretrained(model_ref)
    model = Wav2Vec2ForCTC.from_pretrained(model_ref, torch_dtype=dtype)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Loaded %s (%s) on %s | vocab %d, blank id %d",
                model_ref, dtype_str, device, len(processor.tokenizer), processor.tokenizer.pad_token_id)
    return model, processor, device


def build_lm_processor(processor, args):
    """Wrap the tokenizer + a KenLM-backed pyctcdecode beam decoder into a
    Wav2Vec2ProcessorWithLM. That class knows the tokenizer's blank ([PAD]) and
    word-delimiter ('|') conventions, so we don't hand-map them. Returns None if
    --kenlm is unset (greedy)."""
    if not args.kenlm:
        return None
    from pyctcdecode import build_ctcdecoder
    from transformers import Wav2Vec2ProcessorWithLM

    vocab = processor.tokenizer.get_vocab()
    labels = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]  # class-id order
    decoder = build_ctcdecoder(labels, kenlm_model_path=args.kenlm,
                               alpha=args.lm_alpha, beta=args.lm_beta)
    logger.info("KenLM decoder: %s | alpha=%.2f beta=%.2f beam=%d",
                args.kenlm, args.lm_alpha, args.lm_beta, args.beam_width)
    return Wav2Vec2ProcessorWithLM(feature_extractor=processor.feature_extractor,
                                   tokenizer=processor.tokenizer, decoder=decoder)


# ---------------------------------------------------------------------------
# Direct CTC decode (the trustworthy path)
# ---------------------------------------------------------------------------

def _decode(model, processor, audio_arrays, device, sample_rate, lm_processor=None, beam_width=100):
    import torch

    inputs = processor(audio_arrays, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device=device, dtype=model.dtype)
    attn = inputs.get("attention_mask")
    attn = attn.to(device) if attn is not None else None
    with torch.no_grad():
        logits = model(input_values, attention_mask=attn).logits
    if lm_processor is None:
        texts = processor.batch_decode(torch.argmax(logits, dim=-1))   # greedy CTC collapse
    else:
        # KenLM beam search over the (fp32) logits; ProcessorWithLM handles blank/delimiter.
        texts = lm_processor.batch_decode(logits.float().cpu().numpy(), beam_width=beam_width).text
    return [normalize_for_ctc(t) for t in texts]


def _is_oom(e) -> bool:
    import torch

    return isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()


def _safe_decode(model, processor, audio_arrays, device, args):
    """_decode, but if the batch OOMs, empty the cache and retry on halves — so a
    stray long clip (padded x batch_size) can never crash the whole submission."""
    import torch

    try:
        return _decode(model, processor, audio_arrays, device, args.sample_rate,
                       lm_processor=getattr(args, "_lm_processor", None), beam_width=args.beam_width)
    except RuntimeError as e:            # torch.cuda.OutOfMemoryError subclasses RuntimeError
        if not _is_oom(e):
            raise
        torch.cuda.empty_cache()
        if len(audio_arrays) <= 1:
            raise                         # a single clip OOM is a real problem — surface it
        mid = len(audio_arrays) // 2
        logger.warning("CUDA OOM on a batch of %d clips; halving and retrying "
                       "(pass --dtype bfloat16 or a smaller --batch-size to avoid this).",
                       len(audio_arrays))
        return (_safe_decode(model, processor, audio_arrays[:mid], device, args)
                + _safe_decode(model, processor, audio_arrays[mid:], device, args))


def _decode_long(model, processor, audio, device, args):
    """Safety net for absurdly long clips: split into windows, decode, concatenate."""
    sr = args.sample_rate
    win = int(args.chunk_window_sec * sr)
    windows = [audio[s:s + win] for s in range(0, len(audio), win)]
    pieces = _safe_decode(model, processor, windows, device, args)
    return normalize_for_ctc(" ".join(p for p in pieces if p))


def transcribe_batch(model, processor, audio_arrays, device, args):
    max_len = int(args.max_chunk_sec * args.sample_rate) if args.max_chunk_sec else None
    short = [i for i, a in enumerate(audio_arrays) if max_len is None or len(a) <= max_len]
    long_ = [i for i, a in enumerate(audio_arrays) if max_len is not None and len(a) > max_len]

    results: list[str] = [""] * len(audio_arrays)
    if short:
        for j, txt in zip(short, _decode(model, processor, [audio_arrays[i] for i in short],
                                         device, args.sample_rate)):
            results[j] = txt
    for i in long_:
        results[i] = _decode_long(model, processor, audio_arrays[i], device, args)
        logger.info("Chunk-decoded a %.0f s clip.", len(audio_arrays[i]) / args.sample_rate)
    return results


# ---------------------------------------------------------------------------
# Evaluation (public test labels — EVALUATION ONLY)
# ---------------------------------------------------------------------------

def run_evaluation(args, model, processor, device, audio_to_array, log_dir):
    logger.info("=== Local eval on the WaxalNLP test split (per language; evaluation only) ===")
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
            hyps.extend(transcribe_batch(model, processor, list(batch_audio), device, args))
            refs.extend(batch_ref)
            batch_audio.clear()
            batch_ref.clear()

        for ex in ds:
            batch_audio.append(audio_to_array(ex["audio"]))
            batch_ref.append(normalize_for_ctc(ex["transcription"]))
            if len(batch_audio) >= args.batch_size:
                flush()
        flush()

        all_refs += refs
        all_hyps += hyps
        all_langs += [lang] * len(refs)
        m = per_language_metrics(refs, hyps, [lang] * len(refs))
        logger.info("%s: WER %.3f | CER %.3f | n=%d | %s", lang,
                    m.get(f"wer_{lang}", m["wer"]), m.get(f"cer_{lang}", m["cer"]),
                    len(refs), fmt_duration(time.monotonic() - start))

    metrics = per_language_metrics(all_refs, all_hyps, all_langs)
    logger.info("TEST  combined(macro) %.4f | WER %.3f CER %.3f | lin W%.3f/C%.3f lug W%.3f/C%.3f "
                "sna W%.3f/C%.3f", metrics["combined"], metrics["wer_macro"], metrics["cer_macro"],
                metrics.get("wer_lin", 0), metrics.get("cer_lin", 0), metrics.get("wer_lug", 0),
                metrics.get("cer_lug", 0), metrics.get("wer_sna", 0), metrics.get("cer_sna", 0))
    write_json(f"{log_dir}/test_eval.json", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def generate_submission(args, model, processor, device, audio_to_array):
    import pandas as pd

    logger.info("=== Generating submission ===")
    sample_sub = pd.read_csv(args.sample_submission)
    wanted = set(sample_sub["ID"].astype(str))
    logger.info("Sample submission requires %d predictions.", len(wanted))

    predictions: dict[str, str] = {}
    start = time.monotonic()
    for lang in args.languages:
        ds = load_streaming_split(args.dataset_id, lang, "test", args.sample_rate)
        id_key = detect_id_key(next(iter(ds)), wanted)
        ds = load_streaming_split(args.dataset_id, lang, "test", args.sample_rate)
        batch_ids, batch_audio = [], []

        def flush():
            if not batch_audio:
                return
            for _id, txt in zip(batch_ids, transcribe_batch(model, processor, list(batch_audio),
                                                             device, args)):
                if _id in wanted:
                    predictions[_id] = txt
            batch_ids.clear()
            batch_audio.clear()

        for ex in ds:
            _id = str(ex[id_key])
            if _id not in wanted:
                continue
            batch_ids.append(_id)
            batch_audio.append(audio_to_array(ex["audio"]))
            if len(batch_audio) >= args.batch_size:
                flush()
                if len(predictions) % 500 < args.batch_size:
                    logger.info("%d/%d done | %s", len(predictions), len(wanted),
                                fmt_duration(time.monotonic() - start))
        flush()

    sample_sub["Target"] = sample_sub["ID"].astype(str).map(predictions).fillna("")
    missing = int((sample_sub["Target"] == "").sum())
    logger.info("Filled %d/%d rows; %d missing.", len(sample_sub) - missing, len(sample_sub), missing)
    if missing:
        logger.warning("Some rows empty — must be 0 before submitting.")
    sample_sub.to_csv(args.submission_output, index=False)
    logger.info("Wrote %s (took %s)", args.submission_output, fmt_duration(time.monotonic() - start))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()
    log_dir = str(workdir() / "artifacts" / "infer-logs")
    setup_file_logging(log_dir, name="infer")
    login_to_hf(args.hf_token)

    model, processor, device = load_model_and_processor(args.model, args.dtype)
    args._lm_processor = build_lm_processor(processor, args)   # None unless --kenlm given
    audio_to_array = make_audio_to_array(args.sample_rate)
    logger.info("Decoding: %s", "KenLM beam search" if args._lm_processor else "greedy")

    if not args.skip_evaluation:
        run_evaluation(args, model, processor, device, audio_to_array, log_dir)
    if not args.skip_submission:
        generate_submission(args, model, processor, device, audio_to_array)

    import logging as _l
    _l.shutdown()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
