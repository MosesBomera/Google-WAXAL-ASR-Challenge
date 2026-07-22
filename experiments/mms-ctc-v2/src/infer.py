"""mms-ctc-v2 / inference — authoritative val eval, public-test eval, and submission.

DIRECT per-clip CTC decode (load -> logits -> argmax -> collapse) — the trustworthy path
(trainer.predict over-counts WER ~0.01 via cross-batch padding). Greedy only: v1 proved KenLM
adds nothing (the model is too confident), so it's gone.

Three jobs:
  * --eval-set val  (default): the YARDSTICK — our speaker-disjoint val (prepared/ + manifests/
    val.csv), board-faithful metric. Run for BOTH v1 and v2 models to get a clean val delta.
  * --eval-set test: the public test labels (streamed) — the board proxy (v1 = 0.2087 ≈ board).
  * submission.csv from the test audio (corruption gaps filled with a neutral placeholder).

    python infer.py --model artifacts/mms-1b-all --dtype bfloat16 --skip-submission        # v2 val
    python infer.py --model mosesbomera/waxal-mms-1b-ctc --eval-set val --skip-submission   # v1 val (compare)
    python infer.py --model artifacts/mms-1b-all --eval-set test                            # board proxy + submission
"""

from __future__ import annotations

import argparse
import logging
import os
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
    normalize_text,
    per_language_metrics,
    prepared_dir,
    quiet_hf_logging,
    setup_file_logging,
    workdir,
    write_json,
)

logger = logging.getLogger("infer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(artifacts_dir()),
                   help="Local dir (default: artifacts/mms-1b-all) OR a Hub repo id.")
    p.add_argument("--dataset-id", default="google/WaxalNLP")
    p.add_argument("--languages", nargs="+", default=["lin", "sna", "lug"])
    p.add_argument("--sample-rate", type=int, default=16_000)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--batch-size", type=int, default=8)

    # evaluation
    p.add_argument("--eval-set", choices=["val", "test", "none"], default="val",
                   help="val = our speaker-disjoint yardstick (needs prepared/ + manifests/val.csv). "
                        "test = public test labels (streamed, EVALUATION ONLY).")
    p.add_argument("--audio-dir", default=str(prepared_dir()))
    p.add_argument("--manifest-dir", default=str(workdir() / "manifests"))
    p.add_argument("--eval-num-samples", type=int, default=0, help="Per-language cap; 0 = all.")

    # long-clip safety net
    p.add_argument("--max-chunk-sec", type=float, default=60.0, help="0 = never chunk.")
    p.add_argument("--chunk-window-sec", type=float, default=40.0)

    # submission
    p.add_argument("--skip-submission", action="store_true")
    p.add_argument("--sample-submission", default=str(data_dir() / "SampleSubmission.csv"))
    p.add_argument("--submission-output", default=str(workdir() / "submission.csv"))
    p.add_argument("--empty-placeholder", default="a",
                   help="Fills corruption-gap rows the model emits empty for. NEVER the reference "
                        "text (using test labels = disqualification).")

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


# ---------------------------------------------------------------------------
# Direct CTC decode (greedy)
# ---------------------------------------------------------------------------

def _decode(model, processor, audio_arrays, device, sample_rate):
    import torch

    inputs = processor(audio_arrays, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device=device, dtype=model.dtype)
    attn = inputs.get("attention_mask")
    attn = attn.to(device) if attn is not None else None
    with torch.no_grad():
        logits = model(input_values, attention_mask=attn).logits
    texts = processor.batch_decode(torch.argmax(logits, dim=-1))     # greedy CTC collapse
    return [normalize_for_ctc(t) for t in texts]


def _is_oom(e) -> bool:
    import torch
    return isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()


def _safe_decode(model, processor, audio_arrays, device, args):
    import torch
    try:
        return _decode(model, processor, audio_arrays, device, args.sample_rate)
    except RuntimeError as e:
        if not _is_oom(e):
            raise
        torch.cuda.empty_cache()
        if len(audio_arrays) <= 1:
            raise
        mid = len(audio_arrays) // 2
        logger.warning("CUDA OOM on %d clips; halving (use --dtype bfloat16 or smaller --batch-size).",
                       len(audio_arrays))
        return (_safe_decode(model, processor, audio_arrays[:mid], device, args)
                + _safe_decode(model, processor, audio_arrays[mid:], device, args))


def _decode_long(model, processor, audio, device, args):
    win = int(args.chunk_window_sec * args.sample_rate)
    windows = [audio[s:s + win] for s in range(0, len(audio), win)]
    pieces = _safe_decode(model, processor, windows, device, args)
    return normalize_for_ctc(" ".join(p for p in pieces if p))


def transcribe_batch(model, processor, audio_arrays, device, args):
    max_len = int(args.max_chunk_sec * args.sample_rate) if args.max_chunk_sec else None
    short = [i for i, a in enumerate(audio_arrays) if max_len is None or len(a) <= max_len]
    long_ = [i for i, a in enumerate(audio_arrays) if max_len is not None and len(a) > max_len]
    results: list[str] = [""] * len(audio_arrays)
    if short:
        for j, txt in zip(short, _safe_decode(model, processor, [audio_arrays[i] for i in short],
                                              device, args)):
            results[j] = txt
    for i in long_:
        results[i] = _decode_long(model, processor, audio_arrays[i], device, args)
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _report(tag, metrics, log_dir):
    logger.info("%s  combined %.4f (board-faithful) | WER %.3f CER %.3f | "
                "lin W%.3f/C%.3f lug W%.3f/C%.3f sna W%.3f/C%.3f", tag,
                metrics["combined"], metrics["wer"], metrics["cer"],
                metrics.get("wer_lin", 0), metrics.get("cer_lin", 0), metrics.get("wer_lug", 0),
                metrics.get("cer_lug", 0), metrics.get("wer_sna", 0), metrics.get("cer_sna", 0))
    write_json(os.path.join(log_dir, f"eval_{tag.lower()}.json"), metrics)


def run_val_evaluation(args, model, processor, device, audio_to_array, log_dir):
    """THE yardstick: our speaker-disjoint val, decoded from prepared/ chunks."""
    import datasets
    import pandas as pd

    man = pd.read_csv(os.path.join(args.manifest_dir, "val.csv"), dtype={"id": str})
    val_ids = set(man["id"])
    logger.info("=== Val eval (speaker-disjoint yardstick): %d clips ===", len(val_ids))

    parts = []
    for corpus_split in ("train", "validation"):
        d = os.path.join(args.audio_dir, corpus_split)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isdir(p):
                parts.append(datasets.load_from_disk(p))
    if not parts:
        raise FileNotFoundError(f"No prepared chunks under {args.audio_dir} — need them for val eval.")
    val = datasets.concatenate_datasets(parts).filter(lambda ex: ex["id"] in val_ids, num_proc=1)

    refs, hyps, langs = [], [], []
    batch_audio, batch_ref, batch_lang = [], [], []
    start = time.monotonic()

    def flush():
        if not batch_audio:
            return
        hyps.extend(transcribe_batch(model, processor, list(batch_audio), device, args))
        refs.extend(batch_ref); langs.extend(batch_lang)
        batch_audio.clear(); batch_ref.clear(); batch_lang.clear()

    for ex in val:
        batch_audio.append(audio_to_array(ex["audio"]))
        # RAW ref (board-faithful) — the model's hyp is char-mapped, but references are not;
        # normalize_for_ctc'ing the ref here would hide the char-map penalty the board applies.
        batch_ref.append(normalize_text(ex["transcription"]))
        batch_lang.append(ex["language"])
        if len(batch_audio) >= args.batch_size:
            flush()
    flush()

    metrics = per_language_metrics(refs, hyps, langs)
    logger.info("(val decoded %d clips in %s)", len(refs), fmt_duration(time.monotonic() - start))
    _report("VAL", metrics, log_dir)
    return metrics


def run_test_evaluation(args, model, processor, device, audio_to_array, log_dir):
    """Public test labels (streamed) — the board proxy. EVALUATION ONLY."""
    logger.info("=== Test eval on the WaxalNLP test split (board proxy; evaluation only) ===")
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
            batch_audio.clear(); batch_ref.clear()

        for ex in ds:
            batch_audio.append(audio_to_array(ex["audio"]))
            batch_ref.append(normalize_text(ex["transcription"]))   # RAW ref (board-faithful)
            if len(batch_audio) >= args.batch_size:
                flush()
        flush()

        all_refs += refs; all_hyps += hyps; all_langs += [lang] * len(refs)
        m = per_language_metrics(refs, hyps, [lang] * len(refs))
        logger.info("%s: WER %.3f | CER %.3f | n=%d | %s", lang,
                    m.get(f"wer_{lang}", m["wer"]), m.get(f"cer_{lang}", m["cer"]),
                    len(refs), fmt_duration(time.monotonic() - start))

    metrics = per_language_metrics(all_refs, all_hyps, all_langs)
    _report("TEST", metrics, log_dir)
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
        id_key = detect_id_key(next(iter(load_streaming_split(args.dataset_id, lang, "test",
                                                              args.sample_rate))), wanted)
        ds = load_streaming_split(args.dataset_id, lang, "test", args.sample_rate)
        batch_ids, batch_audio = [], []

        def flush():
            if not batch_audio:
                return
            for _id, txt in zip(batch_ids, transcribe_batch(model, processor, list(batch_audio),
                                                             device, args)):
                if _id in wanted:
                    predictions[_id] = txt
            batch_ids.clear(); batch_audio.clear()

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

    target = sample_sub["ID"].astype(str).map(predictions).fillna("")
    empties = target.str.strip() == ""
    n_empty = int(empties.sum())
    # Fill corruption-gap rows with a neutral placeholder (NEVER the reference text).
    target = target.where(~empties, args.empty_placeholder)
    sample_sub["Target"] = target
    logger.info("Filled %d/%d rows; %d were empty (corruption gaps) -> placeholder %r.",
                len(sample_sub), len(sample_sub), n_empty, args.empty_placeholder)
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
    audio_to_array = make_audio_to_array(args.sample_rate)
    logger.info("Decoding: greedy")

    if args.eval_set == "val":
        run_val_evaluation(args, model, processor, device, audio_to_array, log_dir)
    elif args.eval_set == "test":
        run_test_evaluation(args, model, processor, device, audio_to_array, log_dir)
    if not args.skip_submission:
        generate_submission(args, model, processor, device, audio_to_array)

    logging.shutdown()
    os._exit(0)   # HF streaming leaves connections that can hang on teardown


if __name__ == "__main__":
    main()
