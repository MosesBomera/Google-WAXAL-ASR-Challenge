"""experiment-01 / v1 — Fine-tune openai/whisper-large-v3 for multilingual ASR
(Lingala + Shona + Luganda) with LoRA, selecting checkpoints by WER/CER.

What this does that experiment-00 did not (see docs/plans/experiment-01.md):
  * Whisper-large-v3 (speech-specialized) instead of Gemma 3n.
  * Correct sequence lengths (max_target_length ~256) — no label truncation.
  * LoRA r=32/alpha=32 on encoder AND decoder attention + MLP.
  * Per-example language token: Lingala->ln, Shona->sn, Luganda->sw (no native
    Luganda token; sw is the proxy).
  * Language-balanced (temperature) sampling to counter the Luganda imbalance.
  * Evaluation + checkpoint selection by WER/CER (not loss/token-accuracy),
    reported PER LANGUAGE.
  * Training logs persisted to disk (file log + JSONL metrics stream + the
    trainer log history), so nothing is lost when training ends.

Run prepare_data.py first, then:
    python train.py --data-dir ./data/waxal-multilingual
    python train.py --data-dir ./data/waxal-multilingual --num-train-epochs 3
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time
from typing import Any

from common import (
    LANGUAGE_TO_WHISPER,
    append_jsonl,
    artifacts_dir,
    build_language_balanced_indices,
    fmt_duration,
    format_metrics,
    login_to_hf,
    normalize_text,
    per_language_metrics,
    prepared_dir,
    quiet_hf_logging,
    setup_file_logging,
    write_json,
    zip_best_checkpoints,
)

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model
    p.add_argument("--model-id", default="openai/whisper-large-v3",
                   help="openai/whisper-large-v3 (A100) or -v3-turbo for fast iteration.")

    # Data
    p.add_argument("--data-dir", default=str(prepared_dir()),
                   help="Chunks from prepare_data.py: <data-dir>/{train,validation}/<lang>/ "
                        "(default: ./prepared/waxal-multilingual).")
    p.add_argument("--sample-rate", type=int, default=16_000)
    p.add_argument("--max-duration-sec", type=float, default=30.0,
                   help="Drop training clips longer than this (Whisper's 30 s cap).")

    # Artifacts
    p.add_argument("--artifact-dir", default=str(artifacts_dir()),
                   help="Where the LoRA adapter + processor + logs are saved (default: "
                        "./artifacts/whisper-large-v3-lora). Point infer.py here.")

    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32,
                   help="Ratio alpha/r ~1.0 (not 2x): high ratios trigger Whisper-LoRA hallucinations.")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                   help="Matched by suffix across encoder AND decoder layers.")

    # Optimisation
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--lr-scheduler-type", default="linear")
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--per-device-eval-batch-size", type=int, default=16,
                   help="Eval generation has no gradients, so this can exceed the train batch.")
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)  # eff. batch 16
    p.add_argument("--dataloader-num-workers", type=int, default=4)

    # Sequence lengths (kill the truncation bug)
    p.add_argument("--max-target-length", type=int, default=256)
    p.add_argument("--generation-max-length", type=int, default=256)

    # Luganda imbalance
    p.add_argument("--language-sampling-alpha", type=float, default=0.5,
                   help="Temperature for language sampling: 1=natural, 0=uniform, 0.5=sweet spot.")
    p.add_argument("--luganda-language", default="swahili",
                   help="Whisper language used as the proxy token for Luganda (no native lg token).")

    # Eval / checkpointing
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--save-total-limit", type=int, default=5,
                   help="Keep the last N checkpoints for later checkpoint-averaging.")
    p.add_argument("--zip-top-n", type=int, default=3,
                   help="At the end, zip the best-N checkpoints (+processor+logs) for download "
                        "BEFORE you destroy the instance. 0 to skip.")
    p.add_argument("--max-eval-samples-per-language", type=int, default=300,
                   help="Balanced eval subset per language for speed (generation is slow).")

    # Augmentation (off in v1; wired for later versions)
    p.add_argument("--apply-spec-augment", action="store_true", default=False)

    p.add_argument("--hf-token", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_split(data_dir: str, split: str):
    import datasets

    split_dir = os.path.join(data_dir, split)
    lang_dirs = sorted(os.path.join(split_dir, n) for n in os.listdir(split_dir)
                       if os.path.isdir(os.path.join(split_dir, n)))
    parts = [datasets.load_from_disk(p) for p in lang_dirs]
    return datasets.concatenate_datasets(parts)


def filter_by_duration(ds, sample_rate: int, max_sec: float, log_path: str | None = None):
    """Drops clips longer than max_sec (Whisper's 30 s cap) and logs the damage
    PER LANGUAGE + the duration distribution of what was dropped — so we can
    decide how to recover these in v2 (forced-alignment chunking, see docs/plans)."""
    from collections import Counter

    import numpy as np

    kept_idx: list[int] = []
    dropped_by_lang: Counter = Counter()
    kept_by_lang: Counter = Counter()
    dropped_durs: list[float] = []
    for i, ex in enumerate(ds):
        dur = len(ex["audio"]["array"]) / sample_rate
        lang = ex["language"]
        if dur <= max_sec:
            kept_idx.append(i)
            kept_by_lang[lang] += 1
        else:
            dropped_by_lang[lang] += 1
            dropped_durs.append(dur)

    n0 = len(ds)
    if not dropped_durs:
        logger.info("Duration filter (>%.0fs): nothing dropped.", max_sec)
        return ds

    d = np.array(dropped_durs)
    report = {
        "max_sec": max_sec, "total": n0, "dropped": len(dropped_durs),
        "dropped_by_language": {l: dropped_by_lang[l] for l in sorted(dropped_by_lang)},
        "kept_by_language": {l: kept_by_lang[l] for l in sorted(kept_by_lang)},
        "dropped_duration_s": {
            "median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
            "max": float(d.max()),
            "bucket_30_45": int(((d > 30) & (d <= 45)).sum()),
            "bucket_45_60": int(((d > 45) & (d <= 60)).sum()),
            "bucket_60_plus": int((d > 60).sum()),
        },
    }
    logger.info("Duration filter (>%.0fs): dropped %d/%d clips.", max_sec, len(dropped_durs), n0)
    for l in sorted(set(dropped_by_lang) | set(kept_by_lang)):
        tot = kept_by_lang[l] + dropped_by_lang[l]
        logger.info("  %s: dropped %d/%d (%.1f%%)", l, dropped_by_lang[l], tot,
                    100 * dropped_by_lang[l] / max(1, tot))
    ds_stats = report["dropped_duration_s"]
    logger.info("  dropped duration s: median %.1f, p90 %.1f, max %.1f | "
                "30-45s:%d  45-60s:%d  60s+:%d", ds_stats["median"], ds_stats["p90"],
                ds_stats["max"], ds_stats["bucket_30_45"], ds_stats["bucket_45_60"],
                ds_stats["bucket_60_plus"])
    if log_path:
        write_json(log_path, report)
    return ds.select(kept_idx)


def balanced_eval_subset(val_ds, per_language: int, seed: int):
    """A per-language-balanced subset of the validation set for fast eval."""
    import datasets

    langs = val_ds["language"]
    from collections import defaultdict
    import random
    idx_by = defaultdict(list)
    for i, l in enumerate(langs):
        idx_by[l].append(i)
    rng = random.Random(seed)
    picked: list[int] = []
    for l, idx in sorted(idx_by.items()):
        rng.shuffle(idx)
        picked.extend(idx[:per_language])
    picked.sort()  # keep deterministic order for compute_metrics alignment
    return val_ds.select(picked)


# ---------------------------------------------------------------------------
# Collator: build Whisper input_features + per-language labels on the fly
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class WhisperCollator:
    processor: Any
    language_map: dict
    sample_rate: int
    max_target_length: int
    decoder_start_token_id: int
    compute_dtype: Any = None  # cast mel features to the model dtype (e.g. bf16)

    def __call__(self, examples: list[dict]) -> dict:
        import numpy as np
        import torch

        audio_arrays = [np.asarray(ex["audio"]["array"], dtype=np.float32) for ex in examples]
        feats = self.processor.feature_extractor(
            audio_arrays, sampling_rate=self.sample_rate, return_tensors="pt")
        input_features = feats["input_features"]
        if self.compute_dtype is not None:
            # Match the model dtype so eval-time model.generate() (which runs
            # OUTSIDE the Trainer's autocast) doesn't hit a float/bf16 mismatch.
            input_features = input_features.to(self.compute_dtype)

        label_ids: list[list[int]] = []
        for ex in examples:
            whisper_lang = self.language_map.get(ex["language"], "swahili")
            # Per-example language token: sets <|sot|><|lang|><|transcribe|><|notimestamps|> prefix.
            self.processor.tokenizer.set_prefix_tokens(language=whisper_lang, task="transcribe")
            text = normalize_text(ex["transcription"])
            ids = self.processor.tokenizer(text, truncation=True,
                                           max_length=self.max_target_length).input_ids
            label_ids.append(ids)

        labels_batch = self.processor.tokenizer.pad({"input_ids": label_ids}, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        # Drop the leading <|startoftranscript|>: the model prepends decoder_start itself.
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels}


# ---------------------------------------------------------------------------
# Metrics closure (per language, WER-based selection) + disk logging
# ---------------------------------------------------------------------------

def make_compute_metrics(processor, eval_languages: list[str], metrics_jsonl: str):
    def compute_metrics(pred) -> dict:
        import numpy as np

        pred_ids = pred.predictions
        label_ids = np.asarray(pred.label_ids)
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        n = min(len(pred_str), len(eval_languages))
        m = per_language_metrics(label_str[:n], pred_str[:n], eval_languages[:n])
        append_jsonl(metrics_jsonl, {"event": "eval", "time": time.time(), **m})
        logger.info("EVAL  %s", format_metrics(m))
        return m

    return compute_metrics


class JsonlLoggingCallback:
    """Persists every Trainer log dict to disk immediately (loss/lr/eval), so a
    crash mid-run keeps the full history. Implemented as a TrainerCallback at
    runtime (imported lazily to keep --help fast)."""

    def __new__(cls, metrics_jsonl: str):
        import transformers

        class _Cb(transformers.TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if logs:
                    append_jsonl(metrics_jsonl, {"event": "log", "step": state.global_step,
                                                 "time": time.time(), **logs})
        return _Cb()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    import torch
    import peft
    import transformers
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    os.makedirs(args.artifact_dir, exist_ok=True)
    log_dir = os.path.join(args.artifact_dir, "logs")
    setup_file_logging(log_dir, name="train")
    metrics_jsonl = os.path.join(log_dir, "metrics.jsonl")
    write_json(os.path.join(log_dir, "run_config.json"), vars(args))
    logger.info("Run config: %s", vars(args))

    login_to_hf(args.hf_token)
    transformers.set_seed(args.seed)

    # --- data ---
    logger.info("Loading prepared dataset from %s ...", args.data_dir)
    train_ds = load_split(args.data_dir, "train")
    full_val_ds = load_split(args.data_dir, "validation")
    train_ds = filter_by_duration(train_ds, args.sample_rate, args.max_duration_sec,
                                  log_path=os.path.join(log_dir, "duration_filter.json"))

    # Language-balanced (temperature) sampling for the Luganda imbalance.
    indices, targets = build_language_balanced_indices(
        train_ds["language"], args.language_sampling_alpha, args.seed)
    train_ds = train_ds.select(indices)
    logger.info("Train after balancing (alpha=%.2f): %d examples, per-language target=%s",
                args.language_sampling_alpha, len(train_ds), targets)

    # During training we eval on a balanced subset (generation is slow); the
    # FINAL eval (after best-checkpoint reload) runs on the FULL validation set.
    eval_subset = balanced_eval_subset(full_val_ds, args.max_eval_samples_per_language, args.seed)
    eval_languages = list(eval_subset["language"])
    logger.info("Eval subset for checkpoint selection: %d (%s). Final eval on full val: %d.",
                len(eval_subset), {l: eval_languages.count(l) for l in set(eval_languages)},
                len(full_val_ds))

    # --- model + processor ---
    logger.info("Loading %s ...", args.model_id)
    processor = WhisperProcessor.from_pretrained(args.model_id)
    language_map = dict(LANGUAGE_TO_WHISPER, lug=args.luganda_language)

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16)
    model.config.apply_spec_augment = bool(args.apply_spec_augment)
    if args.apply_spec_augment:
        model.config.mask_time_prob = 0.05
        model.config.mask_feature_prob = 0.05
    # Multilingual, no forced language: the model auto-detects at generate time,
    # matching Phase-2 inference (audio only, no language id).
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # required with gradient checkpointing (training forward)
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = None
    model.generation_config.task = "transcribe"
    model.generation_config.max_length = args.generation_max_length
    model.generation_config.no_repeat_ngram_size = 3
    # Use the kv-cache during EVAL generation (model.eval() -> no checkpointing
    # conflict). Without this, generation is ~5x slower (recomputes full context
    # every token). This is separate from model.config.use_cache above.
    model.generation_config.use_cache = True

    lora = peft.LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules, bias="none")
    model = peft.get_peft_model(model, lora)
    model.enable_input_require_grads()  # needed for gradient checkpointing + PEFT
    model.print_trainable_parameters()

    collator = WhisperCollator(
        processor=processor, language_map=language_map, sample_rate=args.sample_rate,
        max_target_length=args.max_target_length,
        decoder_start_token_id=model.config.decoder_start_token_id,
        compute_dtype=torch.bfloat16)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.artifact_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        # --- eval + checkpoint selection by WER/CER, not loss/token-accuracy ---
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="combined",   # 0.5*WER_macro + 0.5*CER_macro (lower better)
        greater_is_better=False,
        logging_steps=args.logging_steps,
        logging_dir=log_dir,
        report_to="none",
        remove_unused_columns=False,        # keep audio/transcription for the collator
        label_names=["labels"],
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        run_name="whisper-large-v3-lora-v1",
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args, data_collator=collator,
        train_dataset=train_ds, eval_dataset=eval_subset,
        compute_metrics=make_compute_metrics(processor, eval_languages, metrics_jsonl),
        callbacks=[JsonlLoggingCallback(metrics_jsonl)],
    )

    logger.info("Starting fine-tuning ...")
    start = time.monotonic()
    trainer.train()
    logger.info("Fine-tuning complete in %s.", fmt_duration(time.monotonic() - start))

    # --- save artifacts + logs ---
    # load_best_model_at_end=True means trainer.model now holds the best
    # checkpoint (by combined WER/CER), so this saved adapter IS the best one.
    trainer.save_model(args.artifact_dir)          # LoRA adapter + config
    processor.save_pretrained(args.artifact_dir)
    write_json(os.path.join(log_dir, "trainer_log_history.json"), trainer.state.log_history)

    # --- final eval on the FULL validation set (auto-detect, Phase-2 style) ---
    final = run_full_eval(trainer.model, processor, full_val_ds, args, metrics_jsonl)
    write_json(os.path.join(log_dir, "final_eval_full_val.json"), final)
    logger.info("Saved artifacts + logs to %s", os.path.abspath(args.artifact_dir))

    # Package best-N checkpoints for download BEFORE the instance is destroyed.
    if args.zip_top_n:
        try:
            out, sel = zip_best_checkpoints(args.artifact_dir, args.zip_top_n)
            logger.info("Zipped best %d checkpoints -> %s  (download this!)", len(sel), out)
            for step, comb in sel:
                logger.info("  checkpoint-%d combined=%s", step,
                            "n/a" if comb == float("inf") else f"{comb:.4f}")
        except Exception as e:  # never let packaging fail the run
            logger.warning("Checkpoint zipping failed (%s); run zip_artifacts.py manually.", e)

    logger.info("Next: python infer.py --artifact-dir %s", args.artifact_dir)


def run_full_eval(model, processor, full_val_ds, args, metrics_jsonl) -> dict:
    """Generation-based per-language eval over the ENTIRE validation set, using
    auto-detect (same constraints as the real evaluation)."""
    import numpy as np

    model.eval()
    device = next(model.parameters()).device
    refs, hyps, langs = [], [], []
    buf_audio, buf_ref, buf_lang = [], [], []
    bs = max(1, args.per_device_eval_batch_size)
    start = time.monotonic()

    def flush():
        if not buf_audio:
            return
        from common import generate_transcriptions
        hyps.extend(generate_transcriptions(model, processor, list(buf_audio), device,
                                            args.sample_rate, args.generation_max_length))
        refs.extend(buf_ref)
        langs.extend(buf_lang)
        buf_audio.clear(); buf_ref.clear(); buf_lang.clear()

    for ex in full_val_ds:
        buf_audio.append(np.asarray(ex["audio"]["array"], dtype=np.float32))
        buf_ref.append(normalize_text(ex["transcription"]))
        buf_lang.append(ex["language"])
        if len(buf_audio) >= bs:
            flush()
    flush()

    m = per_language_metrics(refs, hyps, langs)
    append_jsonl(metrics_jsonl, {"event": "final_eval_full_val", "n": len(refs),
                                 "time": time.time(), **m})
    logger.info("FINAL (full val, n=%d, %s) %s", len(refs), fmt_duration(time.monotonic() - start),
                format_metrics(m))
    return m


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    train(parse_args())


if __name__ == "__main__":
    main()
