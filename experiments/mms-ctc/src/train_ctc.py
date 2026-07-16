"""experiment mms-ctc / Stage 2 — fine-tune facebook/mms-1b-all (CTC) on WAXAL.

Reads the FROZEN Stage-1c artifacts (manifests/ + vocab/), not the corpus split:
the provided train/validation split is ~100% speaker-leaked, so we use our own
speaker-disjoint manifests. Audio comes from the raw chunks written by
prepare_data.py; the manifest decides which of OUR splits each clip belongs to.

Config (plan-mms-ctc.md §2a/§2e):
  * full transformer fine-tune, freeze ONLY the CNN feature extractor — the
    canonical wav2vec2/XLS-R recipe. Our ~40-100 h/language is medium-resource,
    where the MMS guide says fine-tuning the whole checkpoint beats adapters.
  * fresh CASED CTC head over our vocab (MMS's own pipeline lowercases and strips
    punctuation; half our metric is CER on cased, punctuated references).
  * regularized to counter the whisper-lora overfit: SpecAugment + LayerDrop + dropout.
  * 8-bit AdamW + gradient checkpointing + LENGTH-GROUPED batching (1.5-46 s span).
  * checkpoints selected on per-language WER/CER (`combined`), never loss.
  * `--train-mode adapter` runs the cheap adapter-only control.

    python train_ctc.py                       # full FT (the primary)
    python train_ctc.py --train-mode adapter  # control run
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time
from typing import Any

from common import (
    append_jsonl,
    artifacts_dir,
    fmt_duration,
    normalize_for_ctc,
    per_language_metrics,
    prepared_dir,
    quiet_hf_logging,
    setup_file_logging,
    workdir,
    write_json,
)

logger = logging.getLogger("train_ctc")
_EXP = workdir()   # flat on vast.ai (folder/*.py); src/'s parent in the repo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data (Stage-1c outputs)
    p.add_argument("--audio-dir", default=str(prepared_dir()), help="Raw chunks from prepare_data.py.")
    p.add_argument("--manifest-dir", default=str(_EXP / "manifests"))
    p.add_argument("--vocab", default=str(_EXP / "vocab" / "vocab.json"))
    p.add_argument("--artifact-dir", default=str(artifacts_dir()))
    p.add_argument("--sample-rate", type=int, default=16_000)

    # model / capacity
    p.add_argument("--model-id", default="facebook/mms-1b-all",
                   help="Hub id, OR a local dir to WARM START from (e.g. artifacts/mms-1b-all). "
                        "A local dir already has vocab_size=103 so nothing is re-initialised — use "
                        "this to EXTEND a finished run with a FRESH LR schedule (usually at a lower "
                        "--learning-rate). Different from --resume-from-checkpoint.")
    p.add_argument("--train-mode", choices=["full", "adapter"], default="full",
                   help="full = fine-tune the whole transformer (freeze CNN only). "
                        "adapter = the cheap control: train MMS adapters + head only.")
    # regularization
    p.add_argument("--mask-time-prob", type=float, default=0.065)
    p.add_argument("--mask-feature-prob", type=float, default=0.10)
    p.add_argument("--layerdrop", type=float, default=0.1)
    p.add_argument("--hidden-dropout", type=float, default=0.05)
    p.add_argument("--attention-dropout", type=float, default=0.05)

    # optimisation
    p.add_argument("--learning-rate", type=float, default=3e-5, help="Pretrained transformer.")
    p.add_argument("--head-learning-rate", type=float, default=1e-4, help="Fresh CTC head.")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--num-train-epochs", type=float, default=5.0,
                   help="NOT just a ceiling: with a linear schedule the LR decays to 0 at the END "
                        "of this budget, so early-stopping at epoch 3 of an 8-epoch run leaves the "
                        "LR at ~60%% of peak, never annealed — worse than a clean 3-epoch run. Set "
                        "it to a budget you'll actually finish (~4-5 h/epoch here: 2,065 optimizer "
                        "steps/epoch at effective batch 16). If val is still improving when the LR "
                        "hits 0, WARM START a second run (--model-id artifacts/mms-1b-all) at a "
                        "lower LR rather than raising this.")
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--per-device-eval-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--max-duration-sec", type=float, default=46.5,
                   help="Keep everything — longest real clip is 46.04 s (EDA Finding 2).")

    # eval / checkpointing
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--save-total-limit", type=int, default=3,
                   help="Full-FT checkpoints are ~6 GB each (3.9 GB fp32 weights + ~2 GB 8-bit "
                        "optimizer state), on top of ~49 GB of prepared audio. Keep >=3 so "
                        "average_checkpoints.py has candidates — they're deleted as training "
                        "rolls on, and averaging CANNOT be done retroactively. Drop to 1 only "
                        "if disk is tight.")
    p.add_argument("--push-to-hub", action="store_true",
                   help="Push the final model to the HF Hub (survives instance teardown).")
    p.add_argument("--hub-model-id", default=None, help="e.g. <user>/waxal-mms-1b-ctc")
    p.add_argument("--hub-private", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-eval-samples-per-language", type=int, default=250,
                   help="Balanced subset for in-training eval; the FINAL eval uses all of val.")
    p.add_argument("--early-stopping-patience", type=int, default=4)
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="'auto' = pick the latest checkpoint in --artifact-dir, or give a path. "
                        "Continues the SAME LR schedule, optimizer state, step count and RNG — "
                        "this is for CRASHES/PREEMPTION on a 14-40 h run, not for extending a "
                        "finished one (for that, warm start via --model-id <dir>).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-token", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data — audio chunks joined to OUR speaker-disjoint manifests
# ---------------------------------------------------------------------------

def load_split(args, split: str):
    """Our manifest decides the split; the corpus's own train/validation dirs are
    just where the audio happens to live (a clip in OUR val may sit in the corpus's
    train dir — that's the whole point of re-splitting by speaker)."""
    import datasets
    import pandas as pd

    man = pd.read_csv(os.path.join(args.manifest_dir, f"{split}.csv"), dtype={"id": str})
    wanted = set(man["id"])
    logger.info("%s manifest: %d clips, %d speakers", split, len(man), man["speaker_id"].nunique())

    parts = []
    for corpus_split in ("train", "validation"):
        d = os.path.join(args.audio_dir, corpus_split)
        if not os.path.isdir(d):
            continue
        for lang in sorted(os.listdir(d)):
            p = os.path.join(d, lang)
            if os.path.isdir(p):
                parts.append(datasets.load_from_disk(p))
    if not parts:
        raise FileNotFoundError(f"No audio chunks under {args.audio_dir} — run prepare_data.py.")
    ds = datasets.concatenate_datasets(parts)
    ds = ds.filter(lambda ex: ex["id"] in wanted, num_proc=1)
    ds = ds.filter(lambda ex: ex["duration"] <= args.max_duration_sec, num_proc=1)

    # `input_length` drives length-grouped batching (from the stored duration —
    # no audio decode needed).
    ds = ds.map(lambda ex: {"input_length": int(ex["duration"] * args.sample_rate)},
                remove_columns=[], num_proc=1)
    logger.info("%s: %d clips matched (%.1f h)", split, len(ds), sum(ds["duration"]) / 3600)
    return ds


def balanced_eval_subset(val_ds, per_language: int, seed: int):
    import random
    from collections import defaultdict

    langs = val_ds["language"]
    idx_by = defaultdict(list)
    for i, l in enumerate(langs):
        idx_by[l].append(i)
    rng = random.Random(seed)
    picked = []
    for l, idx in sorted(idx_by.items()):
        rng.shuffle(idx)
        picked.extend(idx[:per_language])
    picked.sort()
    return val_ds.select(picked)


# ---------------------------------------------------------------------------
# Processor / model
# ---------------------------------------------------------------------------

def build_processor(args):
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

    tok = Wav2Vec2CTCTokenizer(args.vocab, unk_token="[UNK]", pad_token="[PAD]",
                               word_delimiter_token="|")
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=args.sample_rate,
                                  padding_value=0.0, do_normalize=True,
                                  return_attention_mask=True)
    return Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)


def build_model(args, processor):
    import torch
    from transformers import Wav2Vec2ForCTC

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,          # master weights fp32; bf16 via Trainer autocast
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_proj_dropout=0.0,
        layerdrop=args.layerdrop,
        apply_spec_augment=True,
        mask_time_prob=args.mask_time_prob,
        mask_feature_prob=args.mask_feature_prob,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,             # bad alignments -> 0 not inf
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,       # fresh CASED head over our vocab
    )
    # Always freeze the conv front-end: it carries transferable low-level acoustics,
    # drops 300M+ params from the optimiser, and stabilises low-resource training.
    model.freeze_feature_encoder()

    if args.train_mode == "adapter":
        # Control run: MMS's own parameter-efficient path.
        model.init_adapter_layers()
        model.freeze_base_model()
        for name, p in model.named_parameters():
            if "adapter" in name or name.startswith("lm_head"):
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("train-mode=%s | trainable %.1fM / %.1fM params (%.2f%%)",
                args.train_mode, trainable / 1e6, total / 1e6, 100 * trainable / total)
    return model


@dataclasses.dataclass
class CTCCollator:
    processor: Any
    sample_rate: int

    def __call__(self, features: list[dict]) -> dict:
        import numpy as np
        import torch

        audio = [np.asarray(f["audio"]["array"], dtype=np.float32) for f in features]
        batch = self.processor.feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)

        texts = [normalize_for_ctc(f["transcription"]) for f in features]
        labels = self.processor.tokenizer(texts, return_tensors="pt", padding=True)
        # -100 => ignored by CTC loss
        batch["labels"] = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        return batch


# ---------------------------------------------------------------------------
# Metrics — per language, selected on WER/CER (never loss)
# ---------------------------------------------------------------------------

def make_compute_metrics(processor, eval_languages: list[str], metrics_jsonl: str):
    import numpy as np

    pad_id = processor.tokenizer.pad_token_id

    def compute_metrics(pred) -> dict:
        # predictions are already argmaxed ids (see preprocess_logits_for_metrics),
        # padded across batches with -100 -> map those to blank so they collapse away.
        pred_ids = np.asarray(pred.predictions)
        pred_ids = np.where(pred_ids == -100, pad_id, pred_ids)
        label_ids = np.asarray(pred.label_ids)
        label_ids = np.where(label_ids == -100, pad_id, label_ids)

        hyps = processor.batch_decode(pred_ids)
        refs = processor.batch_decode(label_ids, group_tokens=False)

        n = min(len(hyps), len(eval_languages))
        m = per_language_metrics(refs[:n], hyps[:n], eval_languages[:n])
        append_jsonl(metrics_jsonl, {"event": "eval", "time": time.time(), **m})
        logger.info("EVAL combined(macro) %.4f | WER %.3f CER %.3f | lin W%.3f/C%.3f "
                    "lug W%.3f/C%.3f sna W%.3f/C%.3f",
                    m.get("combined", 1), m.get("wer_macro", 1), m.get("cer_macro", 1),
                    m.get("wer_lin", 1), m.get("cer_lin", 1), m.get("wer_lug", 1),
                    m.get("cer_lug", 1), m.get("wer_sna", 1), m.get("cer_sna", 1))
        return m

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    """Argmax on-device BEFORE the Trainer accumulates predictions: keeps eval
    memory ~100x smaller (ids not vocab-sized logits) — otherwise 46 s clips x
    103 vocab would be gigabytes."""
    import torch

    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def build_optimizer(model, args):
    """Two param groups: the fresh CTC head learns faster than the pretrained
    transformer. 8-bit AdamW keeps a 1B full fine-tune inside 40 GB."""
    import torch

    head, body = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith("lm_head") else body).append(p)
    groups = [{"params": body, "lr": args.learning_rate},
              {"params": head, "lr": args.head_learning_rate}]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, weight_decay=args.weight_decay)
        logger.info("optimizer: bitsandbytes AdamW8bit | body lr %.1e, head lr %.1e",
                    args.learning_rate, args.head_learning_rate)
    except ImportError:
        opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        logger.warning("bitsandbytes unavailable — falling back to fp32 AdamW (more memory).")
    return opt


# ---------------------------------------------------------------------------

def train(args) -> None:
    import torch
    import transformers
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    os.makedirs(args.artifact_dir, exist_ok=True)
    log_dir = os.path.join(args.artifact_dir, "logs")
    setup_file_logging(log_dir, name="train_ctc")
    metrics_jsonl = os.path.join(log_dir, "metrics.jsonl")
    write_json(os.path.join(log_dir, "run_config.json"), vars(args))
    logger.info("Run config: %s", vars(args))
    transformers.set_seed(args.seed)

    train_ds = load_split(args, "train")
    full_val_ds = load_split(args, "val")
    eval_subset = balanced_eval_subset(full_val_ds, args.max_eval_samples_per_language, args.seed)
    eval_languages = list(eval_subset["language"])
    logger.info("Eval subset %d clips %s | final eval on all %d val clips",
                len(eval_subset), {l: eval_languages.count(l) for l in set(eval_languages)},
                len(full_val_ds))

    processor = build_processor(args)
    logger.info("Vocab %d (cased+punctuated), blank/pad id=%d",
                len(processor.tokenizer), processor.tokenizer.pad_token_id)
    model = build_model(args, processor)
    collator = CTCCollator(processor=processor, sample_rate=args.sample_rate)

    ta_kwargs = dict(
        output_dir=args.artifact_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        bf16=True,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="combined",   # 0.5*WER_macro + 0.5*CER_macro
        greater_is_better=False,
        logging_steps=args.logging_steps, logging_dir=log_dir,
        report_to="none",
        remove_unused_columns=False,        # collator needs audio/transcription
        label_names=["labels"],
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        run_name=f"mms-1b-ctc-{args.train_mode}",
    )

    # Length-grouped batching (clips span 1.5-46 s). This arg was RENAMED in recent
    # transformers: the boolean `group_by_length` became `train_sampling_strategy`
    # (default "random", with "group_by_length" as an option). Pick whichever the
    # installed version actually defines so we don't crash on either.
    ta_fields = set(TrainingArguments.__dataclass_fields__)
    if "train_sampling_strategy" in ta_fields:
        ta_kwargs["train_sampling_strategy"] = "group_by_length"
    elif "group_by_length" in ta_fields:
        ta_kwargs["group_by_length"] = True
    else:
        logger.warning("No length-grouping arg in this transformers — batches won't be grouped "
                       "(training still correct, just less memory-even).")
    if "length_column_name" in ta_fields:
        ta_kwargs["length_column_name"] = "input_length"

    # Insurance against further renames on this bleeding-edge build: drop any kwarg
    # TrainingArguments no longer accepts, loudly, instead of crashing the run.
    for k in [k for k in ta_kwargs if k not in ta_fields]:
        logger.warning("TrainingArguments: dropping unsupported kwarg %r (version mismatch).", k)
        ta_kwargs.pop(k)

    targs = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model, args=targs, data_collator=collator,
        train_dataset=train_ds, eval_dataset=eval_subset,
        compute_metrics=make_compute_metrics(processor, eval_languages, metrics_jsonl),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        optimizers=(build_optimizer(model, args), None),   # Trainer builds the scheduler
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    resume = args.resume_from_checkpoint
    if resume == "auto":
        resume = True          # Trainer finds the newest checkpoint-* in output_dir
        logger.info("Resuming from the latest checkpoint in %s (same LR schedule/optimizer/step).",
                    args.artifact_dir)
    elif resume:
        logger.info("Resuming from %s (same LR schedule/optimizer/step).", resume)
    if os.path.isdir(args.model_id):
        logger.info("WARM START from local dir %s — fresh LR schedule (peak %.1e). Not a resume: "
                    "optimizer/step start from scratch.", args.model_id, args.learning_rate)

    logger.info("Starting CTC fine-tune ...")
    start = time.monotonic()
    trainer.train(resume_from_checkpoint=resume)
    logger.info("Done in %s.", fmt_duration(time.monotonic() - start))

    trainer.save_model(args.artifact_dir)
    processor.save_pretrained(args.artifact_dir)
    write_json(os.path.join(log_dir, "trainer_log_history.json"), trainer.state.log_history)

    # Final eval on the FULL speaker-disjoint val set (the honest number).
    logger.info("Final eval on all %d val clips ...", len(full_val_ds))
    langs = list(full_val_ds["language"])
    out = trainer.predict(full_val_ds)
    import numpy as np
    pid = processor.tokenizer.pad_token_id
    hyps = processor.batch_decode(np.where(np.asarray(out.predictions) == -100, pid, out.predictions))
    refs = processor.batch_decode(np.where(np.asarray(out.label_ids) == -100, pid, out.label_ids),
                                  group_tokens=False)
    final = per_language_metrics(refs, hyps, langs)
    write_json(os.path.join(log_dir, "final_eval_full_val.json"), final)
    append_jsonl(metrics_jsonl, {"event": "final_eval_full_val", "n": len(refs), **final})
    logger.info("FINAL (full speaker-disjoint val, n=%d): combined %.4f | WER %.3f CER %.3f | "
                "lin W%.3f/C%.3f lug W%.3f/C%.3f sna W%.3f/C%.3f",
                len(refs), final["combined"], final["wer_macro"], final["cer_macro"],
                final["wer_lin"], final["cer_lin"], final["wer_lug"], final["cer_lug"],
                final["wer_sna"], final["cer_sna"])
    logger.info("Model + logs -> %s", os.path.abspath(args.artifact_dir))

    if args.push_to_hub:
        from push_model import push_folder
        repo = args.hub_model_id or f"waxal-mms-1b-ctc-{args.train_mode}"
        url = push_folder(args.artifact_dir, repo, private=args.hub_private, token=args.hf_token)
        logger.info("Pushed model to %s", url)

    logger.info("Checkpoints kept for averaging: %s",
                sorted(d for d in os.listdir(args.artifact_dir) if d.startswith("checkpoint-")))
    logger.info("Next: python average_checkpoints.py   (then Stage 3 — KenLM decoding)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    train(parse_args())
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
