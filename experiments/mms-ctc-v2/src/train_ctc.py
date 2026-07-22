"""mms-ctc-v2 / Stage 2 — fresh fine-tune of facebook/mms-1b-all (CTC) on WAXAL + FLEURS + LRSC.

Bundle B: fresh from the base checkpoint (NOT a warm-start of v1), same proven recipe, so the
val delta cleanly attributes to the new levers (external data + augmentation + longer schedule).

Carried from v1 (proven):
  * full transformer FT, freeze ONLY the CNN feature extractor;
  * fresh CASED+PUNCT CTC head over our rebuilt vocab (the board scores CER case+punct-sensitive);
  * SpecAugment + LayerDrop + dropout; 8-bit AdamW; gradient checkpointing; length-grouped batching;
  * checkpoints selected on the BOARD-FAITHFUL `combined` (micro; WER lowercased, CER cased), never loss.

New in v2:
  * WAVEFORM augmentation (augment.py: light speed + white noise), TRAIN ONLY;
  * default 10 epochs (~20 h; v1's 5 epochs took ~10 h);
  * board-faithful eval metric (common.per_language_metrics).

    python train_ctc.py                       # fresh full FT, 10 epochs, light aug (the primary)
    python train_ctc.py --no-augment          # ablation: no waveform aug
    python train_ctc.py --train-mode adapter   # cheap control
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time
from typing import Any, Optional

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
_EXP = workdir()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--audio-dir", default=str(prepared_dir()),
                   help="Chunks from prepare_data (WAXAL) + prepare_external (FLEURS) + prepare_lrsc (LRSC).")
    p.add_argument("--manifest-dir", default=str(_EXP / "manifests"))
    p.add_argument("--vocab", default=str(_EXP / "vocab" / "vocab.json"))
    p.add_argument("--artifact-dir", default=str(artifacts_dir()))
    p.add_argument("--sample-rate", type=int, default=16_000)

    # model / capacity
    p.add_argument("--model-id", default="facebook/mms-1b-all",
                   help="Hub id (fresh, the v2 default), OR a local dir to WARM START from.")
    p.add_argument("--train-mode", choices=["full", "adapter"], default="full")

    # regularization
    p.add_argument("--mask-time-prob", type=float, default=0.065)
    p.add_argument("--mask-feature-prob", type=float, default=0.10)
    p.add_argument("--layerdrop", type=float, default=0.1)
    p.add_argument("--hidden-dropout", type=float, default=0.05)
    p.add_argument("--attention-dropout", type=float, default=0.05)

    # waveform augmentation (NEW, train-only; light-first per the Sunbird lesson)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aug-speed-prob", type=float, default=0.5)
    p.add_argument("--aug-speed-min", type=float, default=0.9)
    p.add_argument("--aug-speed-max", type=float, default=1.1)
    p.add_argument("--aug-noise-prob", type=float, default=0.5)
    p.add_argument("--aug-snr-min-db", type=float, default=10.0)
    p.add_argument("--aug-snr-max-db", type=float, default=20.0)

    # optimisation
    p.add_argument("--learning-rate", type=float, default=3e-5, help="Pretrained transformer body.")
    p.add_argument("--head-learning-rate", type=float, default=1e-4, help="Fresh CTC head.")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--num-train-epochs", type=float, default=10.0,
                   help="LR anneals to 0 at the END of this budget (linear). v1's 5 epochs = ~10 h, "
                        "so 10 ≈ ~20 h. If val is still improving at the end, WARM START a second run "
                        "(--model-id <dir>) at a lower LR rather than raising this; if it plateaus by "
                        "~epoch 6, the budget was too long (best-checkpoint-on-val protects us).")
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--per-device-eval-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--max-duration-sec", type=float, default=46.5,
                   help="Keep everything — longest real WAXAL clip is 46.04 s; FLEURS is shorter.")

    # eval / checkpointing
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--save-total-limit", type=int, default=2,
                   help="Full-FT checkpoints are ~6 GB each. v1 proved checkpoint-averaging useless, "
                        "so we only need best + last — 2 is plenty (drop to 1 if disk is tight).")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", default=None, help="e.g. <user>/waxal-mms-1b-ctc-v2")
    p.add_argument("--hub-private", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-eval-samples-per-language", type=int, default=250,
                   help="Balanced subset for in-training eval; the FINAL eval uses all of val.")
    p.add_argument("--early-stopping-patience", type=int, default=4)
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="'auto' = latest checkpoint in --artifact-dir (crash/preemption recovery); "
                        "same LR schedule/optimizer/step. Different from warm start (--model-id <dir>).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-token", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data — audio chunks joined to OUR speaker-disjoint manifests
# ---------------------------------------------------------------------------

def load_split(args, split: str):
    """Our manifest decides the split; a clip in OUR val may physically sit in the corpus's
    train dir. External chunks (train/fleurs-*, train/lrsc-*) are picked up here for the TRAIN
    split by id — every prepared/{train,validation}/* subdir is scanned generically."""
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
    ds = ds.map(lambda ex: {"input_length": int(ex["duration"] * args.sample_rate)},
                remove_columns=[], num_proc=1)
    logger.info("%s: %d clips matched (%.1f h)", split, len(ds), sum(ds["duration"]) / 3600)
    return ds


def balanced_eval_subset(val_ds, per_language: int, seed: int):
    import random
    from collections import defaultdict

    idx_by = defaultdict(list)
    for i, l in enumerate(val_ds["language"]):
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
        torch_dtype=torch.float32,
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        feat_proj_dropout=0.0,
        layerdrop=args.layerdrop,
        apply_spec_augment=True,
        mask_time_prob=args.mask_time_prob,
        mask_feature_prob=args.mask_feature_prob,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,       # fresh CASED head over our rebuilt vocab
    )
    model.freeze_feature_encoder()

    if args.train_mode == "adapter":
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
    augmenter: Optional[Any] = None       # set ONLY on the train collator

    def __call__(self, features: list[dict]) -> dict:
        import numpy as np

        audio = []
        for f in features:
            wav = np.asarray(f["audio"]["array"], dtype=np.float32)
            if self.augmenter is not None:
                wav = self.augmenter(wav)
            audio.append(wav)
        batch = self.processor.feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)

        texts = [normalize_for_ctc(f["transcription"]) for f in features]
        labels = self.processor.tokenizer(texts, return_tensors="pt", padding=True)
        batch["labels"] = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        return batch


# ---------------------------------------------------------------------------
# Metrics — board-faithful (common.per_language_metrics), selected on `combined`
# ---------------------------------------------------------------------------

def make_compute_metrics(processor, eval_languages: list[str], metrics_jsonl: str):
    import numpy as np

    pad_id = processor.tokenizer.pad_token_id

    def compute_metrics(pred) -> dict:
        pred_ids = np.where(np.asarray(pred.predictions) == -100, pad_id, np.asarray(pred.predictions))
        label_ids = np.where(np.asarray(pred.label_ids) == -100, pad_id, np.asarray(pred.label_ids))
        hyps = processor.batch_decode(pred_ids)
        refs = processor.batch_decode(label_ids, group_tokens=False)

        n = min(len(hyps), len(eval_languages))
        m = per_language_metrics(refs[:n], hyps[:n], eval_languages[:n])
        append_jsonl(metrics_jsonl, {"event": "eval", "time": time.time(), **m})
        logger.info("EVAL combined %.4f (board-faithful micro) | WER %.3f CER %.3f | "
                    "lin W%.3f/C%.3f lug W%.3f/C%.3f sna W%.3f/C%.3f",
                    m.get("combined", 1), m.get("wer", 1), m.get("cer", 1),
                    m.get("wer_lin", 1), m.get("cer_lin", 1), m.get("wer_lug", 1),
                    m.get("cer_lug", 1), m.get("wer_sna", 1), m.get("cer_sna", 1))
        return m

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    """Argmax on-device before the Trainer accumulates predictions (ids, not vocab-sized logits)."""
    import torch
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def build_optimizer(model, args):
    """Two param groups (fresh head learns faster than the pretrained body); 8-bit AdamW."""
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
    import numpy as np
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

    # Waveform augmentation: TRAIN collator only. A separate plain collator is used for
    # eval/predict via AugTrainer so validation is never augmented.
    augmenter = None
    if args.augment and args.train_mode != "adapter":
        from augment import WaveformAugmenter
        augmenter = WaveformAugmenter(
            args.sample_rate, speed_prob=args.aug_speed_prob,
            speed_min=args.aug_speed_min, speed_max=args.aug_speed_max,
            noise_prob=args.aug_noise_prob, snr_min_db=args.aug_snr_min_db,
            snr_max_db=args.aug_snr_max_db)
        logger.info("Waveform augmentation ON (train only): speed p=%.2f [%.2f,%.2f], "
                    "white-noise p=%.2f SNR[%.0f,%.0f]dB", args.aug_speed_prob, args.aug_speed_min,
                    args.aug_speed_max, args.aug_noise_prob, args.aug_snr_min_db, args.aug_snr_max_db)
    elif args.augment:
        logger.info("Augmentation requested but train-mode=adapter — leaving it OFF for the control.")
    train_collator = CTCCollator(processor, args.sample_rate, augmenter=augmenter)
    eval_collator = CTCCollator(processor, args.sample_rate, augmenter=None)

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
        metric_for_best_model="combined",   # board-faithful micro (0.5*WER_lower + 0.5*CER_cased)
        greater_is_better=False,
        logging_steps=args.logging_steps, logging_dir=log_dir,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        run_name=f"mms-1b-ctc-v2-{args.train_mode}",
    )

    # Length-grouped batching — the arg was renamed across transformers versions.
    ta_fields = set(TrainingArguments.__dataclass_fields__)
    if "train_sampling_strategy" in ta_fields:
        ta_kwargs["train_sampling_strategy"] = "group_by_length"
    elif "group_by_length" in ta_fields:
        ta_kwargs["group_by_length"] = True
    else:
        logger.warning("No length-grouping arg in this transformers — batches won't be grouped.")
    if "length_column_name" in ta_fields:
        ta_kwargs["length_column_name"] = "input_length"
    for k in [k for k in ta_kwargs if k not in ta_fields]:
        logger.warning("TrainingArguments: dropping unsupported kwarg %r (version mismatch).", k)
        ta_kwargs.pop(k)

    targs = TrainingArguments(**ta_kwargs)

    class AugTrainer(Trainer):
        """Use the augmenting collator for training, a plain one for eval/predict."""
        def get_eval_dataloader(self, eval_dataset=None):
            orig, self.data_collator = self.data_collator, eval_collator
            try:
                return super().get_eval_dataloader(eval_dataset)
            finally:
                self.data_collator = orig

        def get_test_dataloader(self, test_dataset):
            orig, self.data_collator = self.data_collator, eval_collator
            try:
                return super().get_test_dataloader(test_dataset)
            finally:
                self.data_collator = orig

    trainer = AugTrainer(
        model=model, args=targs, data_collator=train_collator,
        train_dataset=train_ds, eval_dataset=eval_subset,
        compute_metrics=make_compute_metrics(processor, eval_languages, metrics_jsonl),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        optimizers=(build_optimizer(model, args), None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    resume = args.resume_from_checkpoint
    if resume == "auto":
        resume = True
        logger.info("Resuming from the latest checkpoint in %s.", args.artifact_dir)
    elif resume:
        logger.info("Resuming from %s.", resume)
    if os.path.isdir(args.model_id):
        logger.info("WARM START from local dir %s — fresh LR schedule (peak %.1e).",
                    args.model_id, args.learning_rate)

    logger.info("Starting CTC fine-tune (%.1f epochs) ...", args.num_train_epochs)
    start = time.monotonic()
    trainer.train(resume_from_checkpoint=resume)
    logger.info("Done in %s.", fmt_duration(time.monotonic() - start))

    trainer.save_model(args.artifact_dir)
    processor.save_pretrained(args.artifact_dir)
    write_json(os.path.join(log_dir, "trainer_log_history.json"), trainer.state.log_history)

    # Final eval on the FULL val set. NOTE: trainer.predict cross-batch padding over-counts WER
    # by ~0.01 vs the direct-decode path — treat this as indicative; infer.py is authoritative.
    logger.info("Final (indicative) eval on all %d val clips ...", len(full_val_ds))
    langs = list(full_val_ds["language"])
    out = trainer.predict(full_val_ds)
    pid = processor.tokenizer.pad_token_id
    hyps = processor.batch_decode(np.where(np.asarray(out.predictions) == -100, pid, out.predictions))
    refs = processor.batch_decode(np.where(np.asarray(out.label_ids) == -100, pid, out.label_ids),
                                  group_tokens=False)
    final = per_language_metrics(refs, hyps, langs)
    write_json(os.path.join(log_dir, "final_eval_full_val.json"), final)
    append_jsonl(metrics_jsonl, {"event": "final_eval_full_val", "n": len(refs), **final})
    logger.info("FINAL (full val, n=%d, indicative): combined %.4f | WER %.3f CER %.3f | "
                "lin W%.3f/C%.3f lug W%.3f/C%.3f sna W%.3f/C%.3f",
                len(refs), final["combined"], final["wer"], final["cer"],
                final["wer_lin"], final["cer_lin"], final["wer_lug"], final["cer_lug"],
                final["wer_sna"], final["cer_sna"])
    logger.info("Model + logs -> %s", os.path.abspath(args.artifact_dir))

    if args.push_to_hub:
        from push_model import push_folder
        repo = args.hub_model_id or f"waxal-mms-1b-ctc-v2-{args.train_mode}"
        url = push_folder(args.artifact_dir, repo, private=args.hub_private, token=args.hf_token)
        logger.info("Pushed model to %s", url)

    logger.info("Next: python infer.py --model %s --dtype bfloat16 --skip-submission  "
                "(authoritative val eval), then compare to v1 val 0.2087.", args.artifact_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    train(parse_args())
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
