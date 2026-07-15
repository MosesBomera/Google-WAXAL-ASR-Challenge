"""Fine-tunes Gemma 3n for multilingual ASR (Lingala + Shona + Luganda) with
LoRA, then saves the adapter + processor to --artifact-dir.

Trains off a dataset already materialized to disk by prepare_data.py — this
script does NOT stream from the Hub. Run prepare_data.py first.

Usage:
    python prepare_data.py --output-dir ./data/waxal-multilingual
    python train.py --data-dir ./data/waxal-multilingual
    python train.py --data-dir ./data/waxal-multilingual --max-steps 3000 --lora-r 16
"""

from __future__ import annotations

import argparse
import logging
import os

from common import login_to_hf, make_audio_to_array, make_formatters, quiet_hf_logging

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model
    p.add_argument(
        "--model-id", default="google/gemma-3n-E2B-it",
        help="google/gemma-3n-E2B-it (~2B, T4) or google/gemma-4n-E4B-it (~4B, A100)",
    )

    # Dataset — loaded from disk (see prepare_data.py), not streamed.
    p.add_argument(
        "--data-dir", default="./data/waxal-multilingual",
        help="Chunks saved by prepare_data.py: <data-dir>/{train,validation}/"
             "<language>/. Loaded and concatenated here at train time.",
    )

    # Prompts (MUST match infer.py — used to rebuild 'messages' at collate time)
    p.add_argument(
        "--system-message",
        default="You are an assistant that transcribes speech accurately.",
    )
    p.add_argument("--user-message", default="Please transcribe this audio.")

    # Audio
    p.add_argument("--sample-rate", type=int, default=16_000)

    # Artifacts — the coupling point with infer.py
    p.add_argument(
        "--artifact-dir", default="./gemma3n-asr-multilingual",
        help="Where the LoRA adapter + processor are saved. Point infer.py "
             "at the same directory. Use persistent storage (Drive/bucket) "
             "if train/infer run in separate sessions.",
    )

    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument(
        "--lora-target-modules", nargs="+", default=["v_proj", "o_proj"],
        help="Add q_proj k_proj for more trainable capacity.",
    )

    # Training
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument(
        "--max-steps", type=int, default=500,
        help="Increase for better results, e.g. 3000 for production runs.",
    )
    p.add_argument("--max-seq-length", type=int, default=64)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--num-validation-examples", type=int, default=200)
    p.add_argument("--dataloader-num-workers", type=int, default=2)

    # Auth
    p.add_argument(
        "--hf-token", default=None,
        help="Hugging Face token. Falls back to the HF_TOKEN env var, then "
             "an interactive login prompt if neither is set.",
    )

    return p.parse_args()


def load_prepared_dataset(data_dir: str, seed: int, num_validation_examples: int):
    """Loads and concatenates the per-(split, language) chunks saved by
    prepare_data.py (data_dir/train/<lang>/, data_dir/validation/<lang>/).

    These are regular (map-style) Datasets, not streaming — so, unlike the
    streaming version of this script, the Trainer can shuffle/reshuffle each
    epoch itself and we don't need a manual .shuffle().repeat(None) trick.

    Loading/concatenating is lazy (memory-mapped Arrow, no full
    materialization), so doing it here rather than in prepare_data.py keeps
    prepare_data.py's peak memory bounded to one chunk at a time.
    """
    import datasets

    def load_split(split: str):
        split_dir = os.path.join(data_dir, split)
        lang_dirs = sorted(
            os.path.join(split_dir, name) for name in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, name))
        )
        parts = [datasets.load_from_disk(p) for p in lang_dirs]
        return datasets.concatenate_datasets(parts)

    train_ds = load_split("train").shuffle(seed=seed)
    # Shuffle validation too, before truncating — chunks are saved one
    # language at a time, so an un-shuffled .select() on the concatenated
    # dataset would silently grab only the first language's examples.
    val_ds = load_split("validation").shuffle(seed=seed)
    if num_validation_examples is not None and num_validation_examples < len(val_ds):
        val_ds = val_ds.select(range(num_validation_examples))
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Model / collator
# ---------------------------------------------------------------------------

def load_model_and_processor(model_id: str):
    import timm  # noqa: F401  (required by Gemma3n; import before transformers)
    import torch
    import transformers

    processor = transformers.AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"
    model = transformers.Gemma3nForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    return model, processor


def make_collator(processor, audio_to_array, format_for_chat):
    import torch

    def _mask_labels(labels: "torch.Tensor") -> "torch.Tensor":
        """Masks padding/special tokens to -100 so loss ignores them."""
        labels = labels.clone()
        tok = processor.tokenizer
        special_attrs = ["pad_token_id", "image_token_id", "audio_token_id",
                          "boi_token_id", "eoi_token_id"]
        mask_ids = [getattr(tok, a) for a in special_attrs
                    if getattr(tok, a, None) is not None]
        if mask_ids:
            labels[torch.isin(labels, torch.tensor(mask_ids, device=labels.device))] = -100
        return labels

    def collate_fn(examples):
        # 'messages' is built here (not stored on disk) so the saved dataset
        # keeps exactly one copy of each audio array.
        formatted = [format_for_chat(ex) for ex in examples]
        texts = [processor.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)
            for ex in formatted]
        audios = [audio_to_array(ex["audio"]) for ex in examples]
        batch = processor(text=texts, audio=audios, return_tensors="pt", padding=True)
        batch = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        batch["labels"] = _mask_labels(batch["input_ids"])
        return batch

    return collate_fn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    import torch
    import peft
    import trl

    login_to_hf(args.hf_token)

    logger.info("Loading prepared dataset from %s ...", args.data_dir)
    train_ds, val_ds = load_prepared_dataset(
        args.data_dir, args.seed, args.num_validation_examples
    )
    logger.info("Train examples: %d, Validation examples: %d", len(train_ds), len(val_ds))

    audio_to_array = make_audio_to_array(args.sample_rate)
    format_for_chat, _ = make_formatters(
        audio_to_array, args.system_message, args.user_message
    )

    logger.info("Loading model %s ...", args.model_id)
    model, processor = load_model_and_processor(args.model_id)
    logger.info("Model loaded — device: %s, dtype: %s", model.device, model.dtype)

    collator = make_collator(processor, audio_to_array, format_for_chat)

    class _SFTTrainer(trl.SFTTrainer):
        """No-op create_model_card to avoid importlib issues."""
        def create_model_card(self, model_name=None, dataset_name=None, tags=None):
            pass

    peft_config = peft.LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules, bias="none",
        use_rslora=False, use_dora=False,
    )

    training_args = trl.SFTConfig(
        output_dir=args.artifact_dir,
        max_steps=args.max_steps,
        eval_strategy="steps", eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps, save_steps=args.save_steps,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none", run_name="gemma3n-asr-multilingual",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        max_length=args.max_seq_length, packing=False,
        dataloader_num_workers=args.dataloader_num_workers, seed=args.seed,
    )

    trainer = _SFTTrainer(
        model=model, args=training_args, data_collator=collator,
        train_dataset=train_ds, eval_dataset=val_ds,
        peft_config=peft_config,
    )

    logger.info("Trainer initialised. Starting fine-tuning ...")
    trainer.train()
    logger.info("Fine-tuning complete.")

    os.makedirs(args.artifact_dir, exist_ok=True)
    trainer.save_model(args.artifact_dir)  # adapter_config.json + adapter weights
    processor.save_pretrained(args.artifact_dir)

    logger.info("Saved artifacts to: %s", os.path.abspath(args.artifact_dir))
    for f in sorted(os.listdir(args.artifact_dir)):
        logger.info("  %s", f)
    logger.info("Next: run infer.py with --artifact-dir %s", args.artifact_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
