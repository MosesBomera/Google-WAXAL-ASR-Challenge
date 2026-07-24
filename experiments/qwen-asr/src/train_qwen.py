"""train_qwen.py — LoRA SFT of Qwen3-ASR-1.7B on WAXAL (plan Lever 0 gate + full recipe).

Scaffold adapted from the 3rd-place Children's-ASR solution
(`data/on-top-of-pasketti-main/word-track/3rd Place/qwen3_asr_sft.py`): `patch_outer_forward`,
the audio data-collator (prefix-masked labels), and `MakeEveryCheckpointInferableCallback`.
LoRA config values come from the 1st-place `csrc_qwen/configs/qwen_exp022.yaml` (the 3rd-place
file is full-FT; PEFT here is our add-on). See ../plan-qwen-asr.md §2.

Two deliberate departures from the reference (both proven wins from the mms-ctc line):
  1. DATA: we consume the SAME prepared arrow datasets + FROZEN speaker-disjoint manifests as
     mms-ctc-v2 (audio arrays in-memory; NO wav paths, NO re-split) → val byte-identical, so
     the Qwen number is directly comparable to MMS v2's combined 0.2033.
  2. SELECTION: a board-faithful GENERATION eval (per_language_metrics.combined), not loss —
     this is what makes Lever 0 a real GO/NO-GO signal. The winners selected on loss; we do better.

RUNS ON THE GPU BOX (vast.ai flat layout), NOT locally. Requires: torch, transformers, peft,
datasets, librosa, jiwer, and the `qwen_asr` package (Qwen3ASRModel wrapper, from the reference
or `pip install qwen-asr`). Untested locally (no GPU / no prepared data / no qwen_asr here) —
smoke it on the box with `--smoke` before the real gate run.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import common  # ported verbatim from mms-ctc-v2/src

SCRIPT_DIR = Path(__file__).resolve().parent

# Qwen3-ASR supports none of these; we condition with the plain English language NAME and let
# the fine-tune bind it (Lever 1). WAXAL codes → names.
LANGUAGE_NAMES = {"lin": "Lingala", "sna": "Shona", "lug": "Luganda"}
# Closest OFFICIALLY-supported tag per WAXAL language, for prompt-mode "closest" (Lever 1c).
# (Qwen covers none natively; these are the nearest Bantu-adjacent / regional anchors to probe.)
CLOSEST_SUPPORTED = {"lin": "French", "sna": "English", "lug": "English"}


def workdir() -> Path:
    return SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR


def find_manifest_dir() -> Path:
    """Frozen manifests, resolving BOTH layouts: flat box (workdir()/manifests, e.g.
    /workspace/manifests) and repo (../mms-ctc/manifests). First existing wins; else a
    box-friendly fallback so the error message points somewhere sane."""
    exp = workdir()
    for cand in (exp / "manifests", exp.parent / "mms-ctc" / "manifests", SCRIPT_DIR / "manifests"):
        if (cand / "train.csv").exists() or (cand / "val.csv").exists():
            return cand
    return exp / "manifests"


# ---------------------------------------------------------------------------
# Prompt conditioning (Lever 1)
# ---------------------------------------------------------------------------

def build_prompt(language: str, mode: str) -> str:
    """Per-clip system prompt. `mode`: neutral | named (default) | closest."""
    if mode == "neutral":
        return "Transcribe the audio."
    if mode == "closest":
        return f"Transcribe the {CLOSEST_SUPPORTED.get(language, 'audio')} speech."
    name = LANGUAGE_NAMES.get(language)
    return f"Transcribe the {name} speech." if name else "Transcribe the audio."


# ---------------------------------------------------------------------------
# Model plumbing (from 3rd-place scaffold — kept faithful)
# ---------------------------------------------------------------------------

def patch_outer_forward(model) -> None:
    """Route the wrapper's forward straight to `.thinker.forward` so Trainer/generate hit the
    ASR head with (input_ids, input_features, feature_attention_mask, labels)."""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Model has no `.thinker.forward`; qwen3_asr wrapper incompatible.")

    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features, feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs)

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")
_HF_FILES = ("config.json", "generation_config.json", "preprocessor_config.json",
             "processor_config.json", "tokenizer_config.json", "tokenizer.json",
             "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json")


def copy_required_hf_files(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for fn in _HF_FILES:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


# ---------------------------------------------------------------------------
# LoRA (our add-on; values from 1st-place qwen_exp022.yaml)
# ---------------------------------------------------------------------------

def _set_base_trainable(model, args) -> int:
    """Unfreeze the audio encoder (and/or projector) by name, per the flags. Returns encoder param count.
    Shared by fresh LoRA and warm-start (PeftModel.from_pretrained loads those base modules FROZEN)."""
    n_enc = 0
    for name, p in model.named_parameters():
        is_encoder = ("audio_tower" in name or "audio_encoder" in name or ".encoder." in name)
        is_projector = ("projector" in name or "multi_modal_projector" in name or "audio_proj" in name)
        if is_projector:
            p.requires_grad = args.train_projector
        elif is_encoder and args.train_audio_encoder:
            p.requires_grad = True
            n_enc += p.numel()
    return n_enc


def _log_trainable(model, args, n_enc: int, tag: str) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] r={args.lora_r} | trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.2f}%) | audio-encoder unfrozen: {n_enc/1e6:.1f}M "
          f"(train_audio_encoder={args.train_audio_encoder}, train_projector={args.train_projector})")


def apply_lora(model, args) -> Any:
    """Fresh LoRA wrap (values from 1st-place qwen_exp022) + unfreeze the audio encoder."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    _log_trainable(model, args, _set_base_trainable(model, args), "lora")
    return model


def warm_start(base, args) -> Any:
    """WARM START from a trained checkpoint (adapter + trained_base) with a FRESH LR schedule — the way to
    ADD epochs after a completed run (train `--epochs` MORE at a lower `--lr`), à la train_ctc's `--model-id`
    warm start. Loads the LoRA adapter TRAINABLE + restores the trained encoder, then re-unfreezes it."""
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.init_from, is_trainable=True)
    load_trained_base(model, args.init_from)
    _log_trainable(model, args, _set_base_trainable(model, args), "warm-start")
    print(f"[warm-start] init from {args.init_from} → fresh LR schedule, +{args.epochs} epoch(s) at lr {args.lr}")
    return model


def load_trained_base(model, source: str) -> None:
    """Load `trained_base.safetensors` (the trained audio encoder) from a checkpoint dir OR HF repo, OVER
    the base weights. Shared by warm-start, resume, and infer_qwen. Aborts on key mismatch so we never
    silently continue/eval with an untrained base encoder (the exact failure that scored 0.55 vs 0.22)."""
    from safetensors.torch import load_file
    path = os.path.join(source, "trained_base.safetensors")
    if not os.path.isfile(path):
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(source, "trained_base.safetensors")
        except Exception:
            print(f"[trained_base] none at {source} → BASE audio encoder "
                  "(ok for a pure-LoRA ckpt; NOT for an encoder-trained one).")
            return
    sd = load_file(path)
    res = model.load_state_dict(sd, strict=False)
    unexpected = list(getattr(res, "unexpected_keys", []))
    if unexpected:
        raise SystemExit(f"[trained_base] {len(unexpected)}/{len(sd)} keys didn't map (e.g. "
                         f"{unexpected[:2]}) — encoder NOT restored. Aborting.")
    print(f"[trained_base] restored {len(sd)} encoder tensors from {source}")


def latest_checkpoint(output_dir: str):
    """Path of the highest-step checkpoint-N dir under output_dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    cks = [d for d in os.listdir(output_dir) if _CKPT_RE.match(d)]
    if not cks:
        return None
    return os.path.join(output_dir, max(cks, key=lambda d: int(d.split("-")[1])))


def enable_input_grads(model) -> None:
    """Make the TEXT input-embedding output require grad so gradient-checkpointed decoder blocks
    backprop into the LoRA adapters. `use_reentrant=False` already covers this, but Qwen3-ASR
    doesn't expose `get_input_embeddings()`, so we locate + hook the embedding module directly."""
    try:
        model.enable_input_require_grads()
        print("[grad] enable_input_require_grads() OK.")
        return
    except (AttributeError, NotImplementedError):
        pass
    import torch.nn as nn
    emb = None
    for getter in (lambda m: m.get_input_embeddings(),
                   lambda m: m.thinker.get_input_embeddings(),
                   lambda m: m.thinker.model.embed_tokens,
                   lambda m: m.model.embed_tokens):
        try:
            e = getter(model)
        except (AttributeError, NotImplementedError, TypeError):
            continue
        if isinstance(e, nn.Module):
            emb = e
            break
    if emb is None:  # last resort: first text-side nn.Embedding
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Embedding) and "audio" not in name.lower():
                emb = mod
                break
    if emb is None:
        print("[warn] couldn't locate text input embeddings to hook; relying on use_reentrant=False. "
              "If train loss is flat, this is the first thing to check.")
        return
    emb.register_forward_hook(lambda module, inp, out: out.requires_grad_(True))
    print("[grad] hooked text input embeddings to require grad (fallback path).")


# ---------------------------------------------------------------------------
# Data — SAME prepared arrow chunks + FROZEN manifests as mms-ctc-v2 (do NOT re-split)
# ---------------------------------------------------------------------------

def load_split(args, split: str, manifest_name: str):
    """Manifest (by id) decides membership; audio arrays come from the prepared arrow dirs.
    Mirrors mms-ctc-v2 train_ctc.load_split so val stays byte-identical. WAXAL-only unless
    --external (Lever 3) is set, in which case fleurs-*/lrsc-* chunks join the TRAIN split."""
    import datasets
    import pandas as pd

    man = pd.read_csv(os.path.join(args.manifest_dir, f"{manifest_name}.csv"), dtype={"id": str})
    wanted = set(man["id"])

    parts = []
    for corpus_split in ("train", "validation"):
        d = os.path.join(args.audio_dir, corpus_split)
        if not os.path.isdir(d):
            continue
        for lang in sorted(os.listdir(d)):
            p = os.path.join(d, lang)
            is_external = lang.startswith(("fleurs-", "lrsc-"))
            if is_external and not (args.external and split == "train"):
                continue
            if os.path.isdir(p):
                parts.append(datasets.load_from_disk(p))
    if not parts:
        raise FileNotFoundError(f"No audio chunks under {args.audio_dir} — run prepare_data.py.")
    ds = datasets.concatenate_datasets(parts)
    ds = ds.filter(lambda ex: ex["id"] in wanted, num_proc=1)
    ds = ds.filter(lambda ex: ex["duration"] <= args.max_duration_sec, num_proc=1)
    # input_length (audio samples) for length-grouped batching — group similar-length clips per batch
    # → less padding → faster/less-memory (audio dominates the padded length). Same proxy as train_ctc.
    ds = ds.map(lambda ex: {"input_length": int(ex["duration"] * args.sr)}, num_proc=1)
    print(f"[data] {split}: {len(ds)} clips ({sum(ds['duration'])/3600:.1f} h)")
    return ds


def make_preprocess_fn(processor, prompt_mode: str):
    """Attach the (add_generation_prompt) prefix text + cased+punct target to every clip.
    Keeps the audio array in place (arrow already holds it — no wav path/load)."""
    def _preprocess(ex: dict) -> dict:
        prompt = build_prompt(ex["language"], prompt_mode)
        prefix_msgs = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [{"type": "audio", "audio": None}]},
        ]
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False)[0]
        return {"prefix_text": prefix_text,
                "target": common.normalize_for_ctc(ex["transcription"]),
                "prompt": prompt}
    return _preprocess


@dataclass
class DataCollatorForQwen3ASR:
    """3rd-place collator, adapted to read audio ARRAYS from the arrow dataset (not wav paths).
    Builds full = prefix+target+eos; masks the prefix and padding out of the labels. Optionally applies
    SpecAugment to the log-mel input_features (train-only; 1st-place values)."""
    processor: Any
    sampling_rate: int = 16000
    spec_augment: bool = False
    sa_freq_masks: int = 1
    sa_freq_width: int = 15
    sa_time_masks: int = 1
    sa_time_width: float = 0.05      # fraction of the clip's (non-pad) time length
    sa_prob: float = 0.5

    def _arrays(self, features):
        import numpy as np
        out = []
        for f in features:
            a = f["audio"]
            arr = np.asarray(a["array"] if isinstance(a, dict) else a, dtype=np.float32)
            out.append(arr)
        return out

    def _apply_spec_augment(self, feats, feat_mask) -> None:
        """1st-place SpecAugment on log-mel features [B, n_mels, T]: zero freq bands + time steps,
        per-clip with prob sa_prob. `feat_mask` [B, T] gives the non-pad time length."""
        import random
        n_mels = feats.shape[1]
        for i in range(feats.shape[0]):
            if random.random() >= self.sa_prob:
                continue
            t_len = int(feat_mask[i].sum().item()) if feat_mask is not None else feats.shape[2]
            max_tw = max(1, int(t_len * self.sa_time_width))
            for _ in range(self.sa_freq_masks):
                fw = random.randint(0, self.sa_freq_width)
                f0 = random.randint(0, max(0, n_mels - fw))
                feats[i, f0:f0 + fw, :] = 0.0
            for _ in range(self.sa_time_masks):
                tw = random.randint(0, max_tw)
                t0 = random.randint(0, max(0, t_len - tw))
                feats[i, :, t0:t0 + tw] = 0.0

    def __call__(self, features):
        import torch
        # RIGHT-pad for training: labels[:prefix_len]=-100 only masks the prompt correctly when
        # content is front-aligned. Qwen tokenizers often default to left-pad → mask misaligns →
        # garbage loss. Pin it here.
        self.processor.tokenizer.padding_side = "right"
        audios = self._arrays(features)
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [p + t + eos for p, t in zip(prefix_texts, targets)]

        full = self.processor(text=full_texts, audio=audios, return_tensors="pt",
                              padding=True, truncation=False)
        if self.spec_augment and "input_features" in full:
            self._apply_spec_augment(full["input_features"], full.get("feature_attention_mask"))
        prefix = self.processor(text=prefix_texts, audio=audios, return_tensors="pt",
                                padding=True, truncation=False)

        prefix_lens = prefix["attention_mask"].sum(dim=1).tolist()
        labels = full["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, : int(pl)] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full["labels"] = labels
        return full


# ---------------------------------------------------------------------------
# Board-faithful GENERATION eval (the GO/NO-GO signal — select on combined, not loss)
# ---------------------------------------------------------------------------

class GenerationEvalCallback:
    """Runs at each save: generate on a balanced val subset, compute per_language_metrics, log
    combined (micro) + per-language to metrics.jsonl, track the best checkpoint. Implemented as a
    TrainerCallback subclass built lazily in `main` (keeps transformers import lazy for --help)."""


def build_gen_eval_callback(processor, val_ds, args, log_path: str):
    import torch
    from transformers import TrainerCallback

    # Freeze a balanced subset once (reproducible).
    import random
    idx_by: dict[str, list[int]] = {}
    for i, l in enumerate(val_ds["language"]):
        idx_by.setdefault(l, []).append(i)
    rng = random.Random(args.seed)
    picked: list[int] = []
    for l, idx in sorted(idx_by.items()):
        rng.shuffle(idx)
        picked.extend(idx[: args.val_subset_per_lang])
    picked.sort()
    subset = val_ds.select(picked)
    print(f"[gen-eval] subset {len(subset)} clips "
          f"({', '.join(f'{l}:{min(len(idx_by[l]), args.val_subset_per_lang)}' for l in sorted(idx_by))})")

    class _Cb(TrainerCallback):
        def __init__(self):
            self.best = float("inf")
            self.best_step = -1

        @torch.no_grad()
        def on_save(self, targs, state, control, model=None, **kwargs):
            if targs.process_index != 0 or model is None:
                return control
            model.eval()
            # LEFT-pad for batched generation so every sequence ends at the same position
            # (right-pad would put pad tokens between the prompt and the first generated token).
            processor.tokenizer.padding_side = "left"
            refs, hyps, langs = [], [], []
            bs = args.gen_batch_size
            import numpy as np
            for s in range(0, len(subset), bs):
                chunk = subset.select(range(s, min(s + bs, len(subset))))
                audios = [np.asarray(a["array"] if isinstance(a, dict) else a, dtype=np.float32)
                          for a in chunk["audio"]]
                prefix_texts = chunk["prefix_text"]
                inp = processor(text=prefix_texts, audio=audios, return_tensors="pt",
                                padding=True, truncation=False).to(model.device)
                # generate() runs OUTSIDE the Trainer's bf16 autocast, so float32 input_features
                # would hit bf16 conv weights → dtype mismatch. Cast float inputs to model dtype
                # (same as CastFloatInputsTrainer does for the training path).
                mdtype = next(model.parameters()).dtype
                for k, v in list(inp.items()):
                    if torch.is_tensor(v) and v.is_floating_point():
                        inp[k] = v.to(mdtype)
                gen = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, num_beams=1)
                # Qwen3-ASR generate returns thinker_result (return_dict_in_generate=True):
                # a ModelOutput whose .sequences includes the prompt → slice it off.
                seq = gen.sequences if hasattr(gen, "sequences") else gen
                new = seq[:, inp["input_ids"].shape[1]:]
                texts = processor.tokenizer.batch_decode(new, skip_special_tokens=True)
                refs.extend(chunk["target"]); hyps.extend(texts); langs.extend(chunk["language"])
            m = common.per_language_metrics(refs, hyps, langs)
            rec = {"step": state.global_step, **{k: round(float(v), 4) for k, v in m.items()}}
            common.append_jsonl(log_path, rec)
            tag = ""
            if m["combined"] < self.best:
                self.best, self.best_step = m["combined"], state.global_step
                tag = "  <- best"
            # per-language display: combined_l = 0.5*wer_l + 0.5*cer_l (kept out of the f-string
            # so this stays valid on Python 3.10/3.11 — no nested-same-quote f-strings)
            per_lang = []
            for l in ("lin", "sna", "lug"):
                if f"wer_{l}" in m:
                    cl = 0.5 * m[f"wer_{l}"] + 0.5 * m.get(f"cer_{l}", m[f"wer_{l}"])
                    per_lang.append("{}:{:.3f}".format(l, cl))
            print("[gen-eval] step {}: combined={:.4f} (wer={:.4f} cer={:.4f} | {}){}".format(
                state.global_step, m["combined"], m["wer"], m["cer"], " ".join(per_lang), tag))
            model.train()
            return control

    return _Cb()


def make_checkpoint_inferable_callback(base_model_path: str):
    from transformers import TrainerCallback

    class _Cb(TrainerCallback):
        def on_save(self, targs, state, control, **kwargs):
            if targs.process_index != 0:
                return control
            ckpt = os.path.join(targs.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(ckpt):
                copy_required_hf_files(base_model_path, ckpt)
            return control

    return _Cb()


# Audio encoder ≈ 320M params; a valid capture is far above this. Below it = the capture failed.
_MIN_TRAINED_BASE_PARAMS = 50_000_000


def make_save_trained_base_callback(args):
    """FIX for the gate's checkpoint-save bug: PEFT's adapter save omits non-LoRA trainable params, so
    with train_audio_encoder=true the trained audio encoder is NOT persisted (gate adapter was LoRA-only,
    79 MB; base-encoder reload scored 0.55 vs 0.2242). This writes those params to
    <ckpt>/trained_base.safetensors and HARD-FAILS if the expected encoder isn't captured, so a silent
    LoRA-only checkpoint can never recur. infer_qwen.py loads this back over the base encoder.

    Captured set = every requires_grad param that isn't a LoRA weight — i.e. exactly what
    train_audio_encoder / train_projector unfroze. Names come from the PeftModel and match at load time
    (infer also wraps with PeftModel), so load_state_dict(strict=False) maps them cleanly."""
    from transformers import TrainerCallback

    class _Cb(TrainerCallback):
        def on_save(self, targs, state, control, model=None, **kwargs):
            if targs.process_index != 0 or model is None:
                return control
            ckpt = os.path.join(targs.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isdir(ckpt):
                return control

            extra = {}
            for n, p in model.named_parameters():
                if p.requires_grad and "lora_" not in n.lower():
                    key = n[len("module."):] if n.startswith("module.") else n   # strip any DDP prefix
                    extra[key] = p.detach().to("cpu").contiguous()
            n_params = sum(t.numel() for t in extra.values())

            # Hard guard: if we asked to train the encoder/projector, it MUST be in the capture.
            if (args.train_audio_encoder or args.train_projector) and n_params < _MIN_TRAINED_BASE_PARAMS:
                raise RuntimeError(
                    f"[encoder-save guard] step {state.global_step}: captured only {n_params/1e6:.1f}M "
                    f"non-LoRA trainable params but train_audio_encoder/projector is set (~320M expected). "
                    f"Aborting rather than writing a silent LoRA-only checkpoint.")
            if not extra:
                return control   # pure LoRA run (encoder frozen) — nothing extra to save

            from safetensors.torch import save_file
            path = os.path.join(ckpt, "trained_base.safetensors")
            save_file(extra, path)
            print(f"[save] trained_base.safetensors: {len(extra)} tensors / {n_params/1e6:.1f}M params "
                  f"/ {os.path.getsize(path)/1e9:.2f} GB -> checkpoint-{state.global_step}")
            return control

    return _Cb()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    exp = workdir()
    p = argparse.ArgumentParser(description=__doc__)
    # Paths / data
    p.add_argument("--model-path", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--audio-dir", default=str(common.prepared_dir()))
    p.add_argument("--manifest-dir", default=str(find_manifest_dir()),
                   help="Frozen speaker-disjoint manifests (train.csv/val.csv). Resolves flat-box "
                        "(workdir/manifests) or repo (../mms-ctc/manifests); val stays byte-identical.")
    p.add_argument("--train-manifest", default="train")
    p.add_argument("--val-manifest", default="val")
    p.add_argument("--output-dir", default=str(exp / "artifacts" / "qwen-gate"))
    p.add_argument("--external", action="store_true", help="Lever 3: add fleurs-*/lrsc-* to TRAIN.")
    # Recipe (defaults = 1st-place qwen_exp022 values)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--train-audio-encoder", type=int, default=1)
    p.add_argument("--train-projector", type=int, default=0)
    p.add_argument("--prompt-mode", choices=["neutral", "named", "closest"], default="named",
                   help="Lever 1 conditioning; default = language-named prompt.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-acc", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    # SpecAugment (1st-place qwen_exp022 values) — recipe default for the full runs; --no-spec-augment to ablate.
    p.add_argument("--spec-augment", action=argparse.BooleanOptionalAction, default=True,
                   help="SpecAugment on log-mel features (train-only). Recipe default; --no-spec-augment to ablate.")
    p.add_argument("--sa-freq-masks", type=int, default=1)
    p.add_argument("--sa-freq-width", type=int, default=15)
    p.add_argument("--sa-time-masks", type=int, default=1)
    p.add_argument("--sa-time-width", type=float, default=0.05)
    p.add_argument("--sa-prob", type=float, default=0.5)
    p.add_argument("--max-duration-sec", type=float, default=45.0,
                   help="Qwen native window ~30s; windowed attention handles ~45s. 45 keeps the "
                        "val byte-complete (max val 44.1s) and drops only 8 train clips (>45s).")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--seed", type=int, default=42)
    # Eval / gen
    p.add_argument("--val-subset-per-lang", type=int, default=200)
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=384,
                   help="Longest WAXAL target is ~247 tokens (650 chars); 384 never truncates a "
                        "real reference yet fences hallucination on corrupt clips (wrapper default 4096).")
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=5)
    p.add_argument("--log-steps", type=int, default=10)
    # Modes
    p.add_argument("--gate", action="store_true",
                   help="Lever 0: 1 epoch, WAXAL-only, frequent gen-eval → GO/NO-GO.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny end-to-end check (few clips, 1 eval) to validate the box before spending.")
    # resume vs warm-start (mirrors train_ctc): resume = SAME run (crash/preemption, same LR schedule/optimizer/
    # step); warm-start = NEW run from trained weights with a FRESH schedule (the way to ADD epochs).
    p.add_argument("--resume-from-checkpoint", default="",
                   help="Crash/preemption recovery: 'auto' = latest checkpoint in --output-dir (restores "
                        "optimizer/scheduler/step AND the trained encoder). Finishes the ORIGINAL --epochs budget.")
    p.add_argument("--init-from", default="",
                   help="WARM START from a trained ckpt (adapter+trained_base) with a FRESH LR schedule — the "
                        "way to ADD epochs after a finished run: --init-from <ckpt> --epochs 2 --lr 5e-5. "
                        "Mutually exclusive with --resume-from-checkpoint.")
    args = p.parse_args()
    if args.init_from and args.resume_from_checkpoint:
        p.error("--init-from (warm start, fresh schedule) and --resume-from-checkpoint (same schedule) "
                "are mutually exclusive — pick one.")

    # --gate overrides for the cheap probe
    if args.gate:
        args.epochs = 1.0
        args.external = False
    if args.smoke:
        args.epochs = 1.0
        args.val_subset_per_lang = 4
        args.save_steps = args.eval_steps = 5
        args.max_duration_sec = min(args.max_duration_sec, 20.0)
    args.train_audio_encoder = bool(args.train_audio_encoder)
    args.train_projector = bool(args.train_projector)
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    common.quiet_hf_logging()
    import torch
    from transformers import GenerationConfig, Trainer, TrainingArguments
    from qwen_asr import Qwen3ASRModel  # reference wrapper; install on the box

    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    common.setup_file_logging(log_dir, name="train_qwen")
    common.write_json(os.path.join(log_dir, "run_config.json"), vars(args))
    metrics_jsonl = os.path.join(log_dir, "gen_eval.jsonl")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path, dtype=torch.bfloat16 if use_bf16 else torch.float16, device_map=None)
    model, processor = wrapper.model, wrapper.processor
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.config.use_cache = False
    # Fresh LoRA, OR warm-start from a trained ckpt (adapter+encoder) with a fresh schedule to add epochs.
    model = warm_start(model, args) if args.init_from else apply_lora(model, args)
    enable_input_grads(model)   # required for PEFT + grad-checkpointing to backprop

    # Data (audio arrays + frozen manifests; prefix/target attached, raw cols dropped).
    train_ds = load_split(args, "train", args.train_manifest)
    val_ds = load_split(args, "val", args.val_manifest)
    pre = make_preprocess_fn(processor, args.prompt_mode)
    keep = {"audio", "prefix_text", "target", "prompt", "language", "id", "input_length"}
    for name, ds in (("train", train_ds), ("val", val_ds)):
        ds = ds.map(pre, num_proc=1)
        drop = [c for c in ds.column_names if c not in keep]
        ds = ds.remove_columns(drop) if drop else ds
        if name == "train":
            train_ds = ds
        else:
            val_ds = ds
    if args.smoke:
        train_ds = train_ds.select(range(min(len(train_ds), 32)))

    collator = DataCollatorForQwen3ASR(
        processor=processor, sampling_rate=args.sr,
        spec_augment=args.spec_augment, sa_freq_masks=args.sa_freq_masks,
        sa_freq_width=args.sa_freq_width, sa_time_masks=args.sa_time_masks,
        sa_time_width=args.sa_time_width, sa_prob=args.sa_prob)
    if args.spec_augment:
        print(f"[aug] SpecAugment ON (train only): freq {args.sa_freq_masks}×{args.sa_freq_width}, "
              f"time {args.sa_time_masks}×{args.sa_time_width}, p={args.sa_prob}")

    # Length-grouped batching — group similar-length clips per batch (less padding → faster/less memory).
    # The arg was renamed across transformers versions (train_ctc hit this), so probe the fields.
    ta_extra: dict = {}
    ta_fields = set(TrainingArguments.__dataclass_fields__)
    if "group_by_length" in ta_fields:
        ta_extra["group_by_length"] = True
    elif "train_sampling_strategy" in ta_fields:
        ta_extra["train_sampling_strategy"] = "group_by_length"
    else:
        print("[warn] no length-grouping arg in this transformers — batches won't be grouped.")
    if "length_column_name" in ta_fields:
        ta_extra["length_column_name"] = "input_length"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr, num_train_epochs=args.epochs,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        logging_steps=args.log_steps,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, save_safetensors=True,
        eval_strategy="no",              # we eval by GENERATION in the callback, not loss
        bf16=use_bf16, fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # required with PEFT
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False, remove_unused_columns=False,
        report_to="none", seed=args.seed,
        **ta_extra,
    )

    callbacks = [
        make_checkpoint_inferable_callback(args.model_path),
        make_save_trained_base_callback(args),          # persist the trained audio encoder (gate-bug fix)
        build_gen_eval_callback(processor, val_ds, args, metrics_jsonl),
    ]
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds,
        data_collator=collator, processing_class=processor.tokenizer, callbacks=callbacks)

    resume = args.resume_from_checkpoint
    if resume == "auto":
        resume = latest_checkpoint(args.output_dir)
        print(f"[resume] auto → {resume}")
    if resume:
        # Trainer's PEFT resume restores the ADAPTER + optimizer + scheduler + step, but NOT our
        # trained_base.safetensors — so pre-load the trained encoder here, or training would continue
        # from the base encoder while the restored optimizer momentum expects the trained one.
        load_trained_base(model, resume)
    trainer.train(resume_from_checkpoint=resume or None)
    print(f"[done] checkpoints + gen_eval.jsonl under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
