"""infer_qwen.py — board-faithful val eval (+ test submission) for a Qwen3-ASR LoRA checkpoint.

Loads base Qwen3-ASR + the LoRA adapter and runs generation UNDER THE SAME CONDITIONS as training
so the number is directly comparable to the gate's gen-eval and to MMS v2 (combined 0.2033). The
match is guaranteed by REUSING train_qwen's helpers verbatim — same prompt (build_prompt), same data
plumbing (load_split, make_preprocess_fn), same generation (left-pad, bf16 cast, greedy, .sequences
slice, max_new_tokens). Board-faithful scoring via common.per_language_metrics (cased CER / lc WER).

    # full 3,753 val — the true head-to-head vs MMS 0.2033
    python infer_qwen.py --adapter artifacts/qwen-gate/checkpoint-2065 --eval-set val
    # reproduce the gate's balanced 600-subset (completeness check: should ~match 0.2242 IFF the
    # trained encoder was captured; base-encoder+LoRA will be worse if the encoder mattered)
    python infer_qwen.py --adapter <ckpt-or-repo> --eval-set val --val-subset-per-lang 200
    # test → submission.csv (needs prepared test chunks under --audio-dir/test)
    python infer_qwen.py --adapter <ckpt> --eval-set test

Runs on the box (needs torch/transformers/peft/qwen_asr + prepared audio). --adapter accepts a local
checkpoint dir OR an HF repo id. MMS-v2 reference: combined 0.2033 (wer 0.323, cer 0.084).
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import common
import train_qwen as T   # reuse EXACT training conditions (prompt/data/generation helpers)

MMS_V2 = {"combined": 0.2033, "wer": 0.323, "cer": 0.084}


# ---------------------------------------------------------------------------
# Anti-hallucination post-processing (ported verbatim from 1st-place csrc_qwen)
# ---------------------------------------------------------------------------

def detect_and_fix_repetitions(text, threshold=20):
    """Collapse runaway repetition loops a generative decoder emits on corrupt/silent audio (a char or
    a short pattern repeated >threshold times) — these spike WER/CER. 1st-place `detect_and_fix_repetitions`."""
    def fix_char_repeats(s, thresh):
        res, i, n = [], 0, len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i]); i += count
            else:
                res.append(s[i:i + count]); i += count
        return "".join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
        i, result = 0, []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i:i + k]
                valid = True
                for rep in range(1, thresh):
                    if s[i + rep * k:i + rep * k + k] != pattern:
                        valid = False
                        break
                if valid:
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index + k] == pattern:
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            if found:
                break
            result.append(s[i]); i += 1
        if not found:
            result.append(s[i:])
        return "".join(result)

    text = fix_char_repeats(text, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(base_model: str, adapter: str, dtype):
    import torch
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel
    from transformers import GenerationConfig

    wrapper = Qwen3ASRModel.from_pretrained(base_model, dtype=dtype, device_map=None)
    model, processor = wrapper.model, wrapper.processor
    T.patch_outer_forward(model)                      # same outer→thinker patch as training
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model = PeftModel.from_pretrained(model, adapter)  # apply the trained LoRA adapter
    T.load_trained_base(model, adapter)                # + overlay the trained audio encoder (shared w/ trainer)
    model.config.use_cache = True                      # inference wants the cache (greedy → identical)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[model] base={base_model} + adapter={adapter} on {device} ({next(model.parameters()).dtype})")
    return model, processor, device


# ---------------------------------------------------------------------------
# Generation — byte-identical to train_qwen's GenerationEvalCallback
# ---------------------------------------------------------------------------

def generate_texts(model, processor, ds, args, device) -> list[str]:
    import numpy as np
    import torch

    processor.tokenizer.padding_side = "left"          # batched generation needs left-pad
    mdtype = next(model.parameters()).dtype
    hyps: list[str] = []
    with torch.no_grad():
        for s in range(0, len(ds), args.gen_batch_size):
            chunk = ds.select(range(s, min(s + args.gen_batch_size, len(ds))))
            audios = [np.asarray(a["array"] if isinstance(a, dict) else a, dtype=np.float32)
                      for a in chunk["audio"]]
            inp = processor(text=chunk["prefix_text"], audio=audios, return_tensors="pt",
                            padding=True, truncation=False).to(device)
            for k, v in list(inp.items()):             # cast float inputs to bf16 (autocast is off here)
                if torch.is_tensor(v) and v.is_floating_point():
                    inp[k] = v.to(mdtype)
            gen = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, num_beams=args.num_beams)   # beam>1 = beam search
            seq = gen.sequences if hasattr(gen, "sequences") else gen
            new = seq[:, inp["input_ids"].shape[1]:]
            texts = processor.tokenizer.batch_decode(new, skip_special_tokens=True)
            if args.fix_repetitions:
                texts = [detect_and_fix_repetitions(t, args.rep_threshold) for t in texts]
            hyps.extend(texts)
            if (s // args.gen_batch_size) % 20 == 0:
                print(f"  … {min(s + args.gen_batch_size, len(ds))}/{len(ds)}", flush=True)
    return hyps


def build_dataset(args, processor, manifest_name: str):
    """Val/eval dataset via the SAME load_split + preprocess as training (matched prompt/duration)."""
    ds = T.load_split(args, manifest_name, manifest_name)
    ds = ds.map(T.make_preprocess_fn(processor, args.prompt_mode), num_proc=1)
    keep = {"audio", "prefix_text", "target", "prompt", "language", "id"}
    drop = [c for c in ds.column_names if c not in keep]
    return ds.remove_columns(drop) if drop else ds


def balanced_subset(ds, per_lang: int, seed: int):
    """The gate's exact 200/lang selection (seed 42) so infer reproduces the gate subset."""
    idx_by: dict[str, list[int]] = {}
    for i, l in enumerate(ds["language"]):
        idx_by.setdefault(l, []).append(i)
    rng = random.Random(seed)
    picked: list[int] = []
    for l, idx in sorted(idx_by.items()):
        rng.shuffle(idx)
        picked.extend(idx[:per_lang])
    picked.sort()
    return ds.select(picked)


# ---------------------------------------------------------------------------
# Eval / submission
# ---------------------------------------------------------------------------

def eval_val(args, model, processor, device):
    ds = build_dataset(args, processor, args.val_manifest)
    if args.val_subset_per_lang > 0:
        ds = balanced_subset(ds, args.val_subset_per_lang, args.seed)
    tag = f"balanced {args.val_subset_per_lang}/lang" if args.val_subset_per_lang > 0 else "FULL"
    print(f"[val] {len(ds)} clips ({tag})")
    hyps = generate_texts(model, processor, ds, args, device)
    m = common.per_language_metrics(ds["target"], hyps, ds["language"])

    print("\n=== val (board-faithful) ===")
    print(f"combined={m['combined']:.4f}  wer={m['wer']:.4f}  cer={m['cer']:.4f}")
    for l in ("lin", "sna", "lug"):
        if f"wer_{l}" in m:
            cl = 0.5 * m[f"wer_{l}"] + 0.5 * m[f"cer_{l}"]
            print(f"  {l}: combined={cl:.4f}  wer={m[f'wer_{l}']:.4f}  cer={m[f'cer_{l}']:.4f}  n={m[f'n_{l}']}")
    d = m["combined"] - MMS_V2["combined"]
    print(f"\nvs MMS-v2 {MMS_V2['combined']:.4f}:  Δ={d:+.4f}  "
          f"({'Qwen WINS' if d < 0 else 'MMS wins'} by {abs(d):.4f})")
    out = os.path.join(args.out_dir, "val_metrics.json")
    common.write_json(out, {"adapter": args.adapter, "n": len(ds), "subset": tag, **m, "mms_v2": MMS_V2})
    print(f"[out] {out}")
    return m


def gen_submission(args, model, processor, device):
    import pandas as pd
    sample = pd.read_csv(args.sample_submission)
    id_col, txt_col = sample.columns[0], sample.columns[1]
    wanted = list(sample[id_col].astype(str))

    ds = T.load_split_ids(args, wanted) if hasattr(T, "load_split_ids") else None
    if ds is None:
        raise SystemExit(
            "Test path needs prepared TEST chunks. Prepare them (prepare_data.py streams train/validation; "
            "the board test set is separate) under --audio-dir/test, then wire a test loader. VAL eval is "
            "the decision-maker for now; run --eval-set val.")
    ds = ds.map(T.make_preprocess_fn(processor, args.prompt_mode), num_proc=1)
    hyps = generate_texts(model, processor, ds, args, device)
    pred = dict(zip(ds["id"], hyps))
    texts = [pred.get(i, "") for i in wanted]
    empties = sum(1 for t in texts if not t.strip())
    sample[txt_col] = [t if t.strip() else args.empty_placeholder for t in texts]   # never the reference
    sample.to_csv(args.submission_output, index=False)
    print(f"[submission] {args.submission_output}  ({empties} empties → placeholder {args.empty_placeholder!r})")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--adapter", required=True, help="LoRA checkpoint dir OR HF repo id.")
    p.add_argument("--eval-set", choices=["val", "test"], default="val")
    p.add_argument("--audio-dir", default=str(common.prepared_dir()))
    p.add_argument("--manifest-dir", default=str(T.find_manifest_dir()))
    p.add_argument("--val-manifest", default="val")
    p.add_argument("--out-dir", default=None)
    # matched-to-training knobs (defaults = train_qwen defaults)
    p.add_argument("--prompt-mode", choices=["neutral", "named", "closest"], default="named")
    p.add_argument("--max-duration-sec", type=float, default=45.0)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--num-beams", type=int, default=1,
                   help="1 = greedy (matches training gen-eval). >1 = beam search (2nd-place used 2–5) — "
                        "a cheap inference-time WER win for the final submission/eval.")
    p.add_argument("--fix-repetitions", action=argparse.BooleanOptionalAction, default=True,
                   help="Collapse runaway repetition loops on corrupt audio (1st-place post-proc). --no- to disable.")
    p.add_argument("--rep-threshold", type=int, default=20)
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--val-subset-per-lang", type=int, default=0, help="0 = full val; >0 = balanced subset.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--external", action="store_true")   # load_split needs the attr; default off (WAXAL-only)
    # test/submission
    p.add_argument("--sample-submission", default=str(common.data_dir() / "SampleSubmission.csv"))
    p.add_argument("--submission-output", default=None)
    p.add_argument("--empty-placeholder", default="a")
    args = p.parse_args()
    args.out_dir = args.out_dir or (args.adapter if os.path.isdir(args.adapter) else ".")
    args.submission_output = args.submission_output or os.path.join(args.out_dir, "submission.csv")
    return args


def main() -> int:
    import torch
    args = parse_args()
    common.quiet_hf_logging()
    os.makedirs(args.out_dir, exist_ok=True)
    dtype = torch.bfloat16 if (torch.cuda.is_available()
                              and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    model, processor, device = load_model(args.base_model, args.adapter, dtype)
    if args.eval_set == "val":
        eval_val(args, model, processor, device)
    else:
        gen_submission(args, model, processor, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
