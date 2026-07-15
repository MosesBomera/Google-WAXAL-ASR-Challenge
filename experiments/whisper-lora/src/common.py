"""Shared helpers for whisper-lora / v1 — Whisper-large-v3 LoRA fine-tuning.

Kept dependency-light at import time (torch/datasets/jiwer/librosa are imported
lazily inside functions) so `--help` on any script stays fast.

This module also owns the three cross-cutting concerns v1 adds on top of the
gemma3n-baseline scripts:
  * text normalization applied identically to training labels and to scoring,
  * per-language WER/CER (the competition scores Lingala/Shona/Luganda),
  * persistent logging to disk (a file log + a JSONL metrics stream) so nothing
    is lost when the process exits.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def repo_root() -> Path:
    """Repo root when running from the local checkout (experiments/whisper-lora/src/
    -> parents[3]); on a flat vast.ai layout there is no such root, so fall back
    to the script directory instead of raising."""
    p = Path(__file__).resolve()
    return p.parents[3] if len(p.parents) >= 4 else SCRIPT_DIR


def data_dir() -> Path:
    """Folder holding Train.csv / Test.csv / SampleSubmission.csv.

    Two supported layouts, checked in order:
      * vast.ai (how we run):  `data/` sits next to these scripts, CSVs directly inside.
      * local checkout:        data/google-waxal-.../ under the repo root.

    Falls back to the vast.ai path so `--help` never crashes even if data is absent.
    """
    candidates = [SCRIPT_DIR / "data"]
    if len(Path(__file__).resolve().parents) >= 4:
        candidates.append(repo_root() / "data" / "google-waxal-asr-challenge20260630-10570-elxebu")
    for c in candidates:
        if (c / "Train.csv").exists():
            return c
    return candidates[0]


def prepared_dir() -> Path:
    """Where prepare_data.py writes materialized audio chunks (kept separate
    from the CSV `data/` folder). Always relative to the scripts."""
    return SCRIPT_DIR / "prepared" / "waxal-multilingual"


def artifacts_dir() -> Path:
    return SCRIPT_DIR / "artifacts" / "whisper-large-v3-lora"


# Whisper's tokenizer has language tokens for Lingala (ln) and Shona (sn) but
# NOT Luganda (lg). We train Luganda under the closest supported Bantu token,
# Swahili (sw). Fine-tuning overwrites the token's original meaning, so the
# decoder learns to emit Luganda orthography under it. See plan-whisper-lora.md §8.
LANGUAGE_TO_WHISPER: dict[str, str] = {
    "lin": "lingala",
    "sna": "shona",
    "lug": "swahili",  # proxy — no native Luganda token
}

TARGET_LANGUAGES = ("lin", "sna", "lug")


# ---------------------------------------------------------------------------
# HF logging hygiene
# ---------------------------------------------------------------------------

def quiet_hf_logging() -> None:
    """Mutes the very verbose default output from the HF stack. Call once at the
    top of each script's main(), before anything imports the HF libs."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for name in (
        "huggingface_hub", "datasets", "transformers", "filelock",
        "httpx", "httpcore", "urllib3", "fsspec", "hf_xet",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except ImportError:
        pass
    try:
        import datasets
        datasets.logging.set_verbosity_error()
        datasets.disable_progress_bars()
    except (ImportError, AttributeError):
        pass
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Persistent logging to disk (the user's explicit ask)
# ---------------------------------------------------------------------------

def setup_file_logging(log_dir: str | os.PathLike, name: str = "run") -> str:
    """Attaches a FileHandler to the root logger so ALL logs are written to
    `<log_dir>/<name>.log` on disk as they happen — surviving process exit.

    Returns the log file path. Safe to call once per script.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    root = logging.getLogger()
    # Avoid duplicate handlers if called twice.
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == os.path.abspath(log_path)
               for h in root.handlers):
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
    return log_path


def append_jsonl(path: str | os.PathLike, record: Mapping[str, Any]) -> None:
    """Appends one JSON record + newline. Used for a live metrics stream that
    is flushed to disk immediately (so a crash mid-training keeps history)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def write_json(path: str | os.PathLike, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def login_to_hf(hf_token: str | None) -> None:
    """Authenticates with the Hub (needed to stream google/WaxalNLP)."""
    from huggingface_hub import login

    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        logger.warning("No --hf-token / HF_TOKEN set; falling back to interactive login.")
        login()


# ---------------------------------------------------------------------------
# CSV / dataset loading
# ---------------------------------------------------------------------------

def load_train_csv(path: str | os.PathLike):
    """Loads Zindi's Train.csv (id, transcription, language, original_split).
    It uses backslash-escaped quotes inside transcriptions, so escapechar='\\'
    is required or pandas mis-parses long rows."""
    import pandas as pd

    df = pd.read_csv(path, escapechar="\\")
    df = df.rename(columns={"original_split": "split"})
    assert not df["id"].duplicated().any(), "Train.csv has duplicate ids"
    return df


def detect_id_key(example: Mapping[str, Any], wanted_ids: set) -> str:
    """Finds which field of an HF example holds a known id (e.g. 'lug_96114')."""
    for key in ("id", "ID", "utt_id", "utterance_id", "file", "path", "name"):
        if key in example and str(example[key]) in wanted_ids:
            return key
    for key, value in example.items():
        if isinstance(value, str) and value in wanted_ids:
            return key
    raise KeyError(
        f"Could not find an id field matching known ids. Keys: {list(example.keys())}"
    )


def load_streaming_split(dataset_id: str, language: str, split: str, sample_rate: int):
    """Streams one language/split from the Hub, resampled to sample_rate."""
    import datasets

    ds = datasets.load_dataset(dataset_id, name=f"{language}_asr", split=split, streaming=True)
    return ds.cast_column("audio", datasets.Audio(sampling_rate=sample_rate))


def make_audio_to_array(sample_rate: int):
    """Returns a fn extracting a mono float32 array at `sample_rate` from a
    datasets Audio value (dict form or torchcodec AudioDecoder)."""
    import librosa
    import numpy as np

    def audio_to_array(audio: Any) -> "np.ndarray":
        sr = sample_rate
        if isinstance(audio, Mapping):
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate", sample_rate))
        elif hasattr(audio, "get_all_samples"):  # torchcodec AudioDecoder
            samples = audio.get_all_samples()
            arr = np.asarray(samples.data, dtype=np.float32)
            sr = int(getattr(samples, "sample_rate", sample_rate))
        else:
            arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        arr = arr.flatten()
        if sr != sample_rate:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=sample_rate)
        return arr

    return audio_to_array


# ---------------------------------------------------------------------------
# Text normalization — applied to BOTH training labels and scoring
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def normalize_text(s: Any, lower: bool = False) -> str:
    """Consistent, light normalization. Keeps casing/punctuation by default
    (they earn CER credit); set lower=True to strip case. MUST be identical for
    references and hypotheses, or WER/CER is meaningless."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace('\\"', '"').replace("\\'", "'")  # undo CSV backslash escapes
    s = unicodedata.normalize("NFC", s)
    s = _WS.sub(" ", s).strip()
    if lower:
        s = s.lower()
    return s


# ---------------------------------------------------------------------------
# Metrics — per language + macro + micro
# ---------------------------------------------------------------------------

def per_language_metrics(
    refs: list[str],
    hyps: list[str],
    langs: list[str],
    wer_weight: float = 0.5,
) -> dict[str, float]:
    """Computes WER/CER per language plus macro (mean over languages) and micro
    (pooled over all clips) aggregates.

    Returns a flat dict of float metrics, e.g.:
      wer_lin, cer_lin, n_lin, ... , wer (micro), cer (micro),
      wer_macro, cer_macro, combined (=0.5*wer_macro+0.5*cer_macro),
      combined_micro (=0.5*wer_micro+0.5*cer_micro).

    `combined` (macro) is our checkpoint-selection metric: it weights every
    language equally so Luganda can't be ignored. `combined_micro` mirrors a
    pooled leaderboard; we log both and compare to the board (plan-whisper-lora.md §6a).
    """
    import jiwer

    # jiwer errors on empty reference strings; drop empty-ref pairs defensively.
    triples = [(normalize_text(r), normalize_text(h), l)
               for r, h, l in zip(refs, hyps, langs) if normalize_text(r)]
    if not triples:
        return {"wer": 1.0, "cer": 1.0, "wer_macro": 1.0, "cer_macro": 1.0,
                "combined": 1.0, "combined_micro": 1.0}

    by = defaultdict(lambda: ([], []))
    all_refs, all_hyps = [], []
    for r, h, l in triples:
        by[l][0].append(r)
        by[l][1].append(h)
        all_refs.append(r)
        all_hyps.append(h)

    out: dict[str, float] = {}
    wers, cers = [], []
    for l in sorted(by):
        rs, hs = by[l]
        w = float(jiwer.wer(rs, hs))
        c = float(jiwer.cer(rs, hs))
        out[f"wer_{l}"] = w
        out[f"cer_{l}"] = c
        out[f"n_{l}"] = len(rs)
        wers.append(w)
        cers.append(c)

    out["wer"] = float(jiwer.wer(all_refs, all_hyps))   # micro / pooled
    out["cer"] = float(jiwer.cer(all_refs, all_hyps))
    out["wer_macro"] = sum(wers) / len(wers)
    out["cer_macro"] = sum(cers) / len(cers)
    out["combined"] = wer_weight * out["wer_macro"] + (1 - wer_weight) * out["cer_macro"]
    out["combined_micro"] = wer_weight * out["wer"] + (1 - wer_weight) * out["cer"]
    return out


def format_metrics(m: Mapping[str, float]) -> str:
    """One-line human summary of a per_language_metrics dict."""
    langs = sorted({k.split("_", 1)[1] for k in m if k.startswith("wer_") and k != "wer_macro"})
    parts = [f"{l}: W{m.get(f'wer_{l}', float('nan')):.3f}/C{m.get(f'cer_{l}', float('nan')):.3f}"
             for l in langs]
    return (f"combined(macro) {m.get('combined', float('nan')):.4f} | "
            f"WER {m.get('wer_macro', float('nan')):.3f} CER {m.get('cer_macro', float('nan')):.3f} "
            f"(macro) | micro W{m.get('wer', float('nan')):.3f}/C{m.get('cer', float('nan')):.3f} | "
            + " ".join(parts))


# ---------------------------------------------------------------------------
# Language-balanced (temperature) sampling for the Luganda imbalance
# ---------------------------------------------------------------------------

def build_language_balanced_indices(
    languages: list[str], alpha: float, seed: int
) -> tuple[list[int], dict[str, int]]:
    """Temperature sampling: draw language l with prob ∝ p_l^alpha (plan-whisper-lora.md §6a).

    alpha=1 -> natural (imbalanced), alpha=0 -> uniform per language,
    alpha≈0.5 -> the multilingual-ASR sweet spot (upsamples Luganda ~1.7x).

    Keeps the total dataset size roughly constant; returns (indices, target_counts).
    Sampling is with replacement when a language's target exceeds its supply.
    """
    import random

    counts = Counter(languages)
    total = len(languages)
    if alpha >= 0.999:  # natural proportions — no resampling needed
        idx = list(range(total))
        random.Random(seed).shuffle(idx)
        return idx, dict(counts)

    weighted = {l: (counts[l] / total) ** alpha for l in counts}
    z = sum(weighted.values())
    target = {l: max(1, round(weighted[l] / z * total)) for l in counts}

    lang_idx: dict[str, list[int]] = defaultdict(list)
    for i, l in enumerate(languages):
        lang_idx[l].append(i)

    rng = random.Random(seed)
    indices: list[int] = []
    for l, t in target.items():
        pool = lang_idx[l]
        if t <= len(pool):
            indices.extend(rng.sample(pool, t))
        else:
            indices.extend(pool)
            indices.extend(rng.choice(pool) for _ in range(t - len(pool)))
    rng.shuffle(indices)
    return indices, target


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Package the best-N checkpoints for download BEFORE destroying the instance
# ---------------------------------------------------------------------------

def rank_checkpoints_by_combined(artifact_dir: str | os.PathLike) -> list[tuple[int, str, float]]:
    """Returns [(step, checkpoint_dir, eval_combined), ...] sorted best-first,
    reading eval scores from logs/trainer_log_history.json (falling back to each
    checkpoint's own trainer_state.json). Checkpoints with no recorded score go last."""
    import glob
    import json
    import re

    artifact_dir = str(artifact_dir)
    step_combined: dict[int, float] = {}
    hist = os.path.join(artifact_dir, "logs", "trainer_log_history.json")
    sources = [hist] if os.path.exists(hist) else glob.glob(
        os.path.join(artifact_dir, "checkpoint-*", "trainer_state.json"))
    for src in sources:
        try:
            data = json.load(open(src))
            history = data if isinstance(data, list) else data.get("log_history", [])
            for r in history:
                if "eval_combined" in r and r.get("step") is not None:
                    step_combined[int(r["step"])] = float(r["eval_combined"])
        except (OSError, ValueError):
            continue

    out = []
    for d in glob.glob(os.path.join(artifact_dir, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", d)
        if m and os.path.isdir(d):
            step = int(m.group(1))
            out.append((step, d, step_combined.get(step, float("inf"))))
    out.sort(key=lambda t: t[2])
    return out


def zip_best_checkpoints(artifact_dir: str | os.PathLike, top_n: int = 3,
                         output: str | None = None) -> tuple[str, list]:
    """Writes a single .zip with, for later checkpoint-averaging / model-soup:
      * best_model/  — the root adapter + processor (load_best_model_at_end result),
      * checkpoint-<step>/adapter_config.json + adapter_model.safetensors for the
        top-N checkpoints by eval 'combined' (adapter weights only — NOT the bulky
        optimizer state), and
      * logs/  — the full run logs.

    Returns (output_path, ranking) where ranking is [(step, combined), ...].
    Download this before destroying the instance; checkpoint-averaging in a later
    version just averages the adapter_model.safetensors across these folders.
    """
    import zipfile

    artifact_dir = str(artifact_dir).rstrip("/")
    output = output or f"{artifact_dir}_best{top_n}.zip"
    ranked = rank_checkpoints_by_combined(artifact_dir)
    selected = ranked[:top_n]

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name in os.listdir(artifact_dir):
            fp = os.path.join(artifact_dir, name)
            if os.path.isfile(fp):                       # root adapter + processor
                z.write(fp, arcname=os.path.join("best_model", name))
        logs_dir = os.path.join(artifact_dir, "logs")
        if os.path.isdir(logs_dir):
            for root, _, files in os.walk(logs_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    z.write(fp, arcname=os.path.relpath(fp, artifact_dir))
        for step, d, _ in selected:                       # top-N adapters (no optimizer)
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                fp = os.path.join(d, f)
                if os.path.exists(fp):
                    z.write(fp, arcname=os.path.join(f"checkpoint-{step}", f))

    return output, [(step, combined) for step, _, combined in selected]


# ---------------------------------------------------------------------------
# Generation-based transcription (auto-detect language, Phase-2 style)
# ---------------------------------------------------------------------------

def generate_transcriptions(model, processor, audio_arrays, device, sample_rate=16_000,
                            max_new_tokens=256, num_beams=1):
    """Transcribes a batch of raw waveforms with a (PEFT) Whisper model, letting
    the model auto-detect the language (no forced token) — the same constraints
    as the real evaluation. Returns normalized strings."""
    import torch

    feats = processor.feature_extractor(
        audio_arrays, sampling_rate=sample_rate, return_tensors="pt"
    ).input_features.to(device=device, dtype=model.dtype)
    with torch.no_grad():
        out = model.generate(input_features=feats, max_new_tokens=max_new_tokens,
                             num_beams=num_beams, no_repeat_ngram_size=3)
    return [normalize_text(t) for t in processor.tokenizer.batch_decode(out, skip_special_tokens=True)]
