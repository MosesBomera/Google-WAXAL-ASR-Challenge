"""Shared helpers for mms-ctc — MMS-1B (CTC) fine-tuning for WAXAL.

Stage-1 scope (data foundation): path resolution, HF/audio utils, text
normalization (keeps case + punctuation, unlike MMS's default), the data-cleaning
filters validated on WAXAL by arXiv 2606.02375 §3.2, per-language WER/CER (reused
later), and persistent logging.

Heavy libs (datasets/librosa/jiwer) are imported lazily so `--help` stays fast.
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

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_LANGUAGES = ("lin", "sna", "lug")


# ---------------------------------------------------------------------------
# Paths — robust to both the local repo and a flat vast.ai layout
# ---------------------------------------------------------------------------

def data_dir() -> Path:
    """Folder holding Train.csv / Test.csv / SampleSubmission.csv.
      * vast.ai (how we run):  ./data next to the scripts, CSVs directly inside.
      * local checkout:        data/google-waxal-.../ somewhere above this file.
    Walks up parents so it works regardless of folder depth."""
    candidates = [SCRIPT_DIR / "data"]
    for up in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        candidates.append(up / "data" / "google-waxal-asr-challenge20260630-10570-elxebu")
    for c in candidates:
        if (c / "Train.csv").exists():
            return c
    return candidates[0]


def prepared_dir() -> Path:
    """Where prepare_data.py writes RAW materialized audio chunks (no filtering).
    Download once; EDA and cleaning are separate, cheap, re-runnable steps."""
    return SCRIPT_DIR / "prepared" / "waxal-raw"


def artifacts_dir() -> Path:
    return SCRIPT_DIR / "artifacts" / "mms-1b-all"


# ---------------------------------------------------------------------------
# Logging hygiene + persistence
# ---------------------------------------------------------------------------

def quiet_hf_logging() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in ("huggingface_hub", "datasets", "transformers", "filelock",
                 "httpx", "httpcore", "urllib3", "fsspec", "hf_xet"):
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


def setup_file_logging(log_dir: str | os.PathLike, name: str = "run") -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler)
               and getattr(h, "baseFilename", "") == os.path.abspath(log_path)
               for h in root.handlers):
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
    return log_path


def append_jsonl(path: str | os.PathLike, record: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


def write_json(path: str | os.PathLike, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


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
# Auth / dataset loading
# ---------------------------------------------------------------------------

def login_to_hf(hf_token: str | None) -> None:
    from huggingface_hub import login
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        logger.warning("No --hf-token / HF_TOKEN set; falling back to interactive login.")
        login()


def load_train_csv(path: str | os.PathLike):
    import pandas as pd
    df = pd.read_csv(path, escapechar="\\")
    df = df.rename(columns={"original_split": "split"})
    assert not df["id"].duplicated().any(), "Train.csv has duplicate ids"
    return df


def detect_id_key(example: Mapping[str, Any], wanted_ids: set) -> str:
    for key in ("id", "ID", "utt_id", "utterance_id", "file", "path", "name"):
        if key in example and str(example[key]) in wanted_ids:
            return key
    for key, value in example.items():
        if isinstance(value, str) and value in wanted_ids:
            return key
    raise KeyError(f"No id field matches known ids. Keys: {list(example.keys())}")


def load_streaming_split(dataset_id: str, language: str, split: str, sample_rate: int):
    import datasets
    ds = datasets.load_dataset(dataset_id, name=f"{language}_asr", split=split, streaming=True)
    return ds.cast_column("audio", datasets.Audio(sampling_rate=sample_rate))


def make_audio_to_array(sample_rate: int):
    import librosa
    import numpy as np

    def audio_to_array(audio: Any) -> "np.ndarray":
        sr = sample_rate
        if isinstance(audio, Mapping):
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate", sample_rate))
        elif hasattr(audio, "get_all_samples"):
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
# Text normalization — keeps case + punctuation (protects the CER half)
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def normalize_text(s: Any, lower: bool = False) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace('\\"', '"').replace("\\'", "'")
    s = unicodedata.normalize("NFC", s)
    s = _WS.sub(" ", s).strip()
    if lower:
        s = s.lower()
    return s


# ---------------------------------------------------------------------------
# Data-cleaning filters (WAXAL-NET arXiv 2606.02375 §3.2)
# ---------------------------------------------------------------------------

def clean_verdict(text: str, duration_s: float, *, min_duration: float = 1.5,
                  max_duration: float = 40.0, max_wps: float = 4.0,
                  low_wps: float = 0.5) -> tuple[str | None, dict]:
    """Decides whether to keep a clip. Returns (primary_drop_reason | None, stats).

    Conditions (validated on WAXAL):
      * "empty"      — no transcript,
      * "too_short"  — < min_duration s (< 1.5 s),
      * "too_fast"   — words/sec > max_wps (>4): a physically impossible rate,
                        i.e. a corrupt/misaligned reference,
      * "too_long"   — > max_duration s: kept out of Stage-1 training but logged
                        for forced-alignment recovery in Stage 4 (CTC has no hard
                        cap; we bound it for memory).
    A low words/sec is *flagged* (possible truncated reference) but NOT dropped.

    `stats["violations"]` lists EVERY condition a clip breaks, not just the first.
    Conditions overlap (a 1 s clip holding 20 words is both "too_short" AND
    "too_fast"), so reporting only the first reason hides how often the corrupt-
    reference filter actually fires. Callers should count violations independently
    for diagnosis, and use the returned primary reason only for a single tally.
    """
    n_words = len(text.split())
    wps = n_words / duration_s if duration_s > 0 else 0.0
    violations: list[str] = []
    if n_words == 0:
        violations.append("empty")
    if duration_s < min_duration:
        violations.append("too_short")
    if wps > max_wps:
        violations.append("too_fast")
    if duration_s > max_duration:
        violations.append("too_long")
    stats = {"n_words": n_words, "duration_s": duration_s, "wps": wps,
             "flag_low_wps": wps < low_wps and n_words > 0,
             "violations": violations}
    return (violations[0] if violations else None), stats


def percentiles(values: list[float]) -> dict:
    import numpy as np
    if not values:
        return {}
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "min": float(a.min()), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "mean": float(a.mean())}


# ---------------------------------------------------------------------------
# Metrics (reused in later stages)
# ---------------------------------------------------------------------------

def per_language_metrics(refs: list[str], hyps: list[str], langs: list[str],
                         wer_weight: float = 0.5) -> dict[str, float]:
    import jiwer
    triples = [(normalize_text(r), normalize_text(h), l)
               for r, h, l in zip(refs, hyps, langs) if normalize_text(r)]
    if not triples:
        return {"wer": 1.0, "cer": 1.0, "wer_macro": 1.0, "cer_macro": 1.0,
                "combined": 1.0, "combined_micro": 1.0}
    by = defaultdict(lambda: ([], []))
    all_r, all_h = [], []
    for r, h, l in triples:
        by[l][0].append(r); by[l][1].append(h)
        all_r.append(r); all_h.append(h)
    out: dict[str, float] = {}
    wers, cers = [], []
    for l in sorted(by):
        rs, hs = by[l]
        w = float(jiwer.wer(rs, hs)); c = float(jiwer.cer(rs, hs))
        out[f"wer_{l}"] = w; out[f"cer_{l}"] = c; out[f"n_{l}"] = len(rs)
        wers.append(w); cers.append(c)
    out["wer"] = float(jiwer.wer(all_r, all_h))
    out["cer"] = float(jiwer.cer(all_r, all_h))
    out["wer_macro"] = sum(wers) / len(wers)
    out["cer_macro"] = sum(cers) / len(cers)
    out["combined"] = wer_weight * out["wer_macro"] + (1 - wer_weight) * out["cer_macro"]
    out["combined_micro"] = wer_weight * out["wer"] + (1 - wer_weight) * out["cer"]
    return out
