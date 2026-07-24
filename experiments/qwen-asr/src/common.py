"""Shared helpers for mms-ctc-v2 — fresh MMS-1B (CTC) full FT on WAXAL + FLEURS.

Carries v1 mms-ctc's proven foundation (paths, HF/audio utils, cased+punct text
normalization, WAXAL-validated cleaning filters, logging). The ONE substantive change
is `per_language_metrics`, which is now **board-faithful** — see that section.

Heavy libs (datasets/librosa/jiwer) are imported lazily so `--help` stays fast.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_LANGUAGES = ("lin", "sna", "lug")


# ---------------------------------------------------------------------------
# Paths — robust to both the local repo and a flat vast.ai layout
# ---------------------------------------------------------------------------

def workdir() -> Path:
    """Experiment root — `data/`, `prepared/`, `manifests/`, `vocab/`, `artifacts/`
    hang off this. vast.ai flat: folder/*.py -> folder. repo: .../mms-ctc-v2/src ->
    .../mms-ctc-v2. NEVER use SCRIPT_DIR.parent directly (escapes to `/` when flat)."""
    return SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR


def data_dir() -> Path:
    """Folder with Train.csv / Test.csv / SampleSubmission.csv / metadata CSV — checked
    at every level upward, both flat (<dir>/data) and repo (<dir>/data/google-waxal-…)."""
    candidates: list[Path] = []
    for up in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        candidates.append(up / "data")
        candidates.append(up / "data" / "google-waxal-asr-challenge20260630-10570-elxebu")
    for c in candidates:
        if (c / "Train.csv").exists():
            return c
    return workdir() / "data"


def prepared_dir() -> Path:
    """Where prepare_data.py writes raw audio chunks (WAXAL) and prepare_external.py
    adds FLEURS under train/fleurs-<lang>/."""
    return workdir() / "prepared" / "waxal-raw"


def artifacts_dir() -> Path:
    return workdir() / "artifacts" / "mms-1b-all"


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
    df = pd.read_csv(path, escapechar="\\")            # transcripts have embedded commas/quotes
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
# Text normalization — TRAINING targets: keep case + punctuation (protects CER)
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")

# Junk chars (EDA Finding 5): 39 chars occur <20x across 38k clips = transcription
# noise. MAP them to sane ASCII rather than pollute the vocab / emit [UNK] garbage.
# KEEP the legit French-influenced Lingala diacritics (é à ï ê ç ù û ü ò ó á ì ú).
CHAR_MAP = {
    "“": '"', "”": '"',
    "‘": "'", "’": "'",
    "«": '"', "»": '"',
    "`": "'",
    "×": "x",
    "œ": "oe", "Œ": "OE",
    "þ": "th",
    "Ŋ": "Ng",
    "ĝ": "g",
    "ā": "a",
    "ĺ": "l", "Ĺ": "L",
    "Ķ": "K", "ķ": "k",
    "ε": "ɛ",   # Greek epsilon look-alike for Lingala's open-e (LRSC); canonicalize to one grapheme
    # Wrong-script look-alikes / typos in FLEURS+LRSC text (each <20x in 47k transcripts) — map to
    # the intended Latin letter so they don't become their own rare CTC output classes. These are
    # NOT the legit French Lingala diacritics above (é à ï ê ç ù û ü ò ó á ì ú), which we KEEP.
    "å": "a", "ǎ": "a", "ã": "a", "ă": "a",   # a-ring / a-caron / Portuguese/Romanian a → a
    "õ": "o", "ö": "o", "Õ": "O",             # Portuguese/German o → o/O
    "ł": "l",                                  # Polish l-stroke → l
    "ş": "s",                                  # Turkish s-cedilla → s
    "ğ": "g",                                  # Turkish g-breve → g
    "Ü": "U",                                  # German U-umlaut → U
}
STRIP_CHARS = frozenset({"⭐", "️"})


def normalize_chars(s: str) -> str:
    """Junk-character map + emoji strip. Idempotent."""
    if not s:
        return s
    out = []
    for ch in s:
        if ch in STRIP_CHARS:
            continue
        out.append(CHAR_MAP.get(ch, ch))
    return "".join(out)


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


def normalize_for_ctc(s: Any) -> str:
    """Normalization for CTC LABELS + the vocab (training-side). Keeps case+punct,
    applies the junk-char map. This is what the model learns to EMIT — distinct from
    the board-faithful SCORING normalization below (which does NOT char-map)."""
    return _WS.sub(" ", normalize_chars(normalize_text(s))).strip()


# ---------------------------------------------------------------------------
# Data-cleaning filters (WAXAL-NET arXiv 2606.02375 §3.2)
# ---------------------------------------------------------------------------

def clean_verdict(text: str, duration_s: float, *, min_duration: float = 1.5,
                  max_duration: float = 40.0, max_wps: float = 4.0,
                  low_wps: float = 0.5) -> tuple[str | None, dict]:
    """Keep/drop a clip. Returns (primary_drop_reason | None, stats). Conditions
    (WAXAL-validated): empty | too_short (<1.5s) | too_fast (wps>4 = corrupt/misaligned
    reference) | too_long (>max, bounded for memory). Low wps is flagged, not dropped.
    `stats["violations"]` lists EVERY broken condition (they overlap)."""
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
# Metrics — BOARD-FAITHFUL (reverse-engineered by scorer_probe.py, 2026-07-18)
# ---------------------------------------------------------------------------
# The WAXAL leaderboard scores, MICRO-pooled over all test clips:
#   * WER on LOWERCASED text, punctuation kept   (case-INsensitive)
#   * CER on CASED text,       punctuation kept   (case-sensitive)
#   combined_error = 0.5*WER + 0.5*CER ; the board displays 1 - combined_error.
# Reproduction (v1 cased submission): board CER 0.1037 ≈ cased 0.1035 (but ≠ any
# lowercased scheme); board WER 0.3159 ≈ lowercase-keep-punct 0.3139 (cased 0.3340
# is far off). We deliberately do NOT char-map when scoring — the board doesn't,
# even though our training targets ARE char-mapped (normalize_for_ctc).

def _score_norm(s: Any, *, lower: bool) -> str:
    s = unicodedata.normalize("NFKC", "" if s is None else str(s))
    if lower:
        s = s.lower()
    return _WS.sub(" ", s).strip()


def per_language_metrics(refs: list[str], hyps: list[str], langs: list[str],
                         wer_weight: float = 0.5) -> dict[str, float]:
    """Board-faithful WER/CER. `combined` (= the board metric) is MICRO-pooled:
    0.5·WER(lowercased) + 0.5·CER(cased). Per-language + macro also returned for
    diagnosis; select checkpoints on `combined`."""
    import jiwer

    rows = [(r, h, l) for r, h, l in zip(refs, hyps, langs) if _score_norm(r, lower=False)]
    if not rows:
        return {"wer": 1.0, "cer": 1.0, "combined": 1.0,
                "wer_macro": 1.0, "cer_macro": 1.0, "combined_macro": 1.0}

    by: dict[str, list] = defaultdict(list)
    for r, h, l in rows:
        by[l].append((r, h))

    out: dict[str, float] = {}
    wers, cers = [], []
    aw_r, aw_h, ac_r, ac_h = [], [], [], []       # pooled: WER lists lowercased, CER lists cased
    for l in sorted(by):
        rw = [_score_norm(r, lower=True) for r, _ in by[l]]
        hw = [_score_norm(h, lower=True) for _, h in by[l]]
        rc = [_score_norm(r, lower=False) for r, _ in by[l]]
        hc = [_score_norm(h, lower=False) for _, h in by[l]]
        w, c = float(jiwer.wer(rw, hw)), float(jiwer.cer(rc, hc))
        out[f"wer_{l}"], out[f"cer_{l}"], out[f"n_{l}"] = w, c, len(by[l])
        wers.append(w); cers.append(c)
        aw_r += rw; aw_h += hw; ac_r += rc; ac_h += hc

    out["wer"] = float(jiwer.wer(aw_r, aw_h))         # micro, board-faithful
    out["cer"] = float(jiwer.cer(ac_r, ac_h))
    out["combined"] = wer_weight * out["wer"] + (1 - wer_weight) * out["cer"]   # THE metric
    out["wer_macro"] = sum(wers) / len(wers)
    out["cer_macro"] = sum(cers) / len(cers)
    out["combined_macro"] = wer_weight * out["wer_macro"] + (1 - wer_weight) * out["cer_macro"]
    return out
