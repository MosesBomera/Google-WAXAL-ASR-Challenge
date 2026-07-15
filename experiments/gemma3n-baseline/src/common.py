"""Shared helpers used by prepare_data.py, train.py, and infer.py.

Kept dependency-light at import time (heavy libs like torch/librosa/datasets
are imported lazily inside functions) so `--help` on any script stays fast.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


def quiet_hf_logging() -> None:
    """Mutes the very verbose default output from the HF stack: download
    progress bars, "Resolving data files" spinners, "sending unauthenticated
    requests" warnings, tokenizers fork-parallelism notices, etc.

    Call this once at the very top of each script's main(), before anything
    else touches huggingface_hub/datasets/transformers — some of this only
    takes effect if the env vars are set before those libs are imported.
    """
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


def fmt_duration(seconds: float) -> str:
    """Formats seconds as e.g. '1h 05m 12s' / '5m 03s' / '42s'."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def detect_id_key(example: Mapping[str, Any], wanted_ids: set) -> str:
    """Finds which field of an HF example holds a known id, e.g. an id from
    Train.csv or SampleSubmission.csv (both use ids like 'lug_96114').
    """
    for key in ("id", "ID", "utt_id", "utterance_id", "file", "path", "name"):
        if key in example and str(example[key]) in wanted_ids:
            return key
    for key, value in example.items():
        if isinstance(value, str) and value in wanted_ids:
            return key
    raise KeyError(
        f"Could not find an id field matching known ids. "
        f"Example keys were: {list(example.keys())}"
    )


def login_to_hf(hf_token: str | None) -> None:
    """Authenticates with the Hub. Gemma is gated: accept its license first
    at https://huggingface.co/google/gemma-3n-E2B-it before this will work.
    """
    from huggingface_hub import login

    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        logger.warning(
            "No --hf-token / HF_TOKEN set; falling back to interactive login."
        )
        login()


# ---------------------------------------------------------------------------
# Audio / chat formatting helpers
# ---------------------------------------------------------------------------

def make_audio_to_array(sample_rate: int):
    """Returns a fn that extracts a mono float32 array from a datasets Audio
    value, at ``sample_rate``. Handles both the classic dict form
    ({'array', 'sampling_rate'}) and the torchcodec AudioDecoder returned by
    datasets >= 4.
    """
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
        if arr.ndim > 1:  # downmix channels to mono
            arr = arr.mean(axis=0)
        arr = arr.flatten()
        if sr != sample_rate:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=sample_rate)
        return arr

    return audio_to_array


def make_formatters(audio_to_array, system_message: str, user_message: str):
    """Builds (format_for_chat, format_batch) closed over the given prompts.

    format_for_chat turns one {'audio', 'transcription', ...} example into a
    chat-style {'messages': [...]} example. Kept separate from the saved
    dataset (prepare_data.py does NOT bake 'messages' in) so the raw audio
    is stored on disk exactly once, not duplicated inside the message content.
    """

    def format_for_chat(example: Mapping[str, Any]) -> dict[str, Any]:
        audio_array = audio_to_array(example["audio"])
        return {
            **example,
            "messages": [
                {"role": "system",
                 "content": [{"type": "text", "text": system_message}]},
                {"role": "user",
                 "content": [
                     {"type": "audio", "audio": audio_array},
                     {"type": "text", "text": user_message},
                 ]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": str(example["transcription"])}]},
            ],
        }

    def format_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        num = len(batch["transcription"])
        examples = [{k: batch[k][i] for k in batch} for i in range(num)]
        formatted = [format_for_chat(ex) for ex in examples]
        return {k: [ex[k] for ex in formatted] for k in formatted[0]}

    return format_for_chat, format_batch


def load_train_csv(path: str):
    """Loads Zindi's Train.csv: columns id, transcription, language,
    original_split. Note it uses backslash-escaped quotes inside
    transcriptions (e.g. \\"blue\\"), not standard doubled-quote CSV escaping
    — pandas' default engine mis-parses it without escapechar='\\\\'.
    """
    import pandas as pd

    df = pd.read_csv(path, escapechar="\\")
    df = df.rename(columns={"original_split": "split"})
    assert not df["id"].duplicated().any(), "Train.csv has duplicate ids"
    return df


def load_streaming_split(
    dataset_id: str, language: str, split: str, sample_rate: int
):
    """Loads a single language/split from the Hub as a streaming dataset,
    resampled to sample_rate. Does NOT decode audio or apply chat-formatting
    — callers should do that themselves (materializing / mapping raw/decoded
    audio through Arrow-backed ops like interleave_datasets can break on the
    torchcodec AudioDecoder object; normalize with audio_to_array first).
    """
    import datasets

    ds = datasets.load_dataset(
        dataset_id, name=f"{language}_asr", split=split, streaming=True
    )
    return ds.cast_column("audio", datasets.Audio(sampling_rate=sample_rate))
