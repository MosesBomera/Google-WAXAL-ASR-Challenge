"""mms-ctc-v2 / Stage 1e — fold external FLEURS speech into the training set.

FLEURS (google/fleurs) is read-aloud Wikipedia speech — VERIFIED disjoint from WAXAL by
verify_external_data.py (0 train/val/test overlap), so it is safe, disclosable external
data (CC-BY-4.0). We add it for lin/sna/lug to give the model extra speakers and lexical
coverage — most valuable for Lingala (our weakest, most speaker-poor language).

It is NOT enough to append a manifest row: train_ctc.py loads audio from on-disk Arrow
chunks (prepare_data.py's `prepared/{train,validation}/<lang>/`) and joins them to the
manifest by id. So this script must emit FLEURS in the SAME schema and append the ids to
manifests/train.csv (VAL IS NEVER TOUCHED — it stays 100% WAXAL speaker-disjoint):

    Features({id, audio:Audio(16k), transcription, language, duration:f32,
              speaker_id, gender})   ->  prepared/train/fleurs-<lang>/

Two guards:
  • ids are prefixed `fleurs-...` (never lin_/sna_/lug_) so external data can't be mistaken
    for WAXAL and can't collide.
  • VOCAB COVERAGE: any FLEURS character outside our FROZEN cased vocab would silently become
    [UNK] and rot the CER half of the metric — so we report the out-of-vocab char set and, in
    --dry-run, write nothing until you've decided how to handle it.

    python prepare_external.py --dry-run            # report hours + OOV chars, write nothing
    python prepare_external.py                       # ingest lin+sna+lug FLEURS
    python prepare_external.py --languages lin       # Lingala only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter

from common import (
    login_to_hf,
    normalize_for_ctc,
    normalize_text,
    prepared_dir,
    quiet_hf_logging,
    workdir,
)

logger = logging.getLogger("prepare_external")

# our lang code -> FLEURS config (FLoRes region codes)
FLEURS_CONFIGS = {"lin": "ln_cd", "sna": "sn_zw", "lug": "lg_ug"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--languages", nargs="+", default=list(FLEURS_CONFIGS),
                   help="Our lang codes to ingest (default lin sna lug).")
    p.add_argument("--dataset-id", default="google/fleurs")
    p.add_argument("--fleurs-splits", nargs="+", default=["train", "validation", "test"],
                   help="FLEURS splits to pull (all are external to WAXAL → all usable as TRAIN).")
    p.add_argument("--audio-dir", default=str(prepared_dir()), help="Where prepared/ chunks live.")
    p.add_argument("--manifest-dir", default=str(workdir() / "manifests"))
    p.add_argument("--vocab", default=str(workdir() / "vocab" / "vocab.json"))
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--dir-prefix", default="fleurs")
    p.add_argument("--dry-run", action="store_true", help="Report hours + OOV chars; write nothing.")
    p.add_argument("--hf-token", default=None)
    return p.parse_args()


def vocab_chars(path: str) -> set[str]:
    """The single-character tokens the CTC head can actually emit (excludes [PAD]/[UNK]/<s>/</s>);
    '|' is the word delimiter and stands in for space.

    In the rebuild pipeline this runs BEFORE build_vocab.py creates vocab.json (the OOV report is
    informational — "which external chars are new?"), so a missing vocab is expected, not an error:
    return the empty set (→ every char reads as OOV, the report still prints, build_vocab.py folds
    them all in afterward). See plan-mms-ctc-v2.md §5 / RUN.md §3-4 for the ordering."""
    if not os.path.isfile(path):
        logger.info("No vocab yet at %s — reporting all chars as new (build_vocab.py runs next).", path)
        return set()
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    return {t for t in vocab if len(t) == 1}


def _norm_gender(_g) -> str:
    # FLEURS gender is unreliable to map across versions and is UNUSED in training (the speaker
    # split is already frozen) — record "unknown" rather than risk a wrong label.
    return "unknown"


def _no_audio_decode(ds):
    import datasets
    for col, feat in dict(getattr(ds, "features", None) or {}).items():
        if isinstance(feat, datasets.Audio):
            ds = ds.cast_column(col, datasets.Audio(decode=False))
    return ds


def _count_oov(text: str, vchars: set[str], oov: Counter) -> None:
    for ch in normalize_for_ctc(text):
        if ch not in vchars and ch != " ":
            oov[ch] += 1


def scan_language(args, lang: str, config: str, vchars: set[str]):
    """--dry-run path: TEXT ONLY (no ffmpeg). Duration from FLEURS `num_samples`, text from
    `raw_transcription`. Reports would-be hours + out-of-vocab chars; writes nothing."""
    import datasets

    sr, oov, n, total_s = args.sample_rate, Counter(), 0, 0.0
    for split in args.fleurs_splits:
        ds = _no_audio_decode(datasets.load_dataset(args.dataset_id, config, split=split, streaming=True))
        for ex in ds:
            text = normalize_text(ex.get("raw_transcription") or ex.get("transcription") or "")
            if not text.strip():
                continue
            _count_oov(text, vchars, oov)
            total_s += (ex.get("num_samples") or 0) / sr
            n += 1
    hours = total_s / 3600
    logger.info("[%s] %d clips, %.1f h | OOV chars: %d distinct%s", lang, n, hours,
                len(oov), (" -> " + repr(dict(oov.most_common(20)))) if oov else " (all in vocab ✓)")
    return None, [], oov, hours


def ingest_language(args, lang: str, config: str, vchars: set[str]):
    """Full path (needs ffmpeg): materialize FLEURS into prepared/ + return manifest rows."""
    import datasets

    split_expr = "+".join(args.fleurs_splits)
    logger.info("[%s] loading FLEURS %s (%s)…", lang, config, split_expr)
    raw = datasets.load_dataset(args.dataset_id, config, split=split_expr)
    raw = raw.cast_column("audio", datasets.Audio(sampling_rate=args.sample_rate))

    sr = args.sample_rate
    target = datasets.Features({
        "id": datasets.Value("string"),
        "audio": datasets.Audio(sampling_rate=sr),
        "transcription": datasets.Value("string"),
        "language": datasets.Value("string"),
        "duration": datasets.Value("float32"),
        "speaker_id": datasets.Value("string"),
        "gender": datasets.Value("string"),
    })

    def _transform(ex, idx):
        arr = ex["audio"]["array"]
        dur = len(arr) / sr
        # raw_transcription keeps case + punctuation → matches our cased vocab (transcription is
        # lowercased/stripped). normalize_text mirrors prepare_data.py's stored form.
        text = normalize_text(ex.get("raw_transcription") or ex.get("transcription") or "")
        return {
            "id": f"{args.dir_prefix}-{lang}-{idx}",       # never lin_/sna_/lug_ ; unique per lang
            "audio": {"array": arr, "sampling_rate": sr},
            "transcription": text,
            "language": lang,                              # so per-language metrics bucket right
            "duration": float(dur),
            "speaker_id": f"{args.dir_prefix}-{lang}",     # synthetic; train-only, never in val
            "gender": _norm_gender(ex.get("gender")),
        }

    ds = raw.map(_transform, with_indices=True, remove_columns=raw.column_names,
                 features=target, num_proc=1, desc=f"{lang}: to prepared schema")
    ds = ds.filter(lambda ex: ex["transcription"].strip() != "", num_proc=1)

    hours = sum(ds["duration"]) / 3600
    # OOV against the FROZEN vocab, using the SAME normalization the collator applies at train time
    oov: Counter = Counter()
    for t in ds["transcription"]:
        _count_oov(t, vchars, oov)
    rows = [{"id": _id, "language": lang, "split": "train", "speaker_id": f"{args.dir_prefix}-{lang}",
             "gender": "unknown", "duration_s": round(float(d), 3),
             "n_words": len(t.split()), "wps": round(len(t.split()) / d, 3) if d > 0 else 0.0}
            for _id, d, t in zip(ds["id"], ds["duration"], ds["transcription"])]

    logger.info("[%s] %d clips, %.1f h | OOV chars: %d distinct%s", lang, len(ds), hours,
                len(oov), (" -> " + repr(dict(oov.most_common(20)))) if oov else " (all in vocab ✓)")

    out = os.path.join(args.audio_dir, "train", f"{args.dir_prefix}-{lang}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.save_to_disk(out)
    logger.info("[%s] saved -> %s", lang, out)
    return ds, rows, oov, hours


def append_manifest(manifest_dir: str, new_rows: list[dict]) -> None:
    import pandas as pd

    path = os.path.join(manifest_dir, "train.csv")
    base = pd.read_csv(path, dtype={"id": str})
    new = pd.DataFrame(new_rows)[list(base.columns)]         # align column order; error if a col is missing
    merged = pd.concat([base[~base["id"].isin(set(new["id"]))], new], ignore_index=True)
    merged.to_csv(path, index=False)
    logger.info("manifest %s: %d -> %d rows (+%d external)", path, len(base), len(merged), len(new))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()
    login_to_hf(args.hf_token)

    vchars = vocab_chars(args.vocab)
    logger.info("Frozen vocab: %d emittable characters", len(vchars))

    all_rows: list[dict] = []
    total_oov: Counter = Counter()
    summary = {}
    for lang in args.languages:
        config = FLEURS_CONFIGS.get(lang)
        if not config:
            logger.warning("No FLEURS config known for %r — skipping.", lang)
            continue
        scan = scan_language if args.dry_run else ingest_language
        _ds, rows, oov, hours = scan(args, lang, config, vchars)
        all_rows.extend(rows)
        total_oov.update(oov)
        summary[lang] = {"config": config, "clips": len(rows), "hours": round(hours, 2),
                         "oov_distinct": len(oov), "oov_top": dict(oov.most_common(20))}

    logger.info("=" * 70)
    if total_oov:
        logger.warning("OUT-OF-VOCAB characters across all FLEURS text: %d distinct -> %s",
                       len(total_oov), dict(total_oov.most_common(30)))
        logger.warning("Decide before training: extend common.CHAR_MAP to map these, or accept "
                       "[UNK] (rots CER). Re-run when handled.")
    else:
        logger.info("No out-of-vocab characters — FLEURS text fits the frozen vocab cleanly ✓")

    write_path = os.path.join(args.manifest_dir, "external_report.json")
    with open(write_path, "w", encoding="utf-8") as f:
        json.dump({"languages": summary, "total_clips": len(all_rows),
                   "total_oov_distinct": len(total_oov)}, f, ensure_ascii=False, indent=2)
    logger.info("Report -> %s", write_path)

    if args.dry_run:
        logger.info("DRY RUN — no chunks written, manifest untouched. Total would-be: %d clips.", len(all_rows))
        return
    append_manifest(args.manifest_dir, all_rows)
    logger.info("Done. Train with the usual: python train_ctc.py …  (val unchanged, still 0.205 yardstick)")


if __name__ == "__main__":
    main()
