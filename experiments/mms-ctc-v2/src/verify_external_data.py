"""mms-ctc-v2 / Stage 1d — is an EXTERNAL dataset safe to train on, or WAXAL in disguise?

The rules allow supplementing WAXAL with public open-source data (disclosed), BUT
forbid training on anything that contains the WAXAL **test** audio/transcripts. Some
African-language sets MIGHT be re-releases of the WAXAL collection this challenge is
carved from (e.g. DigitalUmuganda contributed to WAXAL AND ships its own AfriVoice) — but
a shared CONTRIBUTOR is NOT proof of shared DATA: they may have collected separate
recordings. So we decide EMPIRICALLY, not by authorship.

The cheap first test is **transcript overlap** (no audio decode). But interpret it
carefully: a match means a shared SENTENCE, not necessarily shared AUDIO — two independent
collections using the same image prompts produce the same sentences from different speakers.
So transcript overlap flags RISK; the ~1.008 s corruption fingerprint and a speaker-id /
audio-duplicate check are what actually prove shared recordings. We check three WAXAL
reference sets, per external `source` bucket when the column exists:

  • WAXAL **test** transcripts (streamed from the Hub) -> ANY hit = breach material.
  • WAXAL **train** transcripts (Train.csv)           -> high overlap = re-release
                                                          (precautionary: avoid; it may
                                                          also carry hidden Phase-2 clips).
  • our speaker-disjoint **val** transcripts           -> any hit poisons the yardstick.

An optional sampled audio probe corroborates via WAXAL's 1.008 s corruption fingerprint:
if the external set has its own spike of ~1 s clips, it came off the same pipeline.

NOTE ON THE TEST LABELS: reading the public test transcripts *to exclude* contaminated
external data is a compliance check, not training on them — they are never used as
targets. See [[waxal-public-test-labels-rule]].

The decisive "same recording" test is CONTENT-based (transcript + audio DURATION), speaker-id-
independent — duration reads the WAV header via soundfile (`pip install soundfile`), NO ffmpeg
needed (AfriVoice is WAV). speaker_id is reported only as corroboration, with a self-check
(do ids agree on confirmed duplicates?) that reveals whether the id scheme is global or per-dataset.

Verify each ORIGINAL source directly (not the aggregator):
    # FLEURS Lingala — expect INDEPENDENT (read Wikipedia speech, not WAXAL):
    python verify_external_data.py --external-id google/fleurs --external-config ln_cd --audio-sample 0
    # AfriVoice Lingala — decide by evidence (config is 'ln'); soundfile enables the duration check:
    python verify_external_data.py --external-id DigitalUmuganda/AfriVoice --external-config ln --audio-sample 0
    # LRSC (Mendeley) downloaded to disk:
    python verify_external_data.py --external-id ./data/lrsc --from-disk --audio-sample 0
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict

from common import data_dir, load_train_csv, login_to_hf, quiet_hf_logging, workdir, write_json

logger = logging.getLogger("verify_external")

TEXT_FIELDS = ("transcription", "text", "sentence", "transcript", "normalized_text")
SOURCE_FIELDS = ("source", "origin", "dataset", "corpus", "subset", "provenance")
SPEAKER_FIELDS = ("speaker_id", "speaker", "client_id", "client", "spk")
LANG_FIELDS = ("locale", "language", "lang", "lang_id", "locale_id")
CORRUPTION_BAND = (0.90, 1.10)   # WAXAL's truncated clips cluster at ~1.008 s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--external-id", default="KasuleTrevor/Lingala_100hrs",
                   help="HF dataset id, OR a local path when --from-disk is set. Point this at the "
                        "ORIGINAL source (google/fleurs, DigitalUmuganda/AfriVoice, LRSC) — verify "
                        "each source directly rather than trusting an aggregator.")
    p.add_argument("--external-config", default=None, help="e.g. ln_cd for google/fleurs.")
    p.add_argument("--external-splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument("--from-disk", action="store_true",
                   help="Load --external-id as a local datasets dir (load_from_disk) — for LRSC "
                        "or any Mendeley set downloaded to disk.")
    p.add_argument("--external-data-files", default=None,
                   help="Glob passed to load_dataset(data_files=...) to stream only PART of a repo — "
                        "e.g. 'Lingala/**' for AfriVoice, so we don't pull all 6 languages.")
    p.add_argument("--external-lang-field", default=None,
                   help="Field to filter the external set by language (auto: locale/language/lang).")
    p.add_argument("--external-lang-prefix", default=None,
                   help="Keep only external rows whose language field starts with this — e.g. 'ln' "
                        "for AfriVoice Lingala (rows carry locale like 'ln_CD').")
    p.add_argument("--waxal-languages", nargs="+", default=["lin"],
                   help="WAXAL language(s) the external set should be disjoint from.")
    p.add_argument("--train-csv", default=str(data_dir() / "Train.csv"))
    p.add_argument("--val-csv", default=str(workdir() / "manifests" / "val.csv"))
    p.add_argument("--waxal-metadata", default=str(data_dir() / "waxal_clip_metadata.csv"),
                   help="Per-clip metadata (speaker_id + duration) — enables the SPEAKER-overlap and "
                        "(speaker+transcript) clip-duplicate checks, the decisive 'same recordings' test.")
    p.add_argument("--confirm-audio-dur", action=argparse.BooleanOptionalAction, default=True,
                   help="For (speaker+transcript) duplicate candidates, read the external clip's WAV "
                        "header (no ffmpeg) and confirm its duration matches WAXAL's — proves same recording.")
    p.add_argument("--duration-tol", type=float, default=0.05, help="Seconds tolerance for the match.")
    p.add_argument("--waxal-dataset-id", default="google/WaxalNLP")
    p.add_argument("--check-test", action=argparse.BooleanOptionalAction, default=True,
                   help="Stream WAXAL test transcripts from the Hub for the breach check "
                        "(text only; never used as training targets).")
    p.add_argument("--audio-sample", type=int, default=200,
                   help="Clips/split to decode for the corruption-fingerprint probe (0 = skip).")
    p.add_argument("--max-external", type=int, default=0, help="Cap external rows scanned (0 = all).")
    p.add_argument("--log-every", type=int, default=200, help="Heartbeat: log progress every N external rows.")
    p.add_argument("--rerelease-frac", type=float, default=0.02,
                   help="Train-overlap fraction above which a source is flagged a likely re-release.")
    p.add_argument("--output-dir", default=None, help="Default: <workdir>/logs")
    p.add_argument("--hf-token", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# text keys
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def loose_key(s: str) -> str:
    """Aggressive match key: NFKC, lowercase, punctuation->space, whitespace collapsed.
    Diacritics are KEPT (dropping them would create false-positive matches)."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return " ".join(_PUNCT.sub(" ", s).split())


def _text_only(ds):
    """Disable Audio-column decoding on a streaming dataset so iterating reads only the
    transcripts — otherwise `datasets` eagerly decodes audio (torchcodec/ffmpeg), which a
    plain laptop lacks. Bytes still stream, but nothing is decoded."""
    import datasets

    for col, feat in dict(getattr(ds, "features", None) or {}).items():
        if isinstance(feat, datasets.Audio):
            ds = ds.cast_column(col, datasets.Audio(decode=False))
    return ds


def _first_field(example: dict, candidates) -> str | None:
    for c in candidates:
        if c in example and example[c] not in (None, ""):
            return c
    # case-insensitive fallback
    lower = {k.lower(): k for k in example}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


# ---------------------------------------------------------------------------
# WAXAL reference key sets
# ---------------------------------------------------------------------------

def build_waxal_refs(args) -> dict:
    import pandas as pd

    df = load_train_csv(args.train_csv)   # escapechar-aware; transcripts have embedded commas/quotes
    df = df[df["language"].isin(args.waxal_languages)]
    train_keys = {loose_key(t) for t in df["transcription"].dropna()}
    train_keys.discard("")

    val_keys: set[str] = set()
    val_speakers: set[str] = set()
    if os.path.isfile(args.val_csv):
        vdf = pd.read_csv(args.val_csv, dtype={"id": str, "speaker_id": str})
        val_ids = set(vdf["id"])
        if "speaker_id" in vdf.columns:
            val_speakers = {s for s in vdf["speaker_id"].dropna() if s}
        val_keys = {loose_key(t) for _id, t in zip(df["id"].astype(str), df["transcription"])
                    if _id in val_ids}
        val_keys.discard("")
    logger.info("WAXAL refs (%s): %d train transcripts, %d val transcripts, %d val speakers",
                "+".join(args.waxal_languages), len(train_keys), len(val_keys), len(val_speakers))

    test_keys: set[str] = set()
    if args.check_test:
        import datasets
        for lang in args.waxal_languages:
            try:
                stream = datasets.load_dataset(args.waxal_dataset_id, name=f"{lang}_asr",
                                               split="test", streaming=True)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not stream WAXAL %s test split (%s) — breach check SKIPPED.", lang, e)
                continue
            n = 0
            for ex in _text_only(stream):
                tf = _first_field(ex, TEXT_FIELDS)
                if tf:
                    k = loose_key(ex[tf])
                    if k:
                        test_keys.add(k)
                        n += 1
                        if n % 1000 == 0:
                            logger.info("  …WAXAL %s test: %d transcripts pulled", lang, n)
            logger.info("WAXAL %s test: %d transcripts pulled (compliance check only)", lang, n)

    # For the "same recordings" test we use CONTENT (transcript + duration), NOT speaker_id —
    # those UIDs might be minted per-dataset, so a shared person could carry different ids (and a
    # duplicate could carry different ids). text_dur: loose_transcript -> WAXAL durations. text_spk:
    # loose_transcript -> WAXAL speaker_ids, used only to SELF-CHECK whether the id scheme is global
    # (do the ids agree on clips we independently confirm as duplicates?).
    speakers: set[str] = set()
    text_dur: dict[str, list[float]] = defaultdict(list)
    text_spk: dict[str, set[str]] = defaultdict(set)
    if os.path.isfile(args.waxal_metadata):
        meta = pd.read_csv(args.waxal_metadata, dtype={"speaker_id": str, "id": str})
        meta = meta[meta["language"].isin(args.waxal_languages)]
        speakers = {s for s in meta["speaker_id"].dropna() if s}
        id2spk = dict(zip(meta["id"], meta["speaker_id"]))
        id2dur = dict(zip(meta["id"], meta["duration_s"]))
        for _id, t in zip(df["id"].astype(str), df["transcription"]):
            k = loose_key(t)
            if not k:
                continue
            d = id2dur.get(_id)
            if d is not None and str(d) != "nan":
                text_dur[k].append(round(float(d), 2))
            spk = id2spk.get(_id)
            if spk:
                text_spk[k].add(spk)
        logger.info("WAXAL refs: %d distinct speakers, %d distinct transcripts w/ durations",
                    len(speakers), len(text_dur))
    else:
        logger.warning("No metadata at %s — duration/speaker checks DISABLED.", args.waxal_metadata)

    return {"train": train_keys, "val": val_keys, "test": test_keys, "speakers": speakers,
            "val_speakers": val_speakers, "text_dur": dict(text_dur), "text_spk": dict(text_spk)}


# ---------------------------------------------------------------------------
# external scan (text)
# ---------------------------------------------------------------------------

def _external_splits(args):
    """Yield (split_name, iterable). Handles HF streaming and local load_from_disk (LRSC)."""
    import datasets

    if args.from_disk:
        obj = datasets.load_from_disk(args.external_id)
        if isinstance(obj, datasets.DatasetDict):
            yield from obj.items()
        else:
            yield "all", obj
        return
    load_kw = {"data_files": args.external_data_files} if args.external_data_files else {}
    for split in args.external_splits:
        try:
            yield split, datasets.load_dataset(args.external_id, name=args.external_config,
                                               split=split, streaming=True, **load_kw)
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping external split %r (%s)", split, e)


def _audio_field(ex) -> str | None:
    for k, v in ex.items():
        if isinstance(v, dict) and ({"array", "bytes", "path"} & set(v.keys())):
            return k
    return None


def _wav_duration(audio) -> float | None:
    """Duration from a WAV/FLAC header WITHOUT ffmpeg (soundfile reads only the header).
    Operates on the decode=False struct {'bytes':..., 'path':...}. None if unreadable."""
    if not isinstance(audio, dict):
        return None
    try:
        import io
        import soundfile as sf
        b = audio.get("bytes")
        if b:
            info = sf.info(io.BytesIO(b))
            return info.frames / float(info.samplerate)
        p = audio.get("path")
        if p and os.path.isfile(p):
            info = sf.info(p)
            return info.frames / float(info.samplerate)
    except Exception:      # noqa: BLE001  — missing soundfile, non-WAV, truncated bytes, etc.
        return None
    return None


def scan_external(args, refs: dict) -> dict:
    per_source = defaultdict(lambda: {"n": 0, "train_hits": 0, "val_hits": 0, "test_hits": 0,
                                      "no_match": 0, "test_hit_examples": [], "ext_speakers": set(),
                                      "overlap_speakers": set(), "val_speaker_clips": 0,
                                      "dup_clips": 0, "dup_confirmed": 0, "dup_unconfirmed": 0,
                                      "dup_spk_agree": 0, "dup_spk_disagree": 0})
    total = test_tot = val_tot = newtot = dup_tot = skipped_lang = 0
    text_field = source_field = speaker_field = audio_field = lang_field = None
    schema_logged = False
    t0 = time.time()

    for split, stream in _external_splits(args):
        logger.info("Scanning external split %r …", split)
        for ex in _text_only(stream):
            if not schema_logged:
                text_field = _first_field(ex, TEXT_FIELDS)
                source_field = _first_field(ex, SOURCE_FIELDS)
                speaker_field = _first_field(ex, SPEAKER_FIELDS)
                audio_field = _audio_field(ex)
                lang_field = args.external_lang_field or _first_field(ex, LANG_FIELDS)
                schema_logged = True
                logger.info("External schema: transcript=%r speaker=%r audio=%r lang=%r source=%r",
                            text_field, speaker_field, audio_field, lang_field, source_field)
                if text_field is None:
                    raise SystemExit(f"No transcript field in {args.external_id} (looked for {TEXT_FIELDS}).")
                if args.external_lang_prefix and lang_field is None:
                    raise SystemExit(f"--external-lang-prefix set but no language field found "
                                     f"(looked for {LANG_FIELDS}); pass --external-lang-field.")
            # Language filter (AfriVoice mixes 6 langs under one config; keep only the target).
            if args.external_lang_prefix:
                lv = str(ex.get(lang_field) or "").lower()
                if not lv.startswith(args.external_lang_prefix.lower()):
                    skipped_lang += 1
                    continue
            src = str(ex.get(source_field, "all")) if source_field else "all"
            k = loose_key(ex[text_field])
            spk = str(ex.get(speaker_field) or "") if speaker_field else ""
            b = per_source[src]
            b["n"] += 1
            total += 1
            if k:
                if k in refs["test"]:
                    b["test_hits"] += 1; test_tot += 1
                    if len(b["test_hit_examples"]) < 5:
                        b["test_hit_examples"].append(ex[text_field][:100])
                if k in refs["train"]:
                    b["train_hits"] += 1
                if k in refs["val"]:
                    b["val_hits"] += 1; val_tot += 1
                if k not in refs["test"] and k not in refs["train"] and k not in refs["val"]:
                    b["no_match"] += 1; newtot += 1
            # Speaker overlap — REPORTED only (UIDs may be per-dataset → not decisive; see verdict).
            if spk:
                b["ext_speakers"].add(spk)
                if spk in refs.get("speakers", ()):
                    b["overlap_speakers"].add(spk)
                if spk in refs.get("val_speakers", ()):
                    b["val_speaker_clips"] += 1
            # DUPLICATE = same transcript, confirmed by same audio duration. SPEAKER-INDEPENDENT.
            if k and k in refs.get("text_dur", {}):
                b["dup_clips"] += 1; dup_tot += 1
                if args.confirm_audio_dur and audio_field:
                    edur = _wav_duration(ex.get(audio_field))
                    if edur is not None:
                        if any(abs(edur - wd) <= args.duration_tol for wd in refs["text_dur"][k]):
                            b["dup_confirmed"] += 1
                            # self-check: do the speaker_ids AGREE on a clip we know is a duplicate?
                            # agree => ids are global (speaker check trustworthy); disagree => per-dataset.
                            if spk and spk in refs.get("text_spk", {}).get(k, ()):
                                b["dup_spk_agree"] += 1
                            elif spk:
                                b["dup_spk_disagree"] += 1
                        else:
                            b["dup_unconfirmed"] += 1
            if total % args.log_every == 0:
                logger.info("  …%d clips (%.1f/s) | test=%d val=%d net-new=%d | transcript dups=%d",
                            total, total / max(time.time() - t0, 1e-9), test_tot, val_tot, newtot, dup_tot)
            if args.max_external and total >= args.max_external:
                break
        if args.max_external and total >= args.max_external:
            break
    logger.info("Scan done: %d clips in %.0fs (%d skipped by lang filter) | test=%d val=%d net-new=%d | transcript dups=%d",
                total, time.time() - t0, skipped_lang, test_tot, val_tot, newtot, dup_tot)

    out = {}
    for s, v in per_source.items():
        ext_spk, ovl = v.pop("ext_speakers"), v.pop("overlap_speakers")
        v["n_ext_speakers"], v["n_overlap_speakers"] = len(ext_spk), len(ovl)
        v["overlap_speakers_sample"] = sorted(ovl)[:10]
        out[s] = v
    return {"total": total, "text_field": text_field, "speaker_field": speaker_field,
            "audio_field": audio_field, "source_field": source_field, "per_source": out}


# ---------------------------------------------------------------------------
# optional audio probe — corruption fingerprint
# ---------------------------------------------------------------------------

def _duration_s(audio) -> float:
    import numpy as np
    from collections.abc import Mapping
    if isinstance(audio, Mapping) and "array" in audio:
        sr = int(audio.get("sampling_rate", 0)) or 1
        return len(audio["array"]) / sr
    if hasattr(audio, "get_all_samples"):
        s = audio.get_all_samples()
        arr = np.asarray(s.data)
        sr = int(getattr(s, "sample_rate", 0)) or 1
        return (arr.shape[-1] if arr.ndim > 1 else arr.size) / sr
    return 0.0


def probe_audio(args) -> dict:
    out = {}
    for split, stream in _external_splits(args):
        durs = []
        for i, ex in enumerate(stream):
            if i >= args.audio_sample:
                break
            if "audio" in ex:
                durs.append(_duration_s(ex["audio"]))
        if not durs:
            continue
        durs.sort()
        band = sum(1 for d in durs if CORRUPTION_BAND[0] <= d <= CORRUPTION_BAND[1])
        out[split] = {"sampled": len(durs), "min": round(durs[0], 3),
                      "median": round(durs[len(durs) // 2], 3), "max": round(durs[-1], 3),
                      "in_corruption_band": band, "band_frac": round(band / len(durs), 4)}
        logger.info("audio[%s]: n=%d dur min/med/max=%.2f/%.2f/%.2fs | ~1s band: %d (%.1f%%)",
                    split, len(durs), durs[0], durs[len(durs) // 2], durs[-1],
                    band, 100 * band / len(durs))
    return out


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

def verdict(scan: dict, audio: dict, rerelease_frac: float) -> dict:
    """Decision, strongest signal first — all SPEAKER-INDEPENDENT except where noted:
      1. TEST-transcript overlap         -> breach material (exclude; if pervasive, reject).
      2. duration-CONFIRMED duplicates   -> SAME recordings as WAXAL (same transcript + same audio
                                            length) -> those clips ARE WAXAL data -> exclude them.
      3. ~1s corruption fingerprint      -> same WAXAL audio pipeline -> reject.
      4. transcript matches, 0 confirmed -> same PROMPTS, different recordings -> usable.
      5. low overlap + high net-new      -> independent -> safe.
    Speaker overlap is REPORTED corroboration only (ids may be per-dataset); the dup speaker
    agree/disagree tally reveals whether the id scheme is global."""
    verdicts = {}
    for src, b in scan["per_source"].items():
        n = max(b["n"], 1)
        train_frac, new_frac = b["train_hits"] / n, b["no_match"] / n
        band = max((a.get("band_frac", 0) for a in audio.values()), default=0.0)
        dupC, dupU = b.get("dup_confirmed", 0), b.get("dup_unconfirmed", 0)
        dup_checked = dupC + dupU
        n_ovl, n_ext = b.get("n_overlap_speakers", 0), b.get("n_ext_speakers", 0)
        vspk, agree, disagree = b.get("val_speaker_clips", 0), b.get("dup_spk_agree", 0), b.get("dup_spk_disagree", 0)

        if b["test_hits"] > 0:
            v = (f"BREACH RISK — {b['test_hits']}/{b['n']} clips match WAXAL TEST transcripts. Exclude "
                 "those before any use; if pervasive, reject the source.")
        elif dupC > 0:
            v = (f"CONTAINS WAXAL RECORDINGS — {dupC}/{b['n']} clips are duration-CONFIRMED duplicates "
                 "(same transcript + same audio length) = the same recordings. Those are WAXAL data: "
                 f"exclude them; usable remainder is the net-new {100*new_frac:.1f}%.")
        elif band >= 0.01:
            v = (f"SAME AUDIO PIPELINE — ~1s corruption fingerprint ({100*band:.1f}%): shares WAXAL "
                 "recordings. REJECT.")
        elif b["dup_clips"] > 0 and dup_checked == 0:
            v = (f"UNRESOLVED — {b['dup_clips']} clips match a WAXAL transcript but duration was NOT "
                 "checked (need --confirm-audio-dur + soundfile, or a waveform check). Can't yet tell "
                 "same-audio from same-prompt.")
        elif b["dup_clips"] > 0:
            v = (f"SAME PROMPTS, NOT SAME AUDIO — {b['dup_clips']} transcript matches but 0/{dup_checked} "
                 f"duration-confirmed = different recordings (independent collection reusing prompts), "
                 f"{100*new_frac:.1f}% net-new. SAFE to use (drop any test/val matches).")
        else:
            v = (f"INDEPENDENT — {100*new_frac:.1f}% net-new, train overlap {100*train_frac:.2f}%, no "
                 "confirmed duplicates, no corruption fingerprint. SAFE to add to TRAIN (disclose).")

        if n_ext:
            scheme = ""
            if dupC > 0 and (agree + disagree) > 0:
                scheme = (f" [id scheme looks {'GLOBAL' if agree >= disagree else 'PER-DATASET'}: "
                          f"on confirmed dups {agree} agree / {disagree} disagree]")
            v += f" Speakers: {n_ovl}/{n_ext} overlap WAXAL{scheme}."
        if vspk > 0:
            v += f" ⚠ {vspk} clips have a speaker in OUR VAL set — exclude to keep val speaker-disjoint."
        if b["val_hits"] and vspk == 0:
            v += f" ⚠ {b['val_hits']} val-transcript hits — drop those."

        verdicts[src] = {"n": b["n"], "test_hits": b["test_hits"], "train_hits": b["train_hits"],
                         "val_hits": b["val_hits"], "no_match": b["no_match"], "dup_clips": b["dup_clips"],
                         "dup_confirmed": dupC, "dup_unconfirmed": dupU, "dup_spk_agree": agree,
                         "dup_spk_disagree": disagree, "n_ext_speakers": n_ext, "n_overlap_speakers": n_ovl,
                         "val_speaker_clips": vspk, "overlap_speakers_sample": b.get("overlap_speakers_sample", []),
                         "train_overlap_frac": round(train_frac, 4), "net_new_frac": round(new_frac, 4),
                         "verdict": v, "test_hit_examples": b["test_hit_examples"]}
    return verdicts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    quiet_hf_logging()
    args = parse_args()
    out_dir = args.output_dir or os.path.join(str(workdir()), "logs")
    os.makedirs(out_dir, exist_ok=True)
    login_to_hf(args.hf_token)

    logger.info("=" * 78)
    logger.info("Is %s safe to train on, or WAXAL (%s) in disguise?",
                args.external_id, "+".join(args.waxal_languages))
    logger.info("=" * 78)

    refs = build_waxal_refs(args)
    scan = scan_external(args, refs)
    audio = probe_audio(args) if args.audio_sample > 0 else {}
    verdicts = verdict(scan, audio, args.rerelease_frac)

    logger.info("\n" + "=" * 78 + "\nVERDICT (per external source)\n" + "=" * 78)
    for src, v in verdicts.items():
        logger.info("\n[%s]  n=%d  test=%d  train=%d (%.1f%%)  net-new=%d (%.1f%%)  "
                    "dups=%d (confirmed %d)  speakers %d/%d overlap",
                    src, v["n"], v["test_hits"], v["train_hits"], 100 * v["train_overlap_frac"],
                    v["no_match"], 100 * v["net_new_frac"], v["dup_clips"], v["dup_confirmed"],
                    v["n_overlap_speakers"], v["n_ext_speakers"])
        logger.info("  => %s", v["verdict"])
        for ex in v["test_hit_examples"]:
            logger.info("     test-match example: %r", ex)

    report = {"external_id": args.external_id, "waxal_languages": args.waxal_languages,
              "external_total": scan["total"], "text_field": scan["text_field"],
              "source_field": scan["source_field"], "ref_sizes": {k: len(v) for k, v in refs.items()},
              "audio_probe": audio, "verdicts": verdicts}
    path = os.path.join(out_dir, "verify_external_data.json")
    write_json(path, report)
    logger.info("\nSaved -> %s", path)

    logging.shutdown()
    os._exit(0)   # HF streaming leaves connections that can hang on teardown


if __name__ == "__main__":
    main()
