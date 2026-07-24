"""tok_check.py — the pre-Lever-0 TOKENIZER / TEXT gate (plan §4).

CPU-only, no model weights. Confirms the Qwen3-ASR tokenizer can represent WAXAL's
training targets (cased + punctuation) WITHOUT a byte-fallback / [UNK] explosion that
would rot CER. Because Qwen3-ASR is generative, the "target" is just the string the
model must emit — so the gate is a clean round-trip test: decode(encode(t)) == t.

Findings that shaped this gate (measured 2026-07-22 on Train.csv, post normalize_for_ctc):
  * WAXAL transcripts contain ZERO `ɔ`/`ɛ`/greek-`ε` — those enter only via LRSC/FLEURS
    (Lever 3). So `--external` re-runs the gate over that extra inventory when we get there.
  * The real WAXAL-only risk is the Bantu eng nasals `ŋ` (U+014B, 466×) and `ᵑ` (U+1D51,
    60×) plus 20 French-Lingala diacritics (é ŋ è î í ô à ñ â ú ê ï ç É ó á ì ù ò û ü).

Verdict:
  GO  (keep raw)  — round-trip clean, no [UNK], per-char token inflation modest.
  MAP            — a handful of chars byte-fragment/UNK → fold them into v2's CHAR_MAP.
  NO-GO          — pervasive byte fallback across common chars → tokenizer can't host WAXAL.
"""
from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import common  # copied verbatim from mms-ctc-v2/src

SCRIPT_DIR = Path(__file__).resolve().parent


def find_val_ids() -> Path | None:
    """Frozen val (ids only). Search repo + flat-box layouts for the mms line's val.csv."""
    rels = [Path("experiments") / "mms-ctc" / "manifests" / "val.csv",
            Path("mms-ctc") / "manifests" / "val.csv",
            Path("manifests") / "val.csv"]
    for up in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        for rel in rels:
            if (up / rel).exists():
                return up / rel
    return None


def load_transcripts(train_csv: Path, val_ids_csv: Path | None) -> tuple[list[str], set[str]]:
    """Return (normalized transcript strings, set of val ids). Normalization = the exact
    training-target normalization the model will be asked to emit (normalize_for_ctc)."""
    val_ids: set[str] = set()
    if val_ids_csv and val_ids_csv.exists():
        with open(val_ids_csv, newline="") as f:
            val_ids = {row["id"] for row in csv.DictReader(f)}
    texts: list[str] = []
    with open(train_csv, newline="") as f:
        for row in csv.DictReader(f):
            t = common.normalize_for_ctc(row["transcription"])
            if t:
                texts.append(t)
    return texts, val_ids


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer  # lazy
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        print(f"AutoTokenizer failed ({e}); trying AutoProcessor.tokenizer …", file=sys.stderr)
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=True).tokenizer


def char_inventory(texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        for ch in t:
            c[ch] += 1
    return c


def probe_chars(tok, inv: Counter) -> list[dict]:
    """For each non-ASCII char, encode it (with a stable ASCII context so BPE doesn't
    merge across a word boundary) and report token count + whether it survives decode."""
    unk = tok.unk_token or "<unk>"
    rows = []
    for ch in sorted((c for c in inv if ord(c) > 127), key=lambda c: -inv[c]):
        # context 'a{ch}a' isolates the char's own segmentation cost
        ids = tok.encode(f"a{ch}a", add_special_tokens=False)
        toks = tok.convert_ids_to_tokens(ids)
        dec = tok.decode(ids)
        has_unk = unk in toks or (tok.unk_token_id is not None and tok.unk_token_id in ids)
        # byte fallback shows up as <0xNN> style pieces or the char not surviving decode
        byte_frag = any("0x" in str(t).lower() and str(t).startswith(("<", "▁<", "Ġ<")) for t in toks)
        survives = ch in unicodedata.normalize("NFC", dec)
        rows.append({
            "char": ch, "cp": f"U+{ord(ch):04X}",
            "name": unicodedata.name(ch, "?"), "count": inv[ch],
            "n_tokens": len(ids), "unk": has_unk, "byte_frag": byte_frag,
            "survives": survives, "pieces": toks,
        })
    return rows


def roundtrip(tok, texts: list[str]) -> dict:
    """Full-string decode(encode(t)) == NFC(t) over the DISTINCT transcript strings."""
    distinct = sorted(set(texts))
    fails = []
    tok_total = char_total = 0
    for t in distinct:
        ref = unicodedata.normalize("NFC", t)
        ids = tok.encode(t, add_special_tokens=False)
        got = unicodedata.normalize("NFC", tok.decode(ids))
        tok_total += len(ids)
        char_total += len(ref)
        if got != ref:
            fails.append((t, got))
    return {
        "n_distinct": len(distinct),
        "n_fail": len(fails),
        "fail_rate": len(fails) / max(1, len(distinct)),
        "tokens_per_char": tok_total / max(1, char_total),
        "samples": fails[:15],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # data_dir() finds Train.csv in either layout: flat data/Train.csv OR data/google-waxal-…/Train.csv.
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--train-csv", default=str(common.data_dir() / "Train.csv"))
    ap.add_argument("--val-ids", default=str(find_val_ids() or ""))
    ap.add_argument("--max-char-report", type=int, default=40)
    args = ap.parse_args()

    common.quiet_hf_logging()
    val_ids_path = Path(args.val_ids) if args.val_ids else None
    texts, val_ids = load_transcripts(Path(args.train_csv), val_ids_path)
    inv = char_inventory(texts)
    print(f"[data] {len(texts)} transcripts, {len(inv)} distinct chars "
          f"({sum(1 for c in inv if ord(c) > 127)} non-ASCII), {len(val_ids)} frozen-val ids")

    print(f"[tok ] loading {args.model} …")
    tok = load_tokenizer(args.model)
    print(f"[tok ] vocab_size={tok.vocab_size}, unk={tok.unk_token!r}")

    print("\n=== per-character probe (non-ASCII) ===")
    print(f"{'char':>4} {'cp':>7} {'n_tok':>5} {'unk':>3} {'byte':>4} {'ok':>3}  count  name")
    rows = probe_chars(tok, inv)
    problems = []
    for r in rows[: args.max_char_report]:
        flag = "!" if (r["unk"] or r["byte_frag"] or not r["survives"]) else " "
        print(f"{flag}{r['char']:>3} {r['cp']:>7} {r['n_tokens']:>5} "
              f"{'Y' if r['unk'] else '.':>3} {'Y' if r['byte_frag'] else '.':>4} "
              f"{'Y' if r['survives'] else 'N':>3}  {r['count']:>5}  {r['name']}")
    for r in rows:
        if r["unk"] or r["byte_frag"] or not r["survives"]:
            problems.append(r)

    print("\n=== full-string round-trip (distinct targets) ===")
    rt = roundtrip(tok, texts)
    print(f"distinct={rt['n_distinct']}  fails={rt['n_fail']}  "
          f"fail_rate={rt['fail_rate']:.4%}  tokens/char={rt['tokens_per_char']:.3f}")
    for t, got in rt["samples"]:
        print(f"  FAIL  ref={t[:60]!r}\n        got={got[:60]!r}")

    # ---- verdict ----
    print("\n=== VERDICT ===")
    problem_mass = sum(r["count"] for r in problems)
    total_nonascii = sum(inv[c] for c in inv if ord(c) > 127)
    if rt["fail_rate"] == 0 and not problems:
        print("GO (keep raw): tokenizer round-trips every WAXAL target with no UNK/byte "
              "fallback. Feed cased+punct text as-is; no CHAR_MAP change needed for the gate.")
    elif rt["fail_rate"] < 0.005 and problem_mass / max(1, total_nonascii) < 0.02:
        print(f"MAP: {len(problems)} char(s) fragment/UNK (mass {problem_mass}/"
              f"{total_nonascii} non-ASCII occurrences). Fold them into v2 CHAR_MAP, "
              f"re-run to green. Chars: {[r['char'] for r in problems]}")
    else:
        print(f"NO-GO risk: round-trip fail {rt['fail_rate']:.2%}, {len(problems)} problem "
              f"chars (mass {problem_mass}). Tokenizer struggles to host WAXAL text — "
              f"reconsider before spending GPU on Lever 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
