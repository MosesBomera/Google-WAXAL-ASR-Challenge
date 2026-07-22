# report — external data (provenance, verification & disclosure)

**Purpose.** The competition rules require that external datasets be *"publicly accessible,
appropriately licensed, and clearly disclosed in your final solution documentation."* This is that
disclosure — every external source considered for mms-ctc-v2, its licence and provenance, how we
verified it is not WAXAL test material, and the decision. It doubles as the single source of truth
for the **external-data lever** (Lever A of [`plan-mms-ctc-v2.md`](plan-mms-ctc-v2.md)).

## The compliance line we do not cross
Organizer Q&A ([`../../misc/relevant-competiton-questions-answers.md`](../../misc/relevant-competiton-questions-answers.md)):
external open datasets are allowed in Phase 1 and for the Phase-2 model, **but you must not train on
anything containing the WAXAL test split** (Phase-1 test audio/transcripts, or a model/dataset that
already saw them). So every candidate is checked for WAXAL-test contamination before use.

**Iron rules (applied to every source below):**
1. External data goes to **TRAIN only**; the validation set stays **100 % WAXAL speaker-disjoint** —
   so val remains an honest, uncontaminated yardstick and Phase-2 proxy.
2. A candidate is used only after `verify_external_data.py` shows **no WAXAL test overlap** and no
   evidence of shared WAXAL recordings.

## How we verify — [`src/verify_external_data.py`](src/verify_external_data.py)
For each source, scanned against WAXAL train / val / **test** transcripts, per language:
- **Transcript overlap** (loose-normalised exact match). *Caveat:* a match is a shared **sentence**,
  not necessarily shared **audio** — two independent collections using the same image prompts yield
  the same sentences from different speakers. So overlap flags **risk**, it doesn't prove identity.
- **~1.008 s corruption fingerprint** — WAXAL's signature truncation. Its presence means the source
  came off the **same audio pipeline** → shares WAXAL recordings. This (and speaker-id / audio
  duplicates) is what actually proves shared audio.
- **Net-new %** — transcripts matching *nothing* in WAXAL: the measure of how much genuinely new
  data a source adds.

Decision order: **any TEST overlap** → exclude those clips (breach); **corruption fingerprint** →
reject (same recordings); **high train overlap, no test/fingerprint, high net-new** → ambiguous,
likely independent prompt-reuse → confirm & use; **low overlap** → independent → use.

---

## Sources

### ✅ FLEURS — `google/fleurs`, configs `ln_cd` / `sn_zw` / `lg_ug`  — USED
| | |
|---|---|
| Licence | CC-BY-4.0 |
| Content | ~10–12 h/language read-aloud speech (FLoRes Wikipedia sentences), all 3 target langs |
| Provenance | Google FLEURS (IEEE SLT 2022) — an established multilingual benchmark, collected independently of WAXAL (read speech vs WAXAL's spontaneous image-prompted speech) |
| Verification | `verify_external_data.py` on `ln_cd` (2026-07-18): **0 test / 0 val / 0 train hits → INDEPENDENT** |
| Decision | **Use** all three splits (train+dev+test — all external to WAXAL) for all 3 languages, TRAIN only. Ingested by `prepare_external.py`. |
| Caveat | Read-speech domain gap vs WAXAL's spontaneous speech → expect a *modest* gain, largest for Lingala. |

### ❌ AfriVoice — `DigitalUmuganda/Afrivoice` (Lingala + Shona)  — REJECTED (verified 2026-07-19)
| | |
|---|---|
| Licence | CC-BY-4.0 (not the blocker) |
| Access | NOT `datasets`-loadable — per-language `manifest_*.json` (line JSON) via `HfFileSystem`; checked with `src/check_afrivoice.py`. |
| Verification | Content-based (transcript + duration), speaker-id-independent. Results: **Lingala** 95,508 recs — **1,742 WAXAL-TEST matches (breach)**, 15,191 duration-CONFIRMED duplicates (= same recordings), only 2.2% net-new → **AfriVoice Lingala IS WAXAL Lingala.** **Shona** 97,240 recs — **1,678 TEST matches (breach)**, 14,950 train-transcript matches, 82.9% net-new. |
| Speaker-id scheme | **GLOBAL** (0 disagreements on confirmed dups) — same person = same UID across datasets. |
| Decision | **REJECT both.** (1) Both contain WAXAL Phase-1 **test** clips → training on them breaches the rules outright. (2) Lingala is a straight WAXAL re-release. (3) Shona, though 83% new-looking, demonstrably shares recordings with WAXAL (1,678 test + 14,950 train transcript matches) and comes from the same collection pool; WAXAL's hidden **Phase-2 test uses unseen speakers = exactly AfriVoice's novel speakers**, so its "new" clips can't be proven Phase-2-free. Too risky for a breach=DQ competition. |
| Logs | `logs/afrivoice-lin/check_afrivoice.json`, `logs/afrivoice-sna/check_afrivoice.json` |

### ✅ LRSC (Lingala Read Speech Corpus) — Mendeley Data  — USED (verified 2026-07-19)
| | |
|---|---|
| Licence | CC-BY-4.0 |
| Content | 2,940 clips (2,557 train + 383 valid), ~5.0 h read Lingala speech from 32 Congolese adult participants |
| Provenance | "Speech Recognition Datasets for Low-resource Congolese Languages" (Mendeley, `data.mendeley.com/datasets/28x8tc9n9k/1`) — independent read-speech effort, collected via in-person elicitation, not WAXAL's image-prompted spontaneous speech |
| Access | NOT `datasets`-loadable — ships as fairseq-style manifests (`transcript.txt` + `audio/*.wav` per split) on disk, not a `datasets` saved-to-disk dir, so `verify_external_data.py --from-disk` didn't apply; checked instead with the dedicated `src/check_lrsc.py`. |
| Verification | Content-based (transcript + duration), speaker-id-independent — same method as AfriVoice. Result: all 2,940 recs — **0 WAXAL-TEST matches, 0 train-transcript matches, 0 val-transcript matches, 0 duration-confirmed duplicates, 0/31 speaker overlap, 100% net-new → INDEPENDENT.** |
| Decision | **Use** both LRSC splits (its own train/valid isn't WAXAL-comparable) → **TRAIN only**. Ingested by `src/prepare_lrsc.py`: 2,923 clips (~5.0 h) survive the same duration/wps hygiene filter (`common.clean_verdict`) WAXAL's own prep applies (dropped 15 too-short + 2 too-fast, ~0.6%). Adds **31 real distinct speakers** (vs. FLEURS' one synthetic id/language) — pure diversity gain for Lingala, our most speaker-poor language. |
| Vocab | Transcripts use the real Lingala open-vowel letters `ɔ`/`ɛ` (1,969×/719× combined, confirmed absent from WAXAL's own Lingala transcripts) plus 2 stray singleton typos (`å`,`ǎ`). `ɔ`/`ɛ` are folded in as **native vocab characters** by re-running `build_vocab.py` after ingestion — not `[UNK]`'d, not char-mapped away. One genuine encoding quirk fixed: a Greek-epsilon look-alike (`ε`) some LRSC transcribers used interchangeably with the correct IPA `ɛ` is now canonicalized via `common.CHAR_MAP` (zero-risk — neither codepoint appears anywhere in WAXAL's `Train.csv`). |
| Logs | `logs/lrsc-lin/check_lrsc.json`, `manifests/lrsc_report.json` |

### 🔭 Candidates for the other languages (future)
- **Common Voice Luganda**, **Luganda Radio Corpus** (LREC 2022) — for `lug` (our least-data language).
  Verify each the same way before use.

---

## Contamination-compliance statement (as of 2026-07-19)
- **Used so far:** FLEURS `ln_cd`/`sn_zw`/`lg_ug` + LRSC `lin` — both verified independent of WAXAL, both CC-BY-4.0, both disclosed here.
- **No source has been trained on without a verification pass** (`verify_external_data.py` for FLEURS, `check_lrsc.py` for LRSC) showing zero WAXAL-test overlap.
- Validation remains **100 % WAXAL speaker-disjoint** (no external data — LRSC's own valid split folds into our train, same rule as FLEURS), preserving an uncontaminated yardstick.
- AfriVoice remains **rejected** (breach material + re-release, see above) and stays out of any submitted model.

*This document is updated as each source is verified. Verification logs: `logs/verify_external_data.json`.*
