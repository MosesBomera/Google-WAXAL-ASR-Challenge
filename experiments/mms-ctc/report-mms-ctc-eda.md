# Experiment‑02 · Stage 1 — EDA Report (raw WAXAL: Lingala · Shona · Luganda)

**Data:** full raw download, **38,199 clips / 212.5 h**, 0 missing.
**Method:** download everything unfiltered → analyse → *then* choose thresholds. (Deliberately not importing another paper's cut‑offs on faith — and that paid off, see Finding 3.)

| Lang | clips | hours | dur p50 | p90 | p99 | **max** |
|---|---:|---:|---:|---:|---:|---:|
| lin | 16,244 | 81.2 | 17.6 s | 25.0 s | 33.3 s | **37.3 s** |
| lug | 6,119 | 41.9 | 22.3 s | 37.4 s | 44.1 s | **46.0 s** |
| sna | 15,836 | 89.4 | 19.7 s | 27.0 s | 32.7 s | **35.9 s** |

---

## Finding 1 (headline) — a systematic corruption that we trained on in experiment‑01

**~1,400 clips have audio truncated to exactly 1.01 s while retaining their full transcript.**

Evidence this is a real corpus artifact, not noise:
- The durations are **exactly 1.008–1.01 s**, uniformly — not a natural distribution. The duration histogram shows `0–1 s: 0`, `1–1.5 s: 1044`, `1.5–2 s: 0` for Lingala: a spike, isolated from everything else (next real clip is ~2 s+).
- The transcripts are **full‑length**: median **22 words**, up to **44 words**, giving speech rates of **11–44 words/sec**. Human speech is ~1.1–1.5 wps here. **Physically impossible.**
- It is **language‑specific** — which rules out a bug in our decoder (that would hit all three equally):

| Lang | corrupt clips | % of clips | % of **audio hours** |
|---|---:|---:|---:|
| **lin** | **1,023–1,044** | **6.43%** | **0.35%** |
| sna | 354 | 2.24% | 0.11% |
| **lug** | **0** | **0%** | **0%** |

**Two things follow.**

**(a) Dropping them is nearly free.** They're 6.43% of Lingala's *clips* but only **0.35% of its audio hours** (they're 1 s each). We remove 1,400 poisoned examples and lose essentially no speech.

**(b) They plausibly caused experiment‑01's Lingala wall.** Training on a clip that demands 22–44 words from 1.01 s of audio is *literally training the model to hallucinate text from nothing*. Experiment‑01's only filter was `duration > 30 s`, so **we trained on every one of these.** And the corruption rate rank‑orders our experiment‑01 results exactly:

| Lang | corrupt % | exp‑01 WER |
|---|---:|---:|
| lin | 6.43% | 0.45 (worst) |
| sna | 2.24% | 0.39 |
| lug | 0% | 0.32 (best) |

It also matches the failure mode we *observed* — Whisper's repetition/over‑generation (`camion,camion`). Caveat: n=3 is a correlation, not proof, and WAXAL‑NET independently shows Lingala is intrinsically hard (morpheme‑boundary errors). But the mechanism is concrete, the fix is free, and experiment‑02 will test it.

**Bonus:** the corrupt clips' *transcripts are valid Lingala/Shona sentences* — only the audio is broken. Don't waste them: they're usable **text for the Stage‑3 KenLM**.

## Finding 2 — there is no long‑audio problem at all

**The longest clip in the entire dataset is 46.04 s.** Nothing ≥ 60 s. And the longest clips are *legitimate* (lin max 37.3 s @ 48 words, wps 1.29; lug max 46.0 s @ 65 words, wps 1.41) — no upper‑end corruption.

This kills a whole workstream:
- Whisper's hard 30 s cap forced us to drop ~6.2% of training data and to chunk at inference. **CTC has no cap** → with an upper bound of ~46 s we keep **100%** of clips.
- `lin` (max 37.3 s) and `sna` (max 35.9 s) lose **zero** clips. Only `lug` has anything >40 s (395 clips, 6.5%) — and even those are only ≤46 s.
- **Stage‑4 "forced‑alignment recovery of long clips" is unnecessary.** Delete it from the plan. The concern that motivated part of this pivot simply doesn't exist in the data.

The only remaining consideration is *memory*: 46 s ⇒ ~2,300 wav2vec2 frames, fine on an A100 with length‑grouped batching.

## Finding 3 — the paper's thresholds do *not* transfer cleanly (why EDA-first mattered)

WAXAL‑NET's `words/sec > 4` filter and our `< 1.5 s` filter **catch the same clips** (of Lingala's 1,044 short clips, 1,023 are also wps>4). Either alone suffices; together they're belt‑and‑braces. But note:
- On our languages the wps filter removes **6.4% / 2.2% / 0%** — it is *not* the dramatic universal lever the paper reported (their Lingala 113.5→49.0 was for zero‑shot Whisper‑Tiny, a different regime).
- Threshold choice barely matters: wps>3 → 1,062 clips; wps>6 → 997. The corruption is so extreme (11–44 wps) that anything in 3–6 catches it.

## Finding 4 — low words/sec is NOT a truncated‑reference artifact (keep those clips)

WAXAL‑NET found truncated references in *other* languages. We checked ours: the slowest clips are short descriptions over long audio (`'Babeti ndembo ya maboko.'`, 4 w / 20.7 s). **But `lug` — which has zero corruption — shows the same pattern** (`6 w / 20.1 s`). Since WAXAL is *image‑prompted spontaneous speech*, long pauses and hesitation are expected. So low‑wps looks **natural, not corrupt → keep** (CTC handles silence with blanks). Flag only.
One genuine exception: **1 Lingala clip has an empty transcript** (17.7 s, 0 words) → drop.

## Finding 5 — character inventory needs normalization, not `[UNK]`

119 distinct characters (lin 109, lug 70, sna 84); uppercase present. Punctuation: ``!"&'(),-./:;<=>?`«»×“”⭐️``

**39 characters occur <20 times and are mostly junk** — an emoji (`⭐` + `U+FE0F` variation selector), smart quotes `“ ”`, guillemets `« »`, `×`, and Latin‑Extended singletons (`ĝ þ œ Ķ ķ Ĺ Œ Ŋ ā ĺ`), plus `= < > /` and backtick. Others are **legitimate** French‑influenced Lingala diacritics (`é à ï ê ç ù û ü ò ó á ì ú`, 5–17× each) and rare digits (`7`, `8`).

→ Build a **character normalization map** (smart quotes→`"`, guillemets→`"`, strip emoji/variation selectors, map/strip the Latin‑Extended singletons) rather than letting them into the CTC vocab or routing to `[UNK]`. Keep the accented characters — they're real.

## Finding 6 — duplicates are negligible
12 distinct transcripts repeat, covering 24 clips. No action.

## Finding 7 — speaker metadata: the provided validation split is ~100% leaked, and Lingala is speaker‑poor

The Hub exposes `speaker_id` and `gender` (Train.csv does not). Pulling them changed the picture twice over.

**7a. The provided train/validation split shares essentially all speakers.**

| Lang | val clips | val clips whose speaker is in train | **leakage** | val speakers also in train |
|---|---:|---:|---:|---:|
| lin | 1,844 | 1,843 | **99.95%** | 72 / 73 |
| sna | 1,727 | 1,723 | **99.77%** | 97 / 100 |
| lug | 664 | 659 | **99.25%** | 247 / 252 |

**Experiment‑01's validation was therefore optimistic** — it scored the model on voices it had memorised, while **Phase 2 is unseen speakers**. → We must build our **own speaker‑disjoint split**; the provided one cannot measure what the competition actually rewards.

**7b. Speaker diversity rank‑orders our experiment‑01 results — again.**

| Lang | speakers | clips | clips/speaker (mean) | top speaker share | corrupt % | **exp‑01 WER** |
|---|---:|---:|---:|---:|---:|---:|
| **lug** | **339** | 6,119 | 18 | **1.13%** | 0% | **0.32** (best) |
| sna | 160 | 15,836 | 99 | 8.86% | 2.24% | 0.39 |
| **lin** | **96** | 16,244 | **169** | **10.69%** | 6.43% | **0.45** (worst) |

Lingala has the **most clips but the fewest speakers** — one speaker alone is **10.7%** of it. Luganda has the **fewest clips but the most speakers**, very evenly spread. So Lingala's data is *redundant* (many clips, few voices) while Luganda's is *diverse*. Two independent signals — corruption **and** speaker diversity — produce the same ranking as our results. This is a **speaker‑generalisation** story, which is precisely what Phase 2 tests.

**7c. The corruption is speaker‑scoped, not random — whole sessions failed.**

| Lang | speakers with corrupt clips | worst offenders |
|---|---:|---|
| lin | 15 of 96 | **796/796 (100%)**, 94/94, 75/75, 42/42, 8/8 — all 100% |
| sna | 9 of 160 | 183/183 (100%), 108/108 (100%), 39/40, 13/19 |

**One Lingala speaker accounts for 76% of all Lingala corruption** (796 of 1,046 clips), and their *entire* contribution is truncated. This confirms a systematic per‑session export failure rather than random clip damage. Consequence: cleaning removes ~5 Lingala speakers **entirely** (96 → ~91) — but they were never learnable (1 s of truncated audio each), so we lose nothing real and stop being poisoned.

**Order matters:** clean **first**, then split by speaker — otherwise we'd hold out a speaker whose clips all disappear.

`gender` is now usable too (female 21,253 / male 16,889 / unknown 57) — reasonably balanced, available for stratification.

---

## Recommended cleaning config (evidence‑based)

| Filter | Value | Rationale | Cost |
|---|---|---|---|
| Drop empty transcript | — | unusable | 1 clip |
| **Drop `duration < 1.5 s`** | 1.5 s | isolates the 1.01 s corruption spike exactly; nothing legitimate lives below 2 s | lin 6.43%, sna 2.24%, lug 0% of clips = **0.35%/0.11%/0% of hours** |
| **Drop `words/sec > 4`** | 4.0 | belt‑and‑braces on the same corruption; catches the ~2 non‑short outliers | ~0.2% extra |
| **Upper duration bound** | **≈46 s (keep all)** | longest real clip is 46.04 s and legitimate; CTC has no cap | **0%** |
| Low wps | **flag only, don't drop** | natural spontaneous‑speech pauses (present in clean `lug` too) | 0 |
| Character normalization | map junk → ASCII / strip | 39 rare junk chars incl. emoji | — |
| Corrupt clips' text | **route to the KenLM corpus** | transcripts are valid; only audio is broken | — |

Net: we keep **~99.6% of audio hours** and remove ~1,400 hallucination‑teaching examples.

## What this changes in the plan
1. **Delete Stage 4's forced‑alignment recovery** — no clip exceeds 46 s; nothing to recover.
2. **Upper bound = ~46 s, not 40 s** — keeps 100% of clips (my 40 s default would have needlessly dropped 395 Luganda clips).
3. **The Lingala hypothesis is now testable:** if corruption drove the wall, experiment‑02 should show a disproportionate Lingala gain purely from cleaning — independent of the MMS switch.
4. **Vocab builder needs a normalization map** before emitting `vocab.json`.
5. **Length‑grouped batching** is required (durations span 1.5–46 s).

---

## Analysis status — Stage 1 is COMPLETE

| # | Question | Status |
|---|---|---|
| 1 | Is the 1.01 s truncation corpus‑side or our decode bug? | ✅ **CORPUS** — the Hub itself serves 48,384 samples @ 48 kHz = 1.008 s with a matching full transcript; our decode is faithful; the 1 s contains real (truncated) speech. **Safe to drop.** |
| 2 | Does WaxalNLP expose speaker metadata? | ✅ **YES** — `speaker_id` + `gender`. See Finding 7: ~100% val leakage, Lingala speaker‑poor, corruption speaker‑scoped. |
| 3 | EDA on the TEST split | ❌ **Dropped deliberately.** We don't touch the test set, and the organisers state the **Phase‑2 test set — the one that decides awards — is clean**. So any corruption in the *public* test split is an artifact we can't and needn't act on. *Corollary:* our public‑LB Lingala number may be **pessimistic**; don't tune to it. |
| 4 | Character context inspection | 🟡 Optional, small — informs map‑vs‑strip for the ~39 junk chars. Can be done offline from the metadata + Train.csv. |

**Not needed:** long‑audio analysis (settled — max 46 s), duplicate handling (negligible), truncated‑reference hunting (Finding 4: it's natural spontaneous speech).

## Artifacts to keep (survive instance teardown)
- **`data/google-waxal-asr-challenge20260630-10570-elxebu/waxal_clip_metadata.csv`** (~4 MB, 38,199 rows): `id, language, split, speaker_id, gender, duration_s, n_words, n_chars, wps, is_short, is_too_fast`. **595 distinct speakers.** Every cleaning/splitting decision is answerable from this file offline — no GPU, no re‑download.
- `experiments/mms-ctc/eda_logs/` — `eda_report.json`, `verify_corruption.json`, `verify_metadata.json`, `speaker_summary.json`.

The audio itself is re‑downloadable in minutes, so nothing else needs preserving.
