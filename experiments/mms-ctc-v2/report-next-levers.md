# report — next levers for MMS-1b (reducing WER/CER past 0.79)

**Status:** living backlog. **Baseline to beat:** greedy MMS-1b-all full FT — board **0.790**
(WER 0.316 / CER 0.104), speaker-disjoint **val combined 0.205 (WER 0.325 / CER 0.084)**.
**Exhausted (do not re-run):** KenLM n-gram decoding (no α helps — model too confident),
checkpoint (weight) averaging (single best wins). Recorded in v1's runbook
[`../mms-ctc/src/RUN.md`](../mms-ctc/src/RUN.md) §7.

---

## 0. The two things that frame everything

### The val split is the enabler
The public board is corruption-capped and Lingala-poisoned, so it can't rank experiments.
The **speaker-disjoint val can** — and speaker-disjoint is exactly what Phase-2 scores (unseen
voices). **Rule: every idea below is A/B-tested on val before it costs a board submission.**
This is what turns a brainstorm into an ordered backlog.

### Where the error actually is (read the ratio)
| | val WER | val CER | WER/CER |
|---|---|---|---|
| MMS-1b greedy | 0.325 | 0.084 | **≈3.9×** |

A ~4× WER/CER ratio is high (English ASR ≈2–2.5×). It means the model **usually gets the
characters right but the word wrong** — one char off flips a whole word to a WER error while
barely moving CER. Two causes, both diagnostic:

1. **Bantu is agglutinative** — lin/sna/lug stack prefixes/suffixes into long words (the `ba-`
   spacing seen in EDA). One wrong affix = wrong word, tiny char cost. Partly *inherent*.
2. **Reference orthography is noisy** — loanword spelling, apostrophes, affix spacing vary by
   annotator.

**Consequence:** a chunk of the WER gap is an *orthographic/label floor*, not an acoustic
deficiency — which is precisely why KenLM (an acoustic-uncertainty tool) did nothing. The WER
levers that will actually move are **word coverage (data)**, **robustness (augmentation)**, and
**normalization (align our spelling to the references)** — *not* decoding.

---

## Tier 1 — do first (cheap, low-risk, high-confidence)

### 1. Train longer via warm-start s2
The open 5-vs-8-epoch question. Warm-start the saved weights at a lower LR and watch val.
```bash
python train_ctc.py --model-id mosesbomera/waxal-mms-1b-ctc \
                    --learning-rate 1e-5 --artifact-dir artifacts/mms-1b-all-s2
```
Either you bank a better model, or you *prove* 5 epochs was the right budget. No new code.
**Measure:** val `combined` drops → keep; flatlines → 5 was right.

### 2. Waveform audio augmentation — best bang-for-buck
Today only **SpecAugment** (masking on the feature map) is on. Missing is *waveform*-level
augmentation, which the literature calls the biggest low-resource lever *and* which targets our
exact problem — generalization to unseen speakers/channels = Phase-2:
- **Noise + reverb** — convolve with room impulse responses; add noise at random 0–15 dB SNR.
  Repeatedly the top augmentation. Cantonese CER 26.2%→16.9%; low-resource studies report
  ~34% WER / ~50% CER *relative* gains.
- **Speed perturbation** (±10% resample) — near-free classic wav2vec2 booster.
- **Implement on-the-fly in the data collator** (no disk blowup).

*Concept:* augmentation manufactures the speaker/channel diversity we don't have — only 96
Lingala voices exist, so teaching the model that the same words survive noise/speed/room change
is a direct substitute for more speakers.
**Measure:** val WER *and* CER; expect both down, WER most.

### 3. Normalization audit — a WER lever hiding in the metric
Given the 3.9× ratio, spend ~1 h (no GPU) diffing val predictions vs references on the
*word-error* cases. If many are pure orthographic mismatches (affix spacing, apostrophes,
loanword spelling), a principled normalization applied to **both train targets and output**
recovers WER for free.
**Guardrail:** derive rules from linguistics + the *train* set only; validate on val; never
tune to test.

---

## Tier 2 — higher ceiling, more effort

### 4. MMS per-language adapters — the most MMS-native idea
`mms-1b-all` ships **native pretrained adapters for lin, sna, and lug** (~2.5M weights each +
a language-specific head). A per-language adapter *specializes* and can beat a shared head.
The catch is the one we designed around: adapters need language routing, and Phase-2 gives no
language ID — but 3 acoustically-distinct languages are trivial to route (a tiny language-ID
classifier, or "run all 3 adapters, keep the lowest-CTC-loss transcript").
**Research nuance:** adapters dominate at *very* low resource (<4 h); at our ~70 h/lang full FT
is competitive — so the bet is **full-FT shared model + per-language adapter specialization on
top**, not adapters alone. Medium effort, uncertain, high ceiling, uniquely MMS.

### 5. External data — allowed by rules, but contamination is the trap
Data beats everything. **Rules (organizer Q&A, `misc/relevant-competiton-questions-answers.md`)
confirm external open datasets ARE allowed** in Phase 1 and for the Phase-2 model, if publicly
licensed and **disclosed** — BUT **forbid training on anything containing the WAXAL test split**.
So contamination is a *compliance* gate, not just hygiene. Verify with
[`src/verify_external_data.py`](src/verify_external_data.py) (transcript overlap vs WAXAL
train/val/test + 1.008 s corruption fingerprint), run **per original source**.

**Case study — `KasuleTrevor/Lingala_100hrs` (110.7 h):** an aggregator of three sources; go to
each directly.
- **AfriVoice (~80 h, ~83%) → VERIFY EMPIRICALLY (not yet run).** Earlier we provisionally
  excluded it on *authorship inference* (Digital Umuganda contributed to WAXAL; language list +
  ~101.5 h + image-prompted ≥15 s method all match). **But a shared contributor is NOT proof of
  shared data — they may have collected separate recordings.** And if AfriVoice IS independent,
  it's the **highest-value external data available**: ~80 h of *domain-matched spontaneous Lingala*
  (unlike FLEURS read speech), targeting our weakest language. So decide by evidence:
  `python verify_external_data.py --external-id DigitalUmuganda/AfriVoice --external-config lin`.
  Decision: TEST-transcript overlap → exclude those clips (breach); ~1.008 s corruption fingerprint
  → same WAXAL audio → reject; high train overlap but no test/fingerprint + high net-new → likely
  an independent collection reusing the same image prompts → SAFE (confirm via speaker-id/audio
  duplicates); low overlap + net-new → independent, use it. **High upside — worth checking before
  rejecting.**
- **FLEURS `ln_cd` → VERIFIED SAFE (2026-07-18).** `verify_external_data.py` scanned it vs WAXAL
  lin train/val/test: **0 hits everywhere** → `INDEPENDENT`. LRSC (read Congolese, Mendeley) still
  to verify. Add to *train* only. **Domain gap:** read speech vs WAXAL's spontaneous → expect a
  *modest* Lingala gain, not a windfall.
  - **Ingestion contract:** `prepare_external.py` must emit FLEURS into the trainer's on-disk schema
    `Features({id, audio:Audio(16k), transcription, language, duration:f32, speaker_id, gender})`
    under `prepared/train/fleurs-<lang>/`, `language="lin"`, synthetic non-colliding id/speaker_id,
    text from `raw_transcription` (cased+punct), then append rows to `manifests/train.csv` (val
    untouched). Gate: report FLEURS chars outside the frozen vocab (else silent `[UNK]` → CER rot).
- Parallel: **FLEURS sn/lg + Common Voice Luganda + Luganda Radio Corpus** for the other two langs
  (verify each the same way).

**Iron rule:** external data goes to TRAIN only; val stays 100 % WAXAL speaker-disjoint.
Evidence it helps (Bantu): SunbirdAI kinyarwanda scaling curve, 0→50 h cut WER 33 %→12.5 %.

### 6. Pseudo-labeling / self-training
Decode *unlabeled* audio with the model, keep high-confidence transcripts, add to training —
10–20% relative WER reduction even in low-resource. Pairs with #5 (label untranscribed external
audio). **Do NOT** self-train on the test's ground-truth labels (forbidden — disqualification);
test-*audio* pseudo-labeling is a rules gray area → avoid. Medium-high effort.

### 7. Seed / logit ensemble — the closer
Train 2–3 MMS runs (different seeds), **average their logits** at inference. Reliable small
WER/CER drop. This is *different* from the checkpoint (weight) averaging that failed: that
averaged weights *within* one run; this averages *predictions across independent runs*. Costs
N× compute → do it late.

---

## Tier 3 — exhausted (don't spend here)
- **KenLM / n-gram LM** — dead (proven). **Neural-LM rescoring** hits the same wall (model too
  confident). **Beam search w/o LM** — no gain on confident CTC. The WER is upstream
  (data/orthography), not in the decoder.

## Horizon — experiment-03 (big bet)
- **Swap the backbone:** `w2v-BERT 2.0` (SeamlessM4T encoder, 4.5M hrs) or **XEUS** (4000+
  langs, strong on African) as a new experiment, then ensemble with MMS. Highest effort; parked.

---

## Recommended sequence
`warm-start s2` → `waveform augmentation` → `normalization audit`
(all three cheap and independently val-testable) → decide `adapters` vs `external data` from the
normalization findings + a rules check → `ensemble` as the final squeeze.

## Sources
- MMS adapters (HF blog) — https://huggingface.co/blog/mms_adapters
- Augmenting wav2vec2 for low-resource ASR — https://arxiv.org/html/2501.00425v1
- Semantic-aware SpecAugment (Cantonese) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12298586/
- WAXAL corpus — https://arxiv.org/html/2602.02734v3
- Luganda Radio Corpus — https://aclanthology.org/2022.lrec-1.208.pdf
- Multi-ASR fusion & pseudo-labeling — https://arxiv.org/html/2506.11089v1
- Self-training for low-resource ASR — https://www.emergentmind.com/topics/self-training-approach-for-low-resource-asr
- WAXAL-NET (this dataset's benchmark) — https://arxiv.org/pdf/2606.02375
- Sunbird Kinyarwanda ASR hackathon (Bantu; noise lesson: white noise > realistic when test is clean) — https://github.com/SunbirdAI/kinyarwanda-asr-hackathon
