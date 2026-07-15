# Experiment‑01 — Closing Report

**Model:** `openai/whisper-large-v3` + LoRA (multilingual: Lingala, Shona, Luganda)
**Code:** `experiments/whisper-lora/` (versioned by git commits)
**Outcome:** public leaderboard **0.74** — up from experiment‑00's 0.65.
**Verdict:** solid, honest baseline; it exposed exactly where the problem is (Lingala) and the evidence now says the *next* backbone should be **MMS**, not another Whisper iteration.

---

## 1. What we built

Fine‑tuned Whisper‑large‑v3 on all three languages jointly with LoRA (r=32/α=32, encoder+decoder attention+MLP), 3 epochs, effective batch 16, bf16, on a single A100‑40GB (vast.ai). Key design choices that worked:

- **Correct sequence lengths** (`max_target_length=256`) — fixed experiment‑00's truncation bug.
- **Per‑language WER/CER checkpoint selection** (`predict_with_generate`, `metric_for_best_model="combined"`) — not loss/token‑accuracy.
- **Luganda→`sw` proxy token** (Whisper has no `lg` token) + **temperature sampling α=0.5** for the class imbalance.
- **Auto‑detect language at inference** (Phase‑2‑safe: audio only, no ID routing).
- Persistent logs + auto‑zip of best checkpoints.

## 2. Results

| Metric | Submitted (greedy, no chunk) | Best local (greedy + chunk) |
|---|---|---|
| Board | **0.74** | ~0.755 (est.) |
| WER / CER | 0.386 / 0.131 | ~0.38 / ~0.11 |

Per‑language (local test, with 30 s chunking):

| Lang | WER | CER | note |
|---|---|---|---|
| **Lingala** | ~0.45 | ~0.17 | **the wall** — worst on both |
| Shona | ~0.39 | ~0.08 | great CER, WER from word‑segmentation |
| Luganda | ~0.32 | ~0.07 | **best** — the `sw` proxy worked |

Local test eval tracked the leaderboard closely (local micro 0.393/0.123 vs board 0.386/0.131), so our offline number is a faithful proxy.

## 3. What we learned (findings)

1. **Luganda was never the problem.** The `sw`‑proxy token + balanced sampling gave it the best WER/CER. We can stop worrying about it.
2. **Lingala is the wall, and it's orthographic — not mis‑detection.** We verified the predictions: they're fluent Lingala, not French. The error is spelling/word‑boundary (`ba`‑prefix spacing, loanword spelling, compound splitting), which inflates WER more than CER.
3. **The 30 s truncation was real and language‑dependent.** Whisper's hard 30 s cap truncated long clips; because Luganda has the longest transcripts, it was hit hardest. **Inference chunking recovered Luganda CER 0.128 → 0.072** — the single biggest re‑run gain. Decision: **greedy + chunking** is our submission config (beam=5 was neutral‑to‑harmful on lin/sna and ~3× slower).
4. **Mild overfit signature.** Train loss fell to 0.56 while eval WER plateaued at ~0.40 by ~step 5000/5975 — capacity/regularization levers (dropout, label smoothing, augmentation) were all off.
5. **The metric rewards CER, and Bantu languages are CER‑strong.** WER over‑states difficulty for these languages (characters right, word boundaries wrong). Our metric weights CER 50% — so CER wins matter as much as WER.
6. **This is v1's *config* ceiling, not Whisper's — but a label‑noise/orthography floor is real.** Some WAXAL references are inconsistent or truncated; part of the Lingala WER wall is unwinnable.

## 4. Engineering learnings (carried into the harness)

- **Eval‑time dtype mismatch:** loading the model in bf16 crashes `generate()` (float mel vs bf16 conv) because eval runs outside autocast → cast mel features to the model dtype in the collator.
- **`use_cache` for eval:** gradient checkpointing sets `use_cache=False`, which silently made generation ~5× slower; set `generation_config.use_cache=True` (safe, eval runs with `model.training=False`).
- **Log‑to‑disk + zip‑before‑destroy:** cloud instances are ephemeral — `metrics.jsonl`, `trainer_log_history.json`, and `zip_artifacts.py` now capture everything before teardown. (We lost v1's intermediate checkpoints this way once — auto‑zip fixes it.)
- **Per‑language, macro *and* micro metrics** are essential; a good average can hide a failing language.

## 5. The evidence‑based pivot (why experiment‑02 changes backbone)

We deferred a backbone bake‑off in favour of shipping a Whisper baseline fast. Three sources now run that bake‑off for us:

- **Qwen3‑ASR** supports ~52 languages, **none African** — the sibling‑competition winner is irrelevant here. Dropped. ([Qwen3‑ASR](https://github.com/QwenLM/Qwen3-ASR))
- **MMS** cuts Whisper's WER ~58% relative on FLEURS and, zero‑shot on our languages, beats a near‑broken Whisper (Lingala 44.7 vs 93.0; Shona 36.9 vs 112.4; Luganda 32.1 vs no‑support). ([MMS, arXiv 2305.13516](https://arxiv.org/pdf/2305.13516); [African‑ASR benchmark, arXiv 2512.10968](https://arxiv.org/pdf/2512.10968))
- **WAXAL‑NET (arXiv 2606.02375)** — a benchmark on *this exact dataset*: **architecture should follow language family. CTC (MMS) wins or ties on all six Bantu languages** (our three are all Bantu), **leads CER in 17/19 languages**, and is **immune to Whisper's repetition‑loop hallucination**. It also validates a **data‑cleaning pipeline** (drop <1.5 s; drop `words/sec > 4`) that fixed WER‑inflating reference artifacts (Lingala 113.5% → 49.0%).

Tellingly, our *fine‑tuned* Whisper‑large‑v3 Lingala (0.45 WER) barely matches MMS's *zero‑shot* Lingala — MMS fine‑tuned should go well below that.

## 6. What carries forward to experiment‑02

**Decision:** experiment‑02 = fine‑tune **`facebook/mms-1b-all`** (CTC), full data‑cleaning, cased+punctuated vocab, long‑audio handled natively (CTC has no 30 s cap), KenLM decoding. See `../mms-ctc/plan-mms-ctc.md`.

Reusable from experiment‑01:
- Per‑language WER/CER + macro/micro eval, text normalization, log‑to‑disk, zip‑before‑destroy, temperature sampling, the vast.ai path layout, and the data‑duration/drop reporting (already in `train.py`).
- **Whisper‑large‑v3 (v1) is kept as the ensemble partner** — strong CER + autoregressive disambiguation to pair with MMS's Bantu/CER strength.

Open questions to resolve in experiment‑02:
1. **Zindi's normalization** — does the leaderboard lowercase / strip punctuation? Decides how much casing matters (a controlled submission will tell us). We build MMS with a **cased vocab** to be safe either way.
2. **Lingala reference artifacts** — check our three languages for the truncated/`words/sec>4` bad references the paper found.
3. **Ensemble design** — MMS (Bantu/CER, LM‑decoded word boundaries) + Whisper (AR disambiguation) once both are strong.
