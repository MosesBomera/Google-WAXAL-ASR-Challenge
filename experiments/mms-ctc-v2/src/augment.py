"""mms-ctc-v2 — light, on-the-fly waveform augmentation (TRAIN ONLY).

Deliberately minimal, following Sunbird's Kinyarwanda ASR lesson: they dropped realistic
background noise for PLAIN WHITE NOISE because their test set was fairly clean, and
over-realistic augmentation hurt. WAXAL/Phase-2 is "natural-environment" but still not a
noisy-far-field task, so we START LIGHT and let val decide whether to escalate:

  * speed perturbation (resample so the clip is ×0.9–1.1 speed — changes tempo AND pitch,
    the sox/Kaldi style; teaches speed/pitch invariance from the 96 Lingala voices we have),
  * additive WHITE Gaussian noise at a random SNR (mild variation, matched to a clean-ish test).

Each transform fires independently with its own probability. Everything is numpy-only and
fast enough to run in the dataloader collator. SpecAugment (feature-map masking) stays on in
the model — this complements it at the waveform level.

Realistic noise (MUSAN) and reverb (RIR) are intentionally NOT here; add them only if the
val number says the light version helped and there's headroom.
"""

from __future__ import annotations

import numpy as np


class WaveformAugmenter:
    def __init__(self, sample_rate: int,
                 speed_prob: float = 0.5, speed_min: float = 0.9, speed_max: float = 1.1,
                 noise_prob: float = 0.5, snr_min_db: float = 10.0, snr_max_db: float = 20.0):
        self.sr = sample_rate
        self.speed_prob, self.speed_min, self.speed_max = speed_prob, speed_min, speed_max
        self.noise_prob, self.snr_min_db, self.snr_max_db = noise_prob, snr_min_db, snr_max_db

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        # Fresh entropy per call so dataloader workers (which each hold a pickled copy of
        # this object) don't produce correlated augmentations.
        rng = np.random.default_rng()
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size < 2:
            return wav
        if self.speed_prob > 0 and rng.random() < self.speed_prob:
            wav = self._speed(wav, float(rng.uniform(self.speed_min, self.speed_max)))
        if self.noise_prob > 0 and rng.random() < self.noise_prob:
            wav = self._add_white_noise(wav, float(rng.uniform(self.snr_min_db, self.snr_max_db)), rng)
        return wav

    @staticmethod
    def _speed(wav: np.ndarray, factor: float) -> np.ndarray:
        """Resample to length/factor via linear interpolation (numpy-only). factor>1 = faster."""
        n = int(round(len(wav) / factor))
        if n < 2:
            return wav
        idx = np.linspace(0.0, len(wav) - 1.0, n)
        return np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)

    @staticmethod
    def _add_white_noise(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
        sig_power = float(np.mean(wav ** 2)) + 1e-12
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
        return (wav + noise).astype(np.float32)
