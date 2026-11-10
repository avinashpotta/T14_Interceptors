"""Degrade audio in controlled ways, to answer §4.2's "report how much that hurts".

Two degradations, both chosen because they are what actually goes wrong on a
support call, and both implemented in numpy so no dependency is added.

NOISE IS PINK, NOT WHITE. White noise has flat power per hertz, so most of its
energy sits above the speech band and an ASR shrugs it off - it would make the
test look easy. Pink noise falls at 1/f, which is roughly how room hum,
traffic, fans and background chatter are distributed, so it actually competes
with speech. Level is set by SNR against the speech itself, not by a fixed
amplitude, so a quiet dialogue and a loud one are degraded comparably.

ECHO IS EARLY REFLECTIONS, NOT A TAIL. A hands-free phone or a hard-walled
room produces a handful of discrete reflections in the first ~200 ms. Those
smear word boundaries, which is the part that matters here: our whole pipeline
depends on word-level timings, so a degradation that blurs onsets attacks the
mechanism rather than just the transcription.
"""

from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise via spectral shaping of white noise."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    shape = np.ones_like(f)
    shape[1:] = 1.0 / np.sqrt(f[1:])
    out = np.fft.irfft(spectrum * shape, n=n)
    peak = np.max(np.abs(out))
    return (out / peak).astype(np.float32) if peak else out.astype(np.float32)


def add_noise(samples: np.ndarray, snr_db: float, seed: int = 14) -> np.ndarray:
    """Mix pink noise in at a given signal-to-noise ratio."""
    rng = np.random.default_rng(seed)
    speech_rms = float(np.sqrt(np.mean(samples ** 2)))
    if speech_rms == 0:
        return samples.copy()
    noise = _pink(samples.size, rng)
    noise_rms = float(np.sqrt(np.mean(noise ** 2))) or 1.0
    target = speech_rms / (10 ** (snr_db / 20.0))
    return np.clip(samples + noise * (target / noise_rms), -1.0, 1.0).astype(np.float32)


# (delay in ms, amplitude) - a small hard-walled room / hands-free handset
REFLECTIONS = ((17, 0.50), (43, 0.32), (71, 0.22), (113, 0.14), (167, 0.08))


def add_echo(samples: np.ndarray, wet: float = 1.0) -> np.ndarray:
    """Convolve with sparse early reflections."""
    longest = int(max(d for d, _ in REFLECTIONS) * SAMPLE_RATE / 1000) + 1
    ir = np.zeros(longest, dtype=np.float32)
    ir[0] = 1.0
    for ms, amp in REFLECTIONS:
        ir[int(ms * SAMPLE_RATE / 1000)] = amp * wet
    out = np.convolve(samples, ir)[: samples.size]
    peak = float(np.max(np.abs(out)))
    if peak > 1.0:                      # preserve level, do not just clip
        out = out / peak * float(np.max(np.abs(samples)))
    return out.astype(np.float32)


def band_limit(samples: np.ndarray, low: float = 300.0, high: float = 3400.0) -> np.ndarray:
    """Restrict to the telephone band via FFT masking, with soft edges."""
    spec = np.fft.rfft(samples)
    f = np.fft.rfftfreq(samples.size, 1.0 / SAMPLE_RATE)
    mask = np.ones_like(f)
    mask[f < low] = 0.0
    mask[f > high] = 0.0
    # 100 Hz raised-cosine skirts; a brick wall rings badly in the time domain
    for edge, rising in ((low, True), (high, False)):
        band = (f >= edge - 100) & (f <= edge + 100) if rising else \
               (f >= edge - 100) & (f <= edge + 100)
        if band.any():
            x = (f[band] - (edge - 100)) / 200.0
            ramp = 0.5 * (1 - np.cos(np.pi * x))
            mask[band] = ramp if rising else 1.0 - ramp
    return np.fft.irfft(spec * mask, n=samples.size).astype(np.float32)


def mu_law(samples: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """G.711 companding round-trip - the quantisation a phone line applies."""
    x = np.clip(samples, -1.0, 1.0)
    comp = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((comp + 1.0) * 127.5) / 127.5 - 1.0          # 8-bit
    return (np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu).astype(np.float32)


def telephone(samples: np.ndarray, snr_db: float | None = None) -> np.ndarray:
    """What a real support call actually does to speech."""
    out = band_limit(samples)
    out = mu_law(out)
    if snr_db is not None:
        out = add_noise(out, snr_db, seed=7)
    peak = float(np.max(np.abs(out)))
    if peak:
        out = out / peak * float(np.max(np.abs(samples)))
    return out.astype(np.float32)


CONDITIONS = {
    "clean":    lambda s: s,
    "noise20":  lambda s: add_noise(s, 20.0),   # mild room hum
    "noise10":  lambda s: add_noise(s, 10.0),   # clearly audible background
    "echo":     lambda s: add_echo(s),
    # The degradation that actually matters for this project's domain. Pink
    # noise and reverb left Whisper essentially unaffected (it was trained on
    # 680k hours of noisy web audio); a telephone line removes the frequency
    # bands outright and quantises what is left to 8 bits.
    "phone":    lambda s: telephone(s),
    "phone_noisy": lambda s: telephone(s, snr_db=12.0),
}
