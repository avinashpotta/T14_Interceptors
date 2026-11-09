"""Turn text-space detections into overwritten audio.

Two decisions here carry the privacy guarantee.

Overwrite, never attenuate. Section 4.1: ducking the volume or mixing a tone
on top both leave the original signal recoverable by anyone who cares. We
replace the samples.

Margin is asymmetric and deliberate. Whisper's word timings are good to a few
hundred milliseconds, and an error either side is not equally costly: starting
the beep late leaks the first syllable of a name, ending it early leaks the
last digit. We widen both edges, and the margin is a reported parameter
because Section 5 wants the trade-off curve (more margin = better coverage of
sensitive audio, more ordinary speech destroyed) rather than one tuned number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .asr import Transcription
from .audio import SAMPLE_RATE, beep
from .detect.base import Detection

DEFAULT_MARGIN_MS = 150


@dataclass(frozen=True)
class RedactedSpan:
    pii: str
    start: float
    end: float
    source: str
    text: str


def redact(
    samples: np.ndarray,
    detections: list[Detection],
    transcription: Transcription,
    margin_ms: int = DEFAULT_MARGIN_MS,
) -> tuple[np.ndarray, list[RedactedSpan]]:
    """Overwrite detected regions with a tone. Returns (audio, what was beeped)."""
    out = samples.copy()
    margin = margin_ms / 1000.0
    applied: list[RedactedSpan] = []

    for d in detections:
        window = transcription.char_to_time(d.char_start, d.char_end)
        if window is None:
            continue  # span covered no word; beeping arbitrary audio would be worse
        start, end = window
        start = max(0.0, start - margin)
        end = min(len(out) / SAMPLE_RATE, end + margin)

        i0, i1 = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        if i1 <= i0:
            continue
        out[i0:i1] = beep(i1 - i0)
        applied.append(RedactedSpan(d.pii, start, end, d.source, d.text))

    return out, applied


def redacted_fraction(samples: np.ndarray, spans: list[RedactedSpan]) -> float:
    """Share of the audio destroyed. The headline cost of over-beeping."""
    if samples.size == 0:
        return 0.0
    covered = sum(max(0.0, s.end - s.start) for s in spans)
    return covered / (samples.size / SAMPLE_RATE)
