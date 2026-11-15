"""Streaming redaction with a delay buffer (§4.1).

The constraint that shapes everything: **audio already played cannot be
redacted.** So playback runs a fixed delay behind capture, and a sample must
be beeped before it leaves the buffer. The delay is the entire budget we have
to recognise speech, find PII, and overwrite it.

Three parameters, and the tension between them is the whole design:

  delay       how far playback lags capture. Larger = more time to act,
              worse conversational latency.
  window      how much audio each ASR pass sees. Whisper needs context -
              a 1 s window transcribes far worse than a 6 s one - but a
              longer window costs more compute per pass.
  asr_every   how often a pass runs. More often = more chances to catch a
              span before it is emitted, at linear compute cost.

DE-DUPLICATION IS FREE IF YOU USE A MASK.
Overlapping windows re-transcribe the same audio, so the same name is detected
several times at slightly different offsets. Rather than reconciling word
sequences across windows - which is fiddly and error-prone - we keep a boolean
mask over absolute sample positions and OR each detection into it. Re-detecting
a span simply re-sets bits that are already set. Idempotent, order-independent,
and it means a span missed on one pass can still be caught on the next, right
up until the moment it is emitted.

WHY `window` SHOULD EXCEED `delay`.
If the ASR window is longer than the delay, each sample is examined by several
passes before playback reaches it. That is what makes a late detection useful
rather than wasted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .asr import transcribe
from .audio import SAMPLE_RATE, beep
from .detect.base import Detection, merge


@dataclass
class StreamConfig:
    delay_s: float = 3.0
    window_s: float = 6.0
    asr_every_s: float = 1.5
    margin_ms: int = 150
    model: str = "small"
    beam_size: int = 1          # realtime: beam 1 is the usual compromise


@dataclass
class StreamStats:
    asr_passes: int = 0
    detections: int = 0
    samples_emitted: int = 0
    samples_beeped: int = 0
    leaked: list = field(default_factory=list)   # (start, end) emitted unredacted


class StreamingRedactor:
    """Feed it chunks, it returns redacted audio delayed by `delay_s`."""

    def __init__(self, detectors, config: StreamConfig | None = None):
        self.cfg = config or StreamConfig()
        self.detectors = detectors
        self.audio = np.zeros(0, dtype=np.float32)     # everything received
        self.mask = np.zeros(0, dtype=bool)            # absolute redaction mask
        self.emitted = 0                               # samples already played
        self.last_asr = 0                              # sample index of last pass
        self.stats = StreamStats()

    # ------------------------------------------------------------------ core

    def _run_asr(self) -> None:
        """Transcribe the most recent window and OR detections into the mask."""
        win = int(self.cfg.window_s * SAMPLE_RATE)
        start = max(0, self.audio.size - win)
        chunk = self.audio[start:]
        if chunk.size < SAMPLE_RATE // 2:              # <0.5 s is not worth a pass
            return

        tr = transcribe(chunk, model_size=self.cfg.model, beam_size=self.cfg.beam_size)
        self.stats.asr_passes += 1

        dets = merge([d for det in self.detectors for d in det.detect(tr.text)])
        margin = self.cfg.margin_ms / 1000.0
        for d in dets:
            t = tr.char_to_time(d.char_start, d.char_end)
            if not t:
                continue
            a = start + int(max(0.0, t[0] - margin) * SAMPLE_RATE)
            b = start + int((t[1] + margin) * SAMPLE_RATE)
            a, b = max(0, a), min(self.mask.size, b)
            if b > a:
                self.mask[a:b] = True
                self.stats.detections += 1

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Accept captured audio; return whatever is now safe to play."""
        self.audio = np.concatenate([self.audio, chunk.astype(np.float32)])
        self.mask = np.concatenate([self.mask, np.zeros(chunk.size, dtype=bool)])

        if self.audio.size - self.last_asr >= self.cfg.asr_every_s * SAMPLE_RATE:
            self._run_asr()
            self.last_asr = self.audio.size

        delay = int(self.cfg.delay_s * SAMPLE_RATE)
        safe_upto = max(0, self.audio.size - delay)
        return self._emit(safe_upto)

    def flush(self) -> np.ndarray:
        """End of stream: one last pass, then release everything."""
        self._run_asr()
        return self._emit(self.audio.size)

    def _emit(self, upto: int) -> np.ndarray:
        if upto <= self.emitted:
            return np.zeros(0, dtype=np.float32)
        a, b = self.emitted, upto
        out = self.audio[a:b].copy()
        m = self.mask[a:b]
        if m.any():
            # beep each contiguous masked run, so tone edges stay clean
            idx = np.flatnonzero(np.diff(np.concatenate([[0], m.view(np.int8), [0]])))
            for s, e in zip(idx[::2], idx[1::2]):
                out[s:e] = beep(e - s)
            self.stats.samples_beeped += int(m.sum())
        self.emitted = b
        self.stats.samples_emitted += out.size
        return out


def run_file(samples: np.ndarray, detectors, config: StreamConfig | None = None,
             chunk_ms: int = 20) -> tuple[np.ndarray, StreamStats]:
    """Drive the redactor with a file as if it were arriving live."""
    r = StreamingRedactor(detectors, config)
    step = int(chunk_ms * SAMPLE_RATE / 1000)
    pieces = []
    for i in range(0, samples.size, step):
        pieces.append(r.push(samples[i:i + step]))
    pieces.append(r.flush())
    return np.concatenate([p for p in pieces if p.size]), r.stats
