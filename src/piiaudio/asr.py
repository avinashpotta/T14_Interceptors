"""Speech to text with word-level timings (faster-whisper).

Model size is a hardware decision, not a quality one: on an 8 GB M2 the
practical ceiling is `small` with int8 quantisation. `medium` fits but swaps
under any other load, and we transcribe the corpus many times over the week.

Section 2 claims Whisper's beam search as the Unit 2 'best-first search'
content, so `beam_size` is exposed rather than buried - sweeping it is the
quality-versus-latency experiment that backs that claim.

We hand faster-whisper raw float32 samples rather than a file path. It would
otherwise decode the file through PyAV, whose API moved under us (v19 dropped
a kwarg faster-whisper 1.2.1 still passes). Since we already hold 16 kHz mono
float32 - the exact format it wants - routing around the decoder removes a
dependency from the critical path instead of pinning it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from faster_whisper import WhisperModel

from .audio import read_wav

DEFAULT_MODEL = "small"


@dataclass(frozen=True)
class Word:
    text: str
    start: float          # seconds
    end: float
    probability: float
    char_start: int       # offset into Transcription.text
    char_end: int


@dataclass(frozen=True)
class Transcription:
    text: str
    words: tuple[Word, ...]
    language: str
    model: str
    beam_size: int

    def char_to_time(self, char_start: int, char_end: int) -> tuple[float, float] | None:
        """Map a character span in `self.text` back to an audio time range.

        This is the join that makes the pipeline work: detectors operate on
        text, the redactor operates on samples, and this is the bridge.
        Returns None if the span covers no word, so a detector emitting a span
        over whitespace beeps nothing rather than beeping arbitrary audio.
        """
        hit = [w for w in self.words if w.char_end > char_start and w.char_start < char_end]
        if not hit:
            return None
        return min(w.start for w in hit), max(w.end for w in hit)

    # --- serialisation -----------------------------------------------------
    # Transcription is the expensive step (~4 s per dialogue, and the corpus is
    # transcribed repeatedly this week as detectors change). Everything
    # downstream depends only on this output, so we compute it once per
    # (audio, model, beam) and reload from disk thereafter.

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "model": self.model,
            "beam_size": self.beam_size,
            "words": [
                {"text": w.text, "start": w.start, "end": w.end,
                 "probability": w.probability,
                 "char_start": w.char_start, "char_end": w.char_end}
                for w in self.words
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transcription":
        return cls(
            text=d["text"],
            words=tuple(Word(**w) for w in d["words"]),
            language=d["language"],
            model=d["model"],
            beam_size=d["beam_size"],
        )


@lru_cache(maxsize=2)
def _load(model_size: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_size, device="cpu", compute_type=compute_type)


def transcribe(
    audio: str | np.ndarray,
    model_size: str = DEFAULT_MODEL,
    beam_size: int = 5,
    compute_type: str = "int8",
) -> Transcription:
    """Transcribe a wav path or a float32 16 kHz mono sample array."""
    samples = read_wav(audio) if isinstance(audio, str) else audio.astype(np.float32)

    model = _load(model_size, compute_type)
    segments, info = model.transcribe(
        samples,
        beam_size=beam_size,
        word_timestamps=True,
        language="en",
        vad_filter=False,  # VAD would silently drop audio; we want the raw behaviour
    )

    words: list[Word] = []
    text = ""
    for seg in segments:
        for w in seg.words or ():
            token = w.word.strip()
            if not token:
                continue
            char_start = len(text) + (1 if text else 0)
            text = f"{text} {token}" if text else token
            words.append(
                Word(
                    text=token,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                    char_start=char_start,
                    char_end=len(text),
                )
            )

    return Transcription(
        text=text,
        words=tuple(words),
        language=info.language,
        model=model_size,
        beam_size=beam_size,
    )
