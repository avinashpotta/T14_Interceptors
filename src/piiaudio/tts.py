"""Render a dialogue script to audio WITH exact ground-truth timings.

The idea that makes the rest of the project cheap
-------------------------------------------------
A dialogue is written as a list of segments that alternate between ordinary
speech and PII:

    [{"text": "yes my name is"}, {"text": "Rose Mathew", "pii": "GIVEN_NAME"}]

We render each segment as its own utterance, trim its padding, and concatenate
while accumulating a sample offset. The offset at which a PII segment lands IS
its ground-truth start time - exact, by construction.

The alternative was forced alignment (WhisperX/wav2vec2) against a known
script. We rejected it for two reasons. It is a whole extra model and install,
and more importantly the proposal also wanted alignment as a *pipeline* step
to sharpen Whisper's timings - so the same component would have been both the
thing under test and the thing defining truth. Rendering by segment removes
that circularity entirely.

Cost of the choice: `say` applies utterance-final prosody to each segment, so
the speech has slightly unnatural seams at segment boundaries. A pause at a
word boundary is a cue the ASR would not get in real speech, so our numbers
are optimistic in a way we have to declare. The human-recorded subset exists
to measure how much.

Each dialogue yields spans in BOTH coordinate systems:
  * char offsets into the reference transcript - for scoring detectors on text
  * seconds into the audio          - for scoring whether the beep landed
Day 4's error-attribution experiment needs the first; Section 5's coverage
metric needs the second.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import taxonomy
from .audio import SAMPLE_RATE, read_wav, write_wav, trim_silence, silence

# macOS en_IN voices. Three speakers is not much variety, but it is three
# genuinely Indian-accented voices for zero install cost, which is the point.
VOICES: tuple[str, ...] = ("Rishi", "Aman", "Tara")

_EMPTY = np.zeros(0, dtype=np.float32)

_NO_SPACE_BEFORE = set(".,!?;:')")

INTRA_TURN_GAP_MS = 10   # between segments of one sentence
INTER_TURN_GAP_MS = 280  # between speakers


@dataclass(frozen=True)
class GoldSpan:
    pii: str          # leaf category from taxonomy
    text: str         # exactly what was spoken, e.g. "nine eight seven ..."
    start: float      # seconds into the audio
    end: float
    char_start: int   # offset into the reference transcript
    char_end: int
    # The underlying value, independent of how it was read aloud ("9876543210").
    # Day 1 finding F1 means we must be able to ask "did Whisper recover the
    # right digits?" separately from "how were they spoken?" - without this,
    # a normalised transcript looks like a mismatch against the spoken text.
    canonical: str | None = None
    style: str | None = None   # which speaking style produced `text`


def _say(text: str, voice: str, out_path: Path) -> np.ndarray:
    """Render one utterance and return trimmed samples."""
    subprocess.run(
        ["say", "-v", voice, "-o", str(out_path), "--data-format=LEI16@16000", text],
        check=True,
        capture_output=True,
    )
    return trim_silence(read_wav(out_path))


def _join(prefix: str, piece: str) -> str:
    if not prefix:
        return piece
    if piece[:1] in _NO_SPACE_BEFORE:
        return prefix + piece
    return prefix + " " + piece


def render_dialogue(script: dict) -> tuple[np.ndarray, dict]:
    """Render a dialogue script. Returns (samples, gold_record)."""
    chunks: list[np.ndarray] = []
    spans: list[GoldSpan] = []
    transcript = ""
    n_samples = 0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "seg.wav"
        for t_i, turn in enumerate(script["turns"]):
            voice = turn["voice"]
            if voice not in VOICES:
                raise ValueError(f"unknown voice {voice!r}; expected one of {VOICES}")

            if t_i > 0:
                gap = silence(INTER_TURN_GAP_MS)
                chunks.append(gap)
                n_samples += gap.size

            for s_i, seg in enumerate(turn["segments"]):
                text = seg["text"].strip()
                if not text:
                    continue

                # `say` SPEAKS bare punctuation rather than pausing - measured
                # at 0.96 s for "." and 1.10 s for "?". Rendering these would
                # inject a second of spurious audio immediately after most PII
                # spans. Keep them in the reference transcript, render nothing.
                renderable = any(c.isalnum() for c in text)

                if s_i > 0 and renderable:
                    gap = silence(INTRA_TURN_GAP_MS)
                    chunks.append(gap)
                    n_samples += gap.size

                samples = _say(text, voice, tmp) if renderable else _EMPTY

                # _join only ever appends `text` (after an optional separator),
                # so the start offset falls out of the resulting length.
                transcript = _join(transcript, text)
                char_end = len(transcript)
                char_start = char_end - len(text)

                if "pii" in seg:
                    label = taxonomy.normalise(seg["pii"])
                    spans.append(
                        GoldSpan(
                            pii=label,
                            text=text,
                            start=n_samples / SAMPLE_RATE,
                            end=(n_samples + samples.size) / SAMPLE_RATE,
                            char_start=char_start,
                            char_end=char_end,
                            canonical=seg.get("canonical"),
                            style=seg.get("style"),
                        )
                    )

                chunks.append(samples)
                n_samples += samples.size

    audio_out = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    gold = {
        "id": script["id"],
        "template": script.get("template", "unknown"),
        "sample_rate": SAMPLE_RATE,
        "duration": float(audio_out.size / SAMPLE_RATE),
        "transcript": transcript,
        "spans": [asdict(s) for s in spans],
    }
    return audio_out, gold


def render_to_disk(script: dict, audio_dir: Path) -> dict:
    audio_out, gold = render_dialogue(script)
    audio_dir = Path(audio_dir)
    write_wav(audio_dir / f"{script['id']}.wav", audio_out)
    (audio_dir / f"{script['id']}.gold.json").write_text(json.dumps(gold, indent=2))
    return gold
