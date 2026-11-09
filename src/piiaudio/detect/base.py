"""Common interface every detection level implements.

Section 4.3 compares four levels (regex / regex+namelist / Presidio / NER+rules).
They only produce comparable numbers if they emit the same thing, so all four
return a list of Detection over character offsets of whatever text they were
given. Nothing here knows about audio - the text-to-time bridge lives in
asr.Transcription.char_to_time, which keeps detectors testable on plain
strings and makes Day 4's gold-transcript-versus-ASR-transcript experiment a
matter of passing in a different string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    pii: str          # leaf category from taxonomy
    char_start: int
    char_end: int
    text: str
    source: str       # which rule/model fired - needed for the traceability in Section 2
    score: float = 1.0

    def overlaps(self, other: "Detection") -> bool:
        return self.char_end > other.char_start and self.char_start < other.char_end


class Detector(Protocol):
    name: str

    def detect(self, text: str) -> list[Detection]: ...


def merge(detections: list[Detection]) -> list[Detection]:
    """Union overlapping detections, keeping the longer span.

    Section 4.1 says to lean towards catching more rather than fewer, which is
    the same asymmetry that makes us report F2: an over-wide beep is an
    annoyance, a missed one is a breach.
    """
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: (d.char_start, -d.char_end))
    out = [ordered[0]]
    for d in ordered[1:]:
        last = out[-1]
        if d.char_start < last.char_end:
            if d.char_end > last.char_end:
                out[-1] = Detection(
                    pii=last.pii,
                    char_start=last.char_start,
                    char_end=d.char_end,
                    text=last.text,
                    source=f"{last.source}+{d.source}",
                    score=max(last.score, d.score),
                )
        else:
            out.append(d)
    return out
