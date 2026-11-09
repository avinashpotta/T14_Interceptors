"""Level 1: regex only.

This is the baseline the proposal expects to break on spoken numbers. It is
here to be beaten, but the shape of its failure is itself a reported result
(Section 4.3), so the patterns are written to be a fair attempt rather than a
strawman: real Aadhaar/PAN/phone formats, and the spoken-digit variants too.
"""

from __future__ import annotations

import re

from .base import Detection

_DIGIT_WORD = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple)"

PATTERNS: tuple[tuple[str, str, str], ...] = (
    # (category, rule name, pattern) - order matters, see merge()/ordering below
    ("EMAIL", "email_std", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("EMAIL", "email_spoken", r"\b[\w.]+(?:\s+dot\s+[\w.]+)*\s+at\s+[\w.]+\s+dot\s+(?:com|in|org|net)\b"),
    ("PAN", "pan_std", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    ("AADHAAR", "aadhaar_digits", r"\b[2-9][0-9]{3}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b"),
    ("PHONE", "phone_digits", r"(?:\+?91[\s-]?)?\b[6-9][0-9]{9}\b"),
    # spoken-out runs: 10+ consecutive digit words. Deliberately greedy - we are
    # measuring whether this approach can work at all, not tuning it.
    ("PHONE", "phone_spoken", rf"\b(?:{_DIGIT_WORD}\s+){{9,}}{_DIGIT_WORD}\b"),
)

_COMPILED = tuple((cat, name, re.compile(pat, re.IGNORECASE)) for cat, name, pat in PATTERNS)


class RegexDetector:
    name = "regex"
    emits = ("EMAIL", "PAN", "AADHAAR", "PHONE")

    def detect(self, text: str) -> list[Detection]:
        found: list[Detection] = []
        for category, rule, rx in _COMPILED:
            for m in rx.finditer(text):
                found.append(
                    Detection(
                        pii=category,
                        char_start=m.start(),
                        char_end=m.end(),
                        text=m.group(0),
                        source=f"regex:{rule}",
                    )
                )
        return found
