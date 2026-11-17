"""Level 1: regex only.

Section 4.3 is explicit that comparing against a weak regex would be unfair
*in our favour*, so these patterns are written against what the ASR actually
emits rather than against clean typed text. Measured on the corpus, that
distinction is worth 49 points of recall on numeric identifiers, so a naive
baseline would have flattered every later level by the same amount.

What the ASR actually does to identifiers
-----------------------------------------
Whisper never leaves spoken digits as words (0 of 96 in our corpus). It
normalises them to numerals - but it preserves the speaker's *prosodic
grouping as punctuation*:

    spoken  "nine six zero, five nine five, seven four five, four"
    ASR     960 -595 -745 -4
    spoken  "ninety seven, forty seven, seventy six, sixty nine, fourteen"
    ASR     97, 47, 76, 69, 14

The digits are all there and correct. A pattern demanding ten *consecutive*
digits simply cannot see them: 32% recall. Allowing the separators Whisper
inserts where the speaker paused: 81%, with no increase in false positives.

Emails get a half-normalisation that matches neither the typed form nor the
spoken one - "dot" becomes a literal "." glued to the next token while "at"
stays a word:

    spoken  "neha dot goli at gmail dot com"
    ASR     neha .goli at gmail .com

So every separator below accepts the typed form, the spoken word, or the
half-normalised hybrid.

Note the patterns match in the ORIGINAL transcript, so character offsets stay
valid and no text rewriting or offset remapping is needed.
"""

from __future__ import annotations

import re

from .base import Detection

# Separators Whisper inserts where a speaker paused mid-identifier.
_SEP = r"[\s,\-]{0,3}"
# "dot" in any of its three surface forms; likewise "at".
_DOT = r"(?:\s*\.\s*|\s+dot\s+)"
_AT = r"(?:\s*@\s*|\s+at\s+)"
_WORD = r"[A-Za-z0-9_+\-]+"
# Compound Indian suffixes (co.in, ac.in, gov.in ...) must come first so the
# alternation never stops at ".co" and leaves ".in" unredacted.
_TLD = rf"(?:(?:co|ac|gov|nic|res|edu|org|net){_DOT}in|com|in|org|net|co|edu|io|ai|dev|info)"

_DIGIT_WORD = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple)"

PATTERNS: tuple[tuple[str, str, str], ...] = (
    # Order matters only for readability; merge() resolves overlaps.
    ("EMAIL", "email_any",
     rf"\b{_WORD}(?:{_DOT}{_WORD})*{_AT}(?:{_WORD}{_DOT})+{_TLD}\b"),

    ("PAN", "pan_std", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    # PAN read aloud letter by letter survives as spaced characters.
    ("PAN", "pan_spaced",
     rf"\b(?:[A-Z]{_SEP}){{5}}(?:[0-9]{_SEP}){{4}}[A-Z]\b"),

    # 12 digits, first never 0 or 1, separators tolerated between any pair.
    ("AADHAAR", "aadhaar_tolerant", rf"\b[2-9](?:{_SEP}[0-9]){{11}}\b"),
    # 10 digits starting 6-9, same tolerance, optional +91.
    ("PHONE", "phone_tolerant", rf"(?:\+?91{_SEP})?\b[6-9](?:{_SEP}[0-9]){{9}}\b"),

    # Kept for the rare case where the ASR does leave words. Measured at 0% on
    # our corpus, but it costs nothing and its absence would be a silent
    # assumption that Whisper always normalises.
    ("PHONE", "phone_spoken",
     rf"\b(?:{_DIGIT_WORD}\s+){{9,}}{_DIGIT_WORD}\b"),
)

_COMPILED = tuple((cat, name, re.compile(pat, re.IGNORECASE)) for cat, name, pat in PATTERNS)


class RegexDetector:
    """Level 1 of the Section 4.3 comparison: patterns only, no name knowledge."""

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
