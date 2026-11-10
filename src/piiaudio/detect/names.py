"""Name detection: gazetteer lookup, plus the rule layer that makes it usable.

This module provides both Level 2 and the rule half of Level 4, because they
differ only in which filters are switched on:

  Level 2  gazetteer lookup alone - the honest non-AI baseline
  Level 4  + context cues, + capitalisation, + adjacent-token merging

The context cues are Section 2's predicate logic made concrete: a token is
redacted if it is in a name list and not a common English word, OR if it
follows a name-introducing phrase regardless of whether we have ever seen it.
The second clause is what lets us catch a name no list contains, which is the
whole point - a real deployment cannot enumerate its customers in advance.

Every Detection carries the rule that fired it in `source`, so each beep has a
traceable reason (Section 2, and useful when a result looks wrong).
"""

from __future__ import annotations

import re

from .base import Detection
from .. import gazetteer as gz

# Phrases after which the following tokens are a name, list or no list.
# Written against ASR output, so they do not depend on punctuation.
CUES = [
    r"my name is", r"the name is", r"this is", r"i am", r"i'm",
    r"speaking (?:to|with)", r"name on (?:it|the account|the policy|file) is",
    r"under the name(?: of)?", r"account is under", r"it's", r"its",
    r"call me", r"you can call me", r"am i speaking to", r"holder's name is",
    r"name please", r"your full name", r"beneficiary name",
]
_CUE_RE = re.compile(r"\b(?:" + "|".join(CUES) + r")\b[\s,:-]*", re.IGNORECASE)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
MAX_CUE_TOKENS = 2          # how many tokens after a cue to treat as a name


class NameListDetector:
    """Gazetteer lookup with optional rule layer."""

    def __init__(self, source: str = "both", use_context: bool = False,
                 require_capital: bool = False, stoplist_n: int = gz.DEFAULT_STOPLIST_N,
                 name: str | None = None):
        self.source = source
        self.use_context = use_context
        self.require_capital = require_capital
        self.names = gz.load(source, stoplist_n)
        self.stop = gz.stoplist(stoplist_n)
        self.name = name or f"namelist[{source}]" + ("+rules" if use_context else "")
        self.emits = ("GIVEN_NAME", "SURNAME")

    # ---------------------------------------------------------------- rules

    def _gazetteer_hits(self, text: str) -> list[tuple[int, int, str]]:
        out = []
        for m in _TOKEN_RE.finditer(text):
            tok = m.group(0)
            low = tok.lower()
            if low in self.stop or len(low) <= 2:
                continue
            if low not in self.names:
                continue
            if self.require_capital and not tok[0].isupper():
                continue
            out.append((m.start(), m.end(), "gazetteer"))
        return out

    def _context_hits(self, text: str) -> list[tuple[int, int, str]]:
        """Tokens right after a name-introducing phrase, list or no list."""
        out = []
        for cue in _CUE_RE.finditer(text):
            pos = cue.end()
            taken = 0
            for m in _TOKEN_RE.finditer(text, pos):
                if m.start() > pos + 1:        # a non-word intervened; stop
                    break
                tok = m.group(0)
                low = tok.lower()
                if low in self.stop:
                    break
                # a cue is only trusted when the token looks like a name:
                # capitalised, or already known to a list
                if not (tok[0].isupper() or low in self.names):
                    break
                out.append((m.start(), m.end(), "context"))
                pos = m.end()
                taken += 1
                if taken >= MAX_CUE_TOKENS:
                    break
        return out

    # -------------------------------------------------------------- detect

    def detect(self, text: str) -> list[Detection]:
        hits = self._gazetteer_hits(text)
        if self.use_context:
            hits += self._context_hits(text)
        if not hits:
            return []

        # merge adjacent tokens: "Joy Chopra" is one name, not two
        hits.sort()
        merged: list[list] = []
        for s, e, why in hits:
            if merged and s - merged[-1][1] <= 1:
                merged[-1][1] = max(merged[-1][1], e)
                if why not in merged[-1][2]:
                    merged[-1][2] += "+" + why
            else:
                merged.append([s, e, why])

        out = []
        for s, e, why in merged:
            out.append(Detection(
                pii="GIVEN_NAME",      # we do not distinguish given vs family
                char_start=s, char_end=e,
                text=text[s:e],
                source=f"name:{why}",
            ))
        return out
