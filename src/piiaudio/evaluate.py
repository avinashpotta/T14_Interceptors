"""Scoring for the Section 4.3 comparison.

Three questions that are genuinely different, and conflating them is how a
redaction system gets reported as working when it is not:

  1. Did the detector FIND it?      - span overlap in character space
  2. Did the beep LAND?             - fraction of the gold PII audio destroyed
  3. What did we destroy needlessly? - fraction of non-PII audio destroyed

A detector can find a name and still leak it if the timing is off, so (1)
without (2) is not a privacy claim.

TYPE-AGNOSTIC BY DEFAULT. For redaction, beeping is beeping: a phone number
caught by the Aadhaar rule is still destroyed, and the person is still
protected. Mislabelling costs nothing operationally. We therefore score
"was it redacted" ignoring the predicted category, and report the
type-confusion separately for the detector's own sake.

F2, NOT F1. Section 5: missing an identifier is a privacy breach, beeping an
extra word is an annoyance. F2 weights recall four times as heavily. It does
NOT make precision free - a collapse to 0.25 precision still loses to a
conservative detector, which is why `over_redaction` is reported alongside.

BOOTSTRAP OVER DIALOGUES, NOT SPANS. Spans inside one dialogue share a
speaker, a template and a transcription, so they are not independent.
Resampling spans would give confidence intervals several times too narrow.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

STRICT_COVER = 0.5   # a gold span counts as detected if this much is covered


@dataclass(frozen=True)
class Span:
    """A gold or predicted region, in seconds."""
    pii: str
    start: float
    end: float

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


def overlap(a: Span, b: Span) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _union(spans: list[Span]) -> list[tuple[float, float]]:
    """Merge to disjoint intervals - predictions may overlap each other."""
    if not spans:
        return []
    out: list[list[float]] = []
    for s in sorted(spans, key=lambda s: s.start):
        if out and s.start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], s.end)
        else:
            out.append([s.start, s.end])
    return [(a, b) for a, b in out]


def _covered(span: Span, merged: list[tuple[float, float]]) -> float:
    """Seconds of `span` falling inside any predicted interval."""
    return sum(max(0.0, min(span.end, b) - max(span.start, a)) for a, b in merged)


@dataclass
class Tally:
    """Accumulates counts across dialogues; convert with .metrics().

    RECALL IS SPAN-LEVEL, PRECISION IS DURATION-LEVEL, deliberately.

    Recall asks "was each sensitive item protected?" - a person is either
    protected or not, so items are the right unit.

    Precision asks "how much of what we destroyed needed destroying?" -
    seconds are the right unit, because one enormous beep is far worse than
    one small one. Span-counted precision gets this catastrophically wrong: a
    detector that beeps the entire recording overlaps every gold span, records
    zero false positives, and scores a perfect 1.0 while producing audio no
    one can use. Duration-based precision scores that same detector at the
    fraction of the call that was genuinely sensitive.

    `fp` is still counted (spans touching no gold at all) but as a diagnostic
    for the detector, not as an input to the headline score.
    """
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gold_seconds: float = 0.0
    gold_covered: float = 0.0
    destroyed_seconds: float = 0.0
    nonpii_seconds: float = 0.0
    nonpii_destroyed: float = 0.0
    per_type: dict = field(default_factory=dict)

    def metrics(self) -> dict:
        p = (self.gold_covered / self.destroyed_seconds) if self.destroyed_seconds else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f2 = (5 * p * r / (4 * p + r)) if (p + r) else 0.0
        return {
            "precision": p,
            "recall": r,
            "f2": f2,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            # share of sensitive audio actually destroyed
            "coverage": (self.gold_covered / self.gold_seconds) if self.gold_seconds else 0.0,
            # share of ordinary speech destroyed - the cost of over-beeping
            "over_redaction": (self.nonpii_destroyed / self.nonpii_seconds)
                              if self.nonpii_seconds else 0.0,
        }


def score_dialogue(gold: list[Span], pred: list[Span], duration: float,
                   tally: Tally, strict_cover: float = STRICT_COVER) -> None:
    """Accumulate one dialogue into `tally`."""
    merged = _union(pred)

    matched_pred: set[int] = set()
    for g in gold:
        cov = _covered(g, merged)
        tally.gold_seconds += g.dur
        tally.gold_covered += cov
        hit = g.dur > 0 and (cov / g.dur) >= strict_cover
        if hit:
            tally.tp += 1
        else:
            tally.fn += 1
        t = tally.per_type.setdefault(g.pii, Tally())
        t.tp += int(hit); t.fn += int(not hit)
        t.gold_seconds += g.dur; t.gold_covered += cov

        for i, p in enumerate(pred):
            if overlap(g, p) > 0:
                matched_pred.add(i)

    # a predicted span touching no gold span at all is a false positive
    tally.fp += sum(1 for i in range(len(pred)) if i not in matched_pred)

    gold_merged = _union(gold)
    gold_sec = sum(b - a for a, b in gold_merged)
    pred_sec = sum(b - a for a, b in merged)
    both = sum(_covered(Span("", a, b), gold_merged) for a, b in merged)
    tally.destroyed_seconds += pred_sec
    tally.nonpii_seconds += max(0.0, duration - gold_sec)
    tally.nonpii_destroyed += max(0.0, pred_sec - both)


def bootstrap(per_dialogue: list[tuple[list[Span], list[Span], float]],
              n: int = 2000, seed: int = 14,
              strict_cover: float = STRICT_COVER) -> dict:
    """95% CI for each metric, resampling DIALOGUES with replacement.

    The unit of resampling is the dialogue because spans within one share a
    speaker, template and transcription. Resampling spans would understate
    the interval badly.
    """
    rng = random.Random(seed)
    keys = ("precision", "recall", "f2", "coverage", "over_redaction")
    draws = {k: [] for k in keys}
    N = len(per_dialogue)
    if N == 0:
        return {k: (0.0, 0.0) for k in keys}
    for _ in range(n):
        t = Tally()
        for _ in range(N):
            g, p, d = per_dialogue[rng.randrange(N)]
            score_dialogue(g, p, d, t, strict_cover)
        m = t.metrics()
        for k in keys:
            draws[k].append(m[k])
    out = {}
    for k in keys:
        v = sorted(draws[k])
        out[k] = (v[int(0.025 * n)], v[int(0.975 * n)])
    return out
