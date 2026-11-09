#!/usr/bin/env python3
"""Finding F1: is Whisper's spoken-digit normalisation consistent?

Day 1 showed Whisper rendering "nine eight seven..." as `9876543210`, which
made a regex succeed where Section 1 predicted failure. The question that
actually matters is not whether it normalises, but whether it does so
*predictably*: unpredictable normalisation is worse than reliable failure,
because you cannot write a rule against it.

For every numeric identifier we know the canonical digits and the style it
was read in, so we can classify what came out the other side.
"""
from __future__ import annotations

import json
import re
import sys
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio.asr import Transcription

NUMERIC = ("PHONE", "AADHAAR")


def classify(canonical: str, spoken: str, asr_text: str) -> str:
    """What happened to this identifier in the transcript?"""
    flat = re.sub(r"[^0-9]", "", asr_text)
    if canonical in flat:
        return "digits"                      # fully normalised, regex-friendly
    # did the spoken word-form survive intact?
    norm = lambda s: re.sub(r"[^a-z ]", " ", s.lower()).split()
    sw, aw = norm(spoken), norm(asr_text)
    if sw and " ".join(sw) in " ".join(aw):
        return "words"                       # untouched, needs spoken-form rules
    # partial digits present?
    if len(canonical) >= 6 and canonical[:6] in flat:
        return "partial"
    return "mangled"                         # neither form recoverable


def main() -> None:
    man = {m["id"]: m for m in json.loads((ROOT/"data"/"manifest.json").read_text())}
    by_style = collections.defaultdict(collections.Counter)
    by_type = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)

    for did, meta in sorted(man.items()):
        gold_p = ROOT/"data"/"audio"/f"{did}.gold.json"
        asr_p = ROOT/"data"/"asr"/f"{did}.small.beam5.json"
        if not (gold_p.exists() and asr_p.exists()):
            continue
        gold = json.loads(gold_p.read_text())
        asr = Transcription.from_dict(json.loads(asr_p.read_text()))
        for sp in gold["spans"]:
            if sp["pii"] not in NUMERIC or not sp.get("canonical"):
                continue
            outcome = classify(sp["canonical"], sp["text"], asr.text)
            by_style[sp["style"]][outcome] += 1
            by_type[sp["pii"]][outcome] += 1
            if len(examples[outcome]) < 3:
                examples[outcome].append((did, sp["style"], sp["canonical"], sp["text"][:48]))

    cols = ["digits", "words", "partial", "mangled"]
    def table(title, data):
        print(f"\n{title}")
        print(f"  {'':<16}" + "".join(f"{c:>9}" for c in cols) + f"{'n':>7}")
        for k in sorted(data):
            row, n = data[k], sum(data[k].values())
            print(f"  {k:<16}" + "".join(f"{100*row[c]/n:>8.0f}%" for c in cols) + f"{n:>7}")

    table("BY SPEAKING STYLE", by_style)
    table("BY IDENTIFIER TYPE", by_type)

    allc = collections.Counter()
    for c in by_style.values(): allc.update(c)
    n = sum(allc.values())
    print(f"\nOVERALL (n={n}): " + "  ".join(f"{c}={100*allc[c]/n:.0f}%" for c in cols))

    print("\nexamples:")
    for outcome in cols:
        for did, style, canon, spoken in examples[outcome][:2]:
            print(f"  [{outcome:<8}] {did:<9} {style:<15} {canon}  <- {spoken!r}")


if __name__ == "__main__":
    main()
