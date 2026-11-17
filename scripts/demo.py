#!/usr/bin/env python3
"""End-to-end demo: audio in -> redacted audio out.

    python scripts/demo.py input.wav output.wav
    python scripts/demo.py input.wav output.wav --level 1    # regex only
    python scripts/demo.py input.wav output.wav --mask-log   # keep PII out of the console

Runs the full Level 4 stack by default (regex + name gazetteers + context
rules + neural NER). Prints what was redacted and why, so every beep has a
traceable reason.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from piiaudio.asr import transcribe
from piiaudio.audio import SAMPLE_RATE, read_wav, write_wav
from piiaudio.detect.base import merge
from piiaudio.detect.names import NameListDetector
from piiaudio.detect.regex_rules import RegexDetector
from piiaudio.redact import redact, redacted_fraction

LEVELS = {
    1: lambda: [RegexDetector()],
    2: lambda: [RegexDetector(), NameListDetector("both")],
    4: None,  # built lazily, NER import is the slow one
}


def build(level: int):
    if level != 4:
        return LEVELS[level]()
    from piiaudio.detect.ner import NerDetector
    return [RegexDetector(),
            NameListDetector("both", use_context=True, require_capital=True),
            NerDetector()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", default="redacted.wav")
    ap.add_argument("--level", type=int, default=4, choices=[1, 2, 4])
    ap.add_argument("--margin-ms", type=int, default=150)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--mask-log", action="store_true",
                    help="do not print the detected PII text to the console")
    args = ap.parse_args()

    t0 = time.time()
    samples = read_wav(args.input)
    dur = samples.size / SAMPLE_RATE
    print(f"[1/4] loaded {args.input}  ({dur:.1f}s)")

    tr = transcribe(args.input, beam_size=args.beam)
    print(f"[2/4] transcribed: {len(tr.words)} words  ({time.time()-t0:.1f}s)")

    detectors = build(args.level)
    hits = merge([d for det in detectors for d in det.detect(tr.text)])
    print(f"[3/4] level {args.level}: {len(hits)} detection(s)")

    out, applied = redact(samples, hits, tr, margin_ms=args.margin_ms)
    write_wav(args.output, out)
    pct = 100 * redacted_fraction(samples, applied)

    print(f"[4/4] wrote {args.output}  —  {len(applied)} region(s) overwritten, "
          f"{pct:.1f}% of audio destroyed\n")
    if applied:
        print(f"  {'CATEGORY':<12}{'START':>7}{'END':>8}   WHY            TEXT")
        for s in applied:
            print(f"  {s.pii:<12}{s.start:>7.2f}{s.end:>8.2f}   {s.source:<14} {'<hidden>' if args.mask_log else repr(s.text[:34])}")
    else:
        print("  (no PII detected)")
    print(f"\n  total {time.time()-t0:.1f}s for {dur:.1f}s of audio "
          f"({(time.time()-t0)/dur:.2f}x realtime)")


if __name__ == "__main__":
    main()
