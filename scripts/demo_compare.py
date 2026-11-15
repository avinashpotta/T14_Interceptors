#!/usr/bin/env python3
"""Side-by-side demo: the same audio at Level 1 and Level 4.

    python scripts/demo_compare.py data/human/P2_t01_00.wav

Produces <name>_L1.wav and <name>_L4.wav and prints what each caught and
missed. The point is audible: at Level 1 the customer's name plays in the
clear; at Level 4 it does not.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio.asr import transcribe
from piiaudio.audio import SAMPLE_RATE, read_wav, write_wav
from piiaudio.detect.base import merge
from piiaudio.detect.names import NameListDetector
from piiaudio.detect.ner import NerDetector
from piiaudio.detect.regex_rules import RegexDetector
from piiaudio.redact import redact, redacted_fraction


def stacks():
    rx = RegexDetector()
    return {
        1: ("Level 1 — regex only", [rx]),
        4: ("Level 4 — regex + name lists + rules + NER",
            [rx, NameListDetector("both", use_context=True, require_capital=True),
             NerDetector()]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--play", action="store_true", help="play each result (macOS)")
    args = ap.parse_args()

    src = Path(args.input)
    samples = read_wav(src)
    print(f"\n  input: {src.name}   {samples.size/SAMPLE_RATE:.1f}s\n")

    t0 = time.time()
    tr = transcribe(str(src))
    print(f"  transcript ({time.time()-t0:.1f}s):")
    print(f"    {tr.text[:300]}\n")

    outputs = {}
    for level, (label, dets) in stacks().items():
        hits = merge([d for det in dets for d in det.detect(tr.text)])
        out, applied = redact(samples, hits, tr)
        dst = src.with_name(f"{src.stem}_L{level}.wav")
        write_wav(dst, out)
        outputs[level] = (dst, applied)

        print(f"  {label}")
        print(f"    {len(applied)} region(s) redacted, "
              f"{100*redacted_fraction(samples, applied):.0f}% of audio destroyed")
        for s in applied:
            print(f"      {s.start:6.2f}-{s.end:<6.2f} {s.pii:<11} "
                  f"{s.source:<24} {s.text[:30]!r}")
        print()

    caught4 = {s.pii for _, applied in [outputs[4]] for s in applied}
    caught1 = {s.pii for _, applied in [outputs[1]] for s in applied}
    gained = caught4 - caught1
    if gained:
        print(f"  Level 4 additionally redacts: {', '.join(sorted(gained))}")
        print(f"  -> at Level 1 those play in the clear.\n")

    for level in (1, 4):
        print(f"  wrote {outputs[level][0]}")
    if args.play:
        for label, path in (("ORIGINAL", src),
                            ("LEVEL 1", outputs[1][0]), ("LEVEL 4", outputs[4][0])):
            print(f"\n  playing {label} ...")
            subprocess.run(["afplay", str(path)])


if __name__ == "__main__":
    main()
