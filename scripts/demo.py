#!/usr/bin/env python3
"""End-to-end: audio in -> transcript -> detect -> beeped audio out.

    python scripts/demo.py data/audio/d001.wav out.wav
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from piiaudio.asr import transcribe
from piiaudio.audio import read_wav, write_wav
from piiaudio.detect.base import merge
from piiaudio.detect.regex_rules import RegexDetector
from piiaudio.redact import redact, redacted_fraction


def main(in_path: str, out_path: str) -> None:
    samples = read_wav(in_path)
    print(f"[1/4] loaded {samples.size / 16000:.2f}s")

    tr = transcribe(in_path)
    print(f"[2/4] transcribed: {len(tr.words)} words")

    hits = merge(RegexDetector().detect(tr.text))
    print(f"[3/4] detected {len(hits)} span(s)")

    out, applied = redact(samples, hits, tr)
    write_wav(out_path, out)
    pct = 100 * redacted_fraction(samples, applied)
    print(f"[4/4] wrote {out_path}  ({len(applied)} beeped, {pct:.1f}% of audio destroyed)")
    for s in applied:
        print(f"      {s.pii:<9} {s.start:6.2f}-{s.end:6.2f}  via {s.source}  {s.text!r}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out.wav")
