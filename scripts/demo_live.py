#!/usr/bin/env python3
"""Record from the microphone, then redact it. For live demos.

    python scripts/demo_live.py --seconds 30
    python scripts/demo_live.py --seconds 30 --play

Records via ffmpeg/avfoundation straight to 16 kHz mono - the format the
pipeline wants - so there is no conversion step to go wrong on stage.

RUN THIS ONCE BEFORE THE DEMO. The first capture triggers a macOS
microphone-permission dialog, which you do not want appearing live.
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


def record(path: Path, seconds: int, device: str) -> None:
    print(f"\n  RECORDING {seconds}s — speak now", flush=True)
    for i in (3, 2, 1):
        print(f"    {i}...", flush=True); time.sleep(0.6)
    print("    GO\n", flush=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "avfoundation",
         "-i", f":{device}", "-t", str(seconds),
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-y", str(path)],
        check=True,
    )
    print("  done recording\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=str(ROOT / "data" / "live" / "capture.wav"))
    ap.add_argument("--level", type=int, default=4, choices=[1, 4])
    ap.add_argument("--play", action="store_true")
    args = ap.parse_args()

    dst = Path(args.out); dst.parent.mkdir(parents=True, exist_ok=True)
    record(dst, args.seconds, args.device)

    # ffmpeg exits 0 having written a bare WAV header when macOS denies
    # microphone access - no error, no audio, a 78-byte file. Same class of
    # failure as `say` writing silence: a zero exit code means the program did
    # not crash, not that it did the job. Check the artefact.
    samples = read_wav(dst)
    if samples.size == 0:
        sys.exit(
            "\n  CAPTURE FAILED — the recording contains no audio.\n"
            "  ffmpeg reported success but wrote an empty file, which on macOS\n"
            "  means microphone access is denied to this terminal.\n\n"
            "  Fix: System Settings > Privacy & Security > Microphone,\n"
            "       enable your terminal app, then restart the terminal.\n"
            "  Verify with:  ffmpeg -f avfoundation -i :0 -t 2 /tmp/mic.wav\n"
        )
    peak = float(abs(samples).max())
    print(f"  captured {samples.size/SAMPLE_RATE:.1f}s, peak {peak:.3f}")
    if peak < 0.02:
        print("  WARNING: that is near-silence. Check the input device and "
              "speak closer to the mic.\n")
    else:
        print()

    t0 = time.time()
    tr = transcribe(str(dst))
    print(f"  transcript ({time.time()-t0:.1f}s):\n    {tr.text[:400]}\n")

    rx = RegexDetector()
    dets = [rx] if args.level == 1 else [
        rx, NameListDetector("both", use_context=True, require_capital=True),
        NerDetector()]
    hits = merge([d for det in dets for d in det.detect(tr.text)])
    out, applied = redact(samples, hits, tr)
    red = dst.with_name(f"{dst.stem}_redacted.wav")
    write_wav(red, out)

    print(f"  Level {args.level}: {len(applied)} region(s) redacted, "
          f"{100*redacted_fraction(samples, applied):.0f}% of audio destroyed")
    for s in applied:
        print(f"      {s.start:6.2f}-{s.end:<6.2f} {s.pii:<11} {s.source:<24} "
              f"{s.text[:30]!r}")
    if not applied:
        print("      (nothing detected)")
    print(f"\n  wrote {red}")
    if args.play:
        for label, p in (("ORIGINAL", dst), ("REDACTED", red)):
            print(f"\n  playing {label} ...")
            subprocess.run(["afplay", str(p)])


if __name__ == "__main__":
    main()
