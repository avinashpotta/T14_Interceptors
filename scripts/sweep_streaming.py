#!/usr/bin/env python3
"""Delay-budget sweep for the streaming redactor (§4.1, §5 speed).

This is the latency/accuracy trade-off the proposal promised, made concrete.
The hypothesis under test is structural rather than empirical: to redact a
span the ASR must see ALL of it, which cannot happen until capture passes the
span's end. At that moment playback sits `delay` behind. So a span survives
only if its duration is under the delay budget - independent of how fast the
models are.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio.audio import SAMPLE_RATE, read_wav
from piiaudio.detect.names import NameListDetector
from piiaudio.detect.ner import NerDetector
from piiaudio.detect.regex_rules import RegexDetector
from piiaudio.streaming import StreamConfig, run_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--delays", default="2,4,6,8")
    args = ap.parse_args()

    man = [m for m in json.loads((ROOT/"data"/"manifest.json").read_text())
           if m["split"] == "test"][:args.n]
    D = [RegexDetector(),
         NameListDetector("both", use_context=True, require_capital=True),
         NerDetector()]

    print(f"{'delay':>6}{'window':>8}{'redacted':>10}{'leaked':>8}{'leak rate':>11}"
          f"{'beeped':>9}{'xRT':>7}")
    rows = []
    for delay in [float(x) for x in args.delays.split(",")]:
        cfg = StreamConfig(delay_s=delay, window_s=delay + 3.0, asr_every_s=1.5)
        red = leak = 0
        beeped = emitted = 0
        wall = audio_s = 0.0
        by_dur = []
        for m in man:
            wav = ROOT/"data"/"audio"/f"{m['id']}.wav"
            gp = ROOT/"data"/"audio"/f"{m['id']}.gold.json"
            if not (wav.exists() and gp.exists()):
                continue
            s = read_wav(wav); gold = json.loads(gp.read_text())
            t0 = time.time()
            out, st = run_file(s, D, cfg)
            wall += time.time() - t0; audio_s += s.size / SAMPLE_RATE
            beeped += st.samples_beeped; emitted += max(1, st.samples_emitted)
            for sp in gold["spans"]:
                a, b = int(sp["start"]*SAMPLE_RATE), min(int(sp["end"]*SAMPLE_RATE), out.size)
                if b <= a + 10:
                    continue
                corr = float(np.corrcoef(s[a:b], out[a:b])[0, 1])
                ok = abs(corr) < 0.3
                red += ok; leak += (not ok)
                by_dur.append((sp["end"] - sp["start"], ok))
        n = red + leak
        print(f"{delay:>6.1f}{cfg.window_s:>8.1f}{red:>10}{leak:>8}{leak/max(1,n):>10.1%}"
              f"{beeped/emitted:>8.1%}{wall/audio_s:>7.2f}")
        rows.append((delay, by_dur))

    print("\nLEAK RATE BY SPAN DURATION  (does a span longer than the delay survive?)")
    buckets = [(0.0,1.0),(1.0,2.0),(2.0,3.0),(3.0,4.0),(4.0,99.0)]
    print(f"  {'span duration':<16}" + "".join(f"{f'delay {d:.0f}s':>11}" for d,_ in rows))
    for lo, hi in buckets:
        cells = []
        for _, by in rows:
            sel = [ok for dur, ok in by if lo <= dur < hi]
            cells.append(f"{1-sum(sel)/len(sel):>10.0%}" if sel else "         -")
        label = f"{lo:.0f}-{hi:.0f}s" if hi < 99 else f"{lo:.0f}s+"
        print(f"  {label:<16}" + " ".join(cells))
    print("\n  cells are LEAK rate: lower is better")


if __name__ == "__main__":
    main()
