#!/usr/bin/env python3
"""The §5 edge cases: what happens on input the corpus does not contain.

Demo-track grading is on how well the tool actually runs, including on messy
input, so these are tested rather than asserted. Each case reports what the
system DID, not merely that it survived.
"""
from __future__ import annotations
import json, sys, time, resource
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio.asr import transcribe
from piiaudio.audio import SAMPLE_RATE, read_wav
from piiaudio.detect.base import merge
from piiaudio.detect.names import NameListDetector
from piiaudio.detect.ner import NerDetector
from piiaudio.detect.regex_rules import RegexDetector
from piiaudio.redact import redact, redacted_fraction

D = [RegexDetector(),
     NameListDetector("both", use_context=True, require_capital=True),
     NerDetector()]


def run(samples: np.ndarray, label: str) -> dict:
    t0 = time.time()
    try:
        tr = transcribe(samples)
        hits = merge([d for det in D for d in det.detect(tr.text)])
        out, applied = redact(samples, hits, tr)
        frac = redacted_fraction(samples, applied)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        return {"label": label, "ok": True, "dur": samples.size/SAMPLE_RATE,
                "wall": time.time()-t0, "words": len(tr.words),
                "dets": len(hits), "beeped": frac, "rss_mb": rss,
                "text": tr.text[:90]}
    except Exception as e:                                   # noqa: BLE001
        return {"label": label, "ok": False, "err": f"{type(e).__name__}: {e}"}


def main() -> None:
    rng = np.random.default_rng(14)
    cases = []

    # 1. pure digital silence
    cases.append((np.zeros(10 * SAMPLE_RATE, np.float32), "pure silence 10s"))
    # 2. near-silence: realistic room tone, no speech
    cases.append(((rng.standard_normal(10 * SAMPLE_RATE) * 0.002).astype(np.float32),
                  "room tone, no speech 10s"))
    # 3. a dialogue containing no PII at all
    nopii = [m for m in json.loads((ROOT/"data"/"manifest.json").read_text())
             if m["scenario"] == "no_pii"]
    if nopii:
        p = ROOT/"data"/"audio"/f"{nopii[0]['id']}.wav"
        if p.exists():
            cases.append((read_wav(p), f"dialogue with NO PII ({nopii[0]['id']})"))
    # 4. crosstalk: two dialogues talking over each other
    man = json.loads((ROOT/"data"/"manifest.json").read_text())
    a = read_wav(ROOT/"data"/"audio"/f"{man[0]['id']}.wav")
    b = read_wav(ROOT/"data"/"audio"/f"{man[1]['id']}.wav")
    n = min(a.size, b.size)
    cases.append((np.clip(a[:n] + 0.7*b[:n], -1, 1).astype(np.float32),
                  "crosstalk: two speakers at once"))
    # 5. a long file
    chunks = []
    for m in man[:40]:
        p = ROOT/"data"/"audio"/f"{m['id']}.wav"
        if p.exists():
            chunks.append(read_wav(p))
    if chunks:
        long = np.concatenate(chunks)
        cases.append((long, f"long file ({long.size/SAMPLE_RATE/60:.1f} min)"))
    # 6. very short input
    cases.append((np.zeros(1600, np.float32), "0.1s of audio"))
    # 7. loud clipping noise
    cases.append((np.clip(rng.standard_normal(5*SAMPLE_RATE)*3, -1, 1).astype(np.float32),
                  "loud white noise 5s"))

    print(f"{'CASE':<34}{'dur':>8}{'wall':>7}{'words':>7}{'dets':>6}{'beeped':>9}{'RSS':>8}")
    results = []
    for samples, label in cases:
        r = run(samples, label)
        results.append(r)
        if r["ok"]:
            print(f"{label:<34}{r['dur']:>7.1f}s{r['wall']:>6.1f}s{r['words']:>7}"
                  f"{r['dets']:>6}{r['beeped']:>8.1%}{r['rss_mb']:>7.0f}M")
        else:
            print(f"{label:<34}  CRASHED: {r['err']}")

    print("\nWHAT THE ASR PRODUCED ON NON-SPEECH INPUT:")
    for r in results:
        if r["ok"] and ("silence" in r["label"] or "tone" in r["label"]
                        or "noise" in r["label"] or "0.1s" in r["label"]):
            print(f"  {r['label']:<32} {r['words']:>3} words, {r['dets']} detection(s)")
            if r["text"].strip():
                print(f"      hallucinated: {r['text']!r}")

    bad = [r for r in results if not r["ok"]]
    print(f"\n{len(results)-len(bad)}/{len(results)} cases handled without crashing")


if __name__ == "__main__":
    main()
