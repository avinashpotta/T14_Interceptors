#!/usr/bin/env python3
"""Turn phone recordings into verifiable ground truth.

    python scripts/ingest_human.py P1 ~/Downloads/alex_recordings/

The detector is deliberately NOT involved in any of this. We know exactly what
the reader was given, so PII is located by matching the KNOWN SCRIPT against
the transcript - never by detecting it. If we beeped our detector's output and
asked a human "can you still hear the name?", a yes would be ambiguous between
a detector miss (the result we are trying to measure) and a bad label, and we
would end up editing ground truth until our system looked perfect.

So: script supplies the labels, Whisper supplies the timings, a human confirms
the boundaries. Only then does evaluation run.

Outputs per recording:
    <reader>_<id>.wav          16 kHz mono, ready for the pipeline
    <reader>_<id>.proposed.json  candidate gold, with per-span confidence
    <reader>_<id>.check.wav    proposed spans beeped - listen to verify
    <reader>_<id>.labels.txt   Audacity label track, for the ones that fail
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio.asr import transcribe
from piiaudio.audio import SAMPLE_RATE, read_wav, write_wav, beep

AUDIO_EXT = {".m4a", ".mp3", ".wav", ".aac", ".opus", ".ogg", ".caf", ".aiff", ".flac"}
# No padding on a proposed boundary, deliberately. Padding would make the
# check audio insensitive to the very error it exists to catch: beep 120 ms
# either side and a boundary that is 100 ms wrong still sounds clean, so the
# listener confirms a bad label. Ground truth is the true extent of the
# speech; the redaction margin is a separate, reported parameter that lives
# in redact.py. Padding here also made adjacent spans (a given name and the
# surname right after it) overlap.
MARGIN = 0.0
LOW_CONF = 0.55        # token similarity below which a span is flagged


def norm(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def tokenise(text: str) -> list[tuple[str, int, int]]:
    """(token, char_start, char_end) for every word-ish run."""
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def align(ref: list[str], asr: list[str]) -> dict[int, int | None]:
    """Map each reference token index to an ASR token index, or None.

    difflib gives us the exact matches for free. The interesting case is
    `replace` - that is where Whisper mangled a word (Mathew -> Matthew,
    Pearl -> Hull), which is exactly where the PII usually is. We map those
    positionally rather than dropping them, because a mangled name still
    occupies a position in time and that is what we need.
    """
    out: dict[int, int | None] = {}
    sm = difflib.SequenceMatcher(a=ref, b=asr, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out[i1 + k] = j1 + k
        elif tag == "replace":
            n, m = i2 - i1, j2 - j1
            for k in range(n):
                out[i1 + k] = (j1 + min(m - 1, k * m // max(1, n))) if m else None
        elif tag == "delete":
            for k in range(i1, i2):
                out[k] = None
    return out


def propose(gold: dict, tr) -> list[dict]:
    ref_toks = tokenise(gold["transcript"])
    asr_toks = [(w.text, w.start, w.end) for w in tr.words]
    mapping = align([norm(t) for t, _, _ in ref_toks], [norm(t) for t, _, _ in asr_toks])

    spans = []
    for sp in gold["spans"]:
        idx = [i for i, (_, cs, ce) in enumerate(ref_toks)
               if ce > sp["char_start"] and cs < sp["char_end"]]
        hit = [mapping.get(i) for i in idx]
        hit = [h for h in hit if h is not None]
        if not hit:
            spans.append({**sp, "located": False, "confidence": 0.0,
                          "start": None, "end": None, "asr_text": None})
            continue
        lo, hi = min(hit), max(hit)
        got = [asr_toks[j] for j in range(lo, hi + 1)]
        heard = norm(" ".join(t for t, _, _ in got))
        # Compare against BOTH the spoken form and the canonical value. A
        # perfectly transcribed phone number reads "9584917185" where the
        # reader said "nine five eight four..." - zero string overlap despite
        # being exactly right. Scoring only against the spoken form would
        # flag every correct numeric span for manual review.
        forms = [norm(sp["text"])]
        if sp.get("canonical"):
            forms.append(norm(sp["canonical"]))
        sim = max(difflib.SequenceMatcher(a=f, b=heard).ratio() for f in forms)
        spans.append({
            **sp,
            "located": True,
            "confidence": round(sim, 3),
            "start": round(max(0.0, got[0][1] - MARGIN), 3),
            "end": round(got[-1][2] + MARGIN, 3),
            "asr_text": " ".join(t for t, _, _ in got),
        })
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reader", help="reader id, e.g. P1")
    ap.add_argument("indir", help="directory of their recordings, named 1..10")
    ap.add_argument("--outdir", default=str(ROOT / "data" / "human"))
    args = ap.parse_args()

    assign = json.loads((ROOT/"data"/"recording_kit"/"assignment.json").read_text())
    if args.reader not in assign["readers"]:
        sys.exit(f"unknown reader {args.reader}; expected one of {list(assign['readers'])}")
    mine = assign["readers"][args.reader]

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    files = {}
    for p in sorted(Path(args.indir).iterdir()):
        if p.suffix.lower() in AUDIO_EXT:
            m = re.search(r"\d+", p.stem)
            if m:
                files[int(m.group(0))] = p

    missing = [i for i in range(1, len(mine) + 1) if i not in files]
    if missing:
        print(f"  WARNING: no recording found for {missing}")

    flagged = []
    for n, did in enumerate(mine, 1):
        if n not in files:
            continue
        stem = f"{args.reader}_{did}"
        wav = out / f"{stem}.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(files[n]),
                        "-ar", str(SAMPLE_RATE), "-ac", "1", "-y", str(wav)], check=True)

        tr = transcribe(str(wav))
        gold = json.loads((ROOT/"data"/"audio"/f"{did}.gold.json").read_text())
        spans = propose(gold, tr)

        (out / f"{stem}.proposed.json").write_text(json.dumps({
            "id": did, "reader": args.reader, "source": str(files[n].name),
            "reference_transcript": gold["transcript"],
            "asr_transcript": tr.text,
            "spans": spans,
        }, indent=1))

        # check audio: beep the PROPOSED GOLD, not any detector output
        samples = read_wav(wav)
        for sp in spans:
            if not sp["located"]:
                continue
            i0, i1 = int(sp["start"]*SAMPLE_RATE), int(min(sp["end"], samples.size/SAMPLE_RATE)*SAMPLE_RATE)
            if i1 > i0:
                samples[i0:i1] = beep(i1 - i0)
        write_wav(out / f"{stem}.check.wav", samples)

        (out / f"{stem}.labels.txt").write_text("".join(
            f"{sp['start']:.6f}\t{sp['end']:.6f}\t{sp['pii']}\n"
            for sp in spans if sp["located"]))

        bad = [s for s in spans if not s["located"] or s["confidence"] < LOW_CONF]
        status = "OK" if not bad else f"{len(bad)} span(s) need a look"
        print(f"  {n:>2}. {stem:<18} {len(spans)} spans   {status}")
        for s in bad:
            flagged.append((stem, s))

    print(f"\nwrote {len(mine)} recordings to {out}")
    if flagged:
        print(f"\n{len(flagged)} spans to check first (low similarity or unlocated):")
        for stem, s in flagged[:15]:
            got = s["asr_text"] or "NOT FOUND"
            print(f"  {stem:<18} {s['pii']:<11} conf={s['confidence']:.2f}  "
                  f"said {s['text'][:26]!r} -> heard {got[:26]!r}")
    print("\nNext: listen to each *.check.wav. If no PII is audible and nothing")
    print("extra is destroyed, that recording's gold is confirmed.")


if __name__ == "__main__":
    main()
