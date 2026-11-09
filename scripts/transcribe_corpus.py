#!/usr/bin/env python3
"""Transcribe the corpus once and cache to disk.

Run sequentially on purpose: ctranslate2 already saturates the cores with its
own intra-op threads (measured ~314% CPU on one dialogue), so running several
dialogues at once contends rather than helps, and risks memory on 8 GB.

Cache key is (dialogue, model, beam_size). Re-runs skip finished work, so this
is safe to interrupt and resume.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio.asr import transcribe, Transcription


def cache_path(dialogue_id: str, model: str, beam: int) -> Path:
    return ROOT / "data" / "asr" / f"{dialogue_id}.{model}.beam{beam}.json"


def load_cached(dialogue_id: str, model: str, beam: int) -> Transcription | None:
    p = cache_path(dialogue_id, model, beam)
    if p.exists():
        return Transcription.from_dict(json.loads(p.read_text()))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    wavs = sorted((ROOT / "data" / "audio").glob("*.wav"))
    wavs = [w for w in wavs if not w.stem.endswith("_redacted")]
    if args.limit:
        wavs = wavs[: args.limit]

    (ROOT / "data" / "asr").mkdir(parents=True, exist_ok=True)
    t0, skipped, done = time.time(), 0, 0

    for i, wav in enumerate(wavs, 1):
        out = cache_path(wav.stem, args.model, args.beam)
        if out.exists():
            skipped += 1
            continue
        tr = transcribe(str(wav), model_size=args.model, beam_size=args.beam)
        out.write_text(json.dumps(tr.to_dict()))
        done += 1
        if done % 10 == 0:
            rate = (time.time() - t0) / done
            left = (len(wavs) - i) * rate
            print(f"  {i}/{len(wavs)}  {rate:.1f}s each, ~{left/60:.1f} min left", flush=True)

    print(f"\ntranscribed {done}, reused {skipped} cached, in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
