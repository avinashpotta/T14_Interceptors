#!/usr/bin/env python3
"""Render degraded copies of the corpus and transcribe them."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio.asr import transcribe
from piiaudio.audio import read_wav, write_wav
from piiaudio.augment import CONDITIONS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--conditions", default="noise20,noise10,echo")
    args = ap.parse_args()

    man = [m for m in json.loads((ROOT/"data"/"manifest.json").read_text())
           if args.split == "all" or m["split"] == args.split]
    conds = [c.strip() for c in args.conditions.split(",")]
    out_a = ROOT/"data"/"audio_cond"; out_a.mkdir(parents=True, exist_ok=True)
    out_t = ROOT/"data"/"asr"; out_t.mkdir(parents=True, exist_ok=True)

    t0, done, skipped = time.time(), 0, 0
    for cond in conds:
        fn = CONDITIONS[cond]
        for i, m in enumerate(man, 1):
            src = ROOT/"data"/"audio"/f"{m['id']}.wav"
            if not src.exists():
                continue
            cache = out_t/f"{m['id']}.{cond}.small.beam5.json"
            if cache.exists():
                skipped += 1
                continue
            degraded = fn(read_wav(src))
            write_wav(out_a/f"{m['id']}.{cond}.wav", degraded)
            tr = transcribe(degraded)
            cache.write_text(json.dumps(tr.to_dict()))
            done += 1
            if done % 20 == 0:
                rate = (time.time()-t0)/done
                left = (len(man)*len(conds) - done - skipped) * rate
                print(f"  {cond} {i}/{len(man)}  ~{left/60:.1f} min left", flush=True)
    print(f"\ntranscribed {done}, reused {skipped}, in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
