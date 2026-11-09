#!/usr/bin/env python3
"""Render every dialogue script to audio + ground truth.

Parallel over dialogues: `say` is a subprocess, so threads are the right tool
(the GIL is released while we wait on it). render_dialogue uses its own
temporary directory per call, so it is already thread-safe.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio.tts import render_to_disk

# Two, not six. Six concurrent `say` processes plus Python drove an 8 GB M2
# into swap (load average 15, 2.6 GB swapped) and rendering degraded roughly
# 30x - the first 20 dialogues took 59 s, the next 20 took 32 minutes. The
# binding constraint is memory, not CPU, so a smaller pool is strictly faster
# here. Raise it only on a machine with headroom.
WORKERS = 2


def main() -> None:
    scripts = sorted((ROOT / "data" / "scripts").glob("*.json"))
    out_dir = ROOT / "data" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip dialogues already rendered, so an interrupted run costs
    # nothing. A wav without its gold file (or vice versa) is treated as
    # incomplete and re-rendered.
    pending = [
        p for p in scripts
        if not ((out_dir / f"{p.stem}.wav").exists()
                and (out_dir / f"{p.stem}.gold.json").exists())
    ]
    print(f"{len(scripts)} dialogues, {len(scripts)-len(pending)} already rendered, "
          f"{len(pending)} to do")
    scripts = pending
    if not scripts:
        print("nothing to do")
        return

    t0 = time.time()
    done, failed = 0, []
    total_dur, total_spans = 0.0, 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(render_to_disk, json.loads(p.read_text()), out_dir): p.stem
            for p in scripts
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                gold = fut.result()
                total_dur += gold["duration"]
                total_spans += len(gold["spans"])
            except Exception as exc:  # keep going, report at the end
                failed.append((name, repr(exc)))
            done += 1
            if done % 20 == 0 or done == len(scripts):
                print(f"  {done}/{len(scripts)}  ({time.time()-t0:.0f}s elapsed)", flush=True)

    print(f"\nrendered {done - len(failed)}/{len(scripts)} dialogues in {time.time()-t0:.0f}s")
    print(f"  total audio : {total_dur/60:.1f} min")
    print(f"  PII spans   : {total_spans}")
    print(f"  mean length : {total_dur/max(1, done-len(failed)):.1f}s per dialogue")
    if failed:
        print(f"\n{len(failed)} FAILED:")
        for name, err in failed[:10]:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
