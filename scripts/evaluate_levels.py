#!/usr/bin/env python3
"""Run every detection level over the corpus and score it (Section 4.3)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio.asr import Transcription
from piiaudio.detect.base import merge
from piiaudio.detect.names import NameListDetector
from piiaudio.detect.ner import NerDetector
from piiaudio.detect.regex_rules import RegexDetector
from piiaudio.evaluate import Span, Tally, score_dialogue, bootstrap
from piiaudio.redact import DEFAULT_MARGIN_MS

MARGIN = DEFAULT_MARGIN_MS / 1000.0


def levels() -> dict:
    rx = RegexDetector()
    return {
        "L1 regex only":            [rx],
        "L2 + western list":        [rx, NameListDetector("western")],
        "L2 + indian list":         [rx, NameListDetector("indian")],
        "L2 + both lists":          [rx, NameListDetector("both")],
        "L4 rules (no NER)":        [rx, NameListDetector("both", use_context=True,
                                                          require_capital=True)],
        "L3 NER only":              [rx, NerDetector()],
        "L4 rules + NER":           [rx, NameListDetector("both", use_context=True,
                                                          require_capital=True),
                                     NerDetector()],
    }


def run(detectors, asr: Transcription, duration: float) -> list[Span]:
    dets = []
    for d in detectors:
        dets += d.detect(asr.text)
    out = []
    for d in merge(dets):
        t = asr.char_to_time(d.char_start, d.char_end)
        if t:
            out.append(Span(d.pii, max(0.0, t[0] - MARGIN), min(duration, t[1] + MARGIN)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    man = [m for m in json.loads((ROOT/"data"/"manifest.json").read_text())
           if args.split == "all" or m["split"] == args.split]

    loaded = []
    for m in man:
        g = ROOT/"data"/"audio"/f"{m['id']}.gold.json"
        a = ROOT/"data"/"asr"/f"{m['id']}.small.beam5.json"
        if not (g.exists() and a.exists()):
            continue
        gold = json.loads(g.read_text())
        loaded.append((gold, Transcription.from_dict(json.loads(a.read_text()))))

    print(f"split={args.split}  dialogues={len(loaded)}  "
          f"gold spans={sum(len(g['spans']) for g,_ in loaded)}\n")

    rows = {}
    for label, dets in levels().items():
        tally = Tally()
        per_dialogue = []
        for gold, asr in loaded:
            g = [Span(s["pii"], s["start"], s["end"]) for s in gold["spans"]]
            p = run(dets, asr, gold["duration"])
            score_dialogue(g, p, gold["duration"], tally)
            per_dialogue.append((g, p, gold["duration"]))
        m = tally.metrics()
        ci = bootstrap(per_dialogue, n=args.boot)
        rows[label] = (m, ci, tally)

    print(f"{'LEVEL':<22}{'recall':>18}{'precision':>16}{'F2':>18}{'over-redact':>13}")
    for label, (m, ci, _) in rows.items():
        r = f"{m['recall']:.3f} [{ci['recall'][0]:.2f}-{ci['recall'][1]:.2f}]"
        p = f"{m['precision']:.3f}"
        f2 = f"{m['f2']:.3f} [{ci['f2'][0]:.2f}-{ci['f2'][1]:.2f}]"
        print(f"{label:<22}{r:>18}{p:>16}{f2:>18}{m['over_redaction']:>12.1%}")

    print(f"\n{'PER-TYPE RECALL':<22}" + "".join(f"{k[:9]:>10}" for k in
          ("GIVEN_NAM","SURNAME","ADDRESS","LANDMARK","PHONE","EMAIL","AADHAAR","PAN")))
    for label, (_, _, tally) in rows.items():
        cells = []
        for k in ("GIVEN_NAME","SURNAME","ADDRESS","LANDMARK","PHONE","EMAIL","AADHAAR","PAN"):
            t = tally.per_type.get(k)
            cells.append(f"{t.tp/(t.tp+t.fn):>9.0%}" if t and (t.tp+t.fn) else "        -")
        print(f"{label:<22}" + " ".join(cells))


if __name__ == "__main__":
    main()
