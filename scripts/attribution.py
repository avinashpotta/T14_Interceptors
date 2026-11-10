#!/usr/bin/env python3
"""The error-attribution experiment (Section 5).

Run the IDENTICAL detector twice - once on the reference transcript, once on
Whisper's - and diff. The gap is error contributed by speech recognition
rather than by detection, and it tells us which half of the pipeline deserves
any remaining effort.

Both conditions are scored in character space with the same criterion (any
overlap with the item's location), so the only thing that differs is the text
the detector was given. Gold spans are projected into ASR coordinates by token
alignment; a span that cannot be projected was destroyed outright by the ASR
and counts as an ASR-caused miss by construction.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio.align import project_span, token_map, tokenise
from piiaudio.asr import Transcription
from piiaudio.detect.base import merge
from piiaudio.detect.names import NameListDetector
from piiaudio.detect.ner import NerDetector
from piiaudio.detect.regex_rules import RegexDetector


def detectors():
    return [RegexDetector(),
            NameListDetector("both", use_context=True, require_capital=True),
            NerDetector()]


def found(dets, text: str, span: tuple[int, int]) -> bool:
    s, e = span
    return any(d.char_end > s and d.char_start < e for d in merge(dets))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    man = [m for m in json.loads((ROOT/"data"/"manifest.json").read_text())
           if args.split == "all" or m["split"] == args.split]
    D = detectors()

    stat = collections.defaultdict(collections.Counter)
    destroyed_examples = []

    for m in man:
        g = ROOT/"data"/"audio"/f"{m['id']}.gold.json"
        a = ROOT/"data"/"asr"/f"{m['id']}.small.beam5.json"
        if not (g.exists() and a.exists()):
            continue
        gold = json.loads(g.read_text())
        asr = Transcription.from_dict(json.loads(a.read_text()))
        ref_text, hyp_text = gold["transcript"], asr.text

        dets_ref = [d for det in D for d in det.detect(ref_text)]
        dets_hyp = [d for det in D for d in det.detect(hyp_text)]
        rt, ht = tokenise(ref_text), tokenise(hyp_text)
        mp = token_map(ref_text, hyp_text)

        for sp in gold["spans"]:
            k = sp["pii"]
            on_ref = found(dets_ref, ref_text, (sp["char_start"], sp["char_end"]))
            proj = project_span(ref_text, hyp_text, sp["char_start"], sp["char_end"],
                                mp, ht, rt)
            on_hyp = found(dets_hyp, hyp_text, proj) if proj else False

            stat[k]["n"] += 1
            # exactly one of four cells
            if on_ref and on_hyp:       cell = "both"
            elif on_ref and not on_hyp: cell = "lost"
            elif on_hyp:                cell = "rescued"
            else:                       cell = "neither"
            stat[k][cell] += 1
            if cell == "lost" and len(destroyed_examples) < 8:
                heard = hyp_text[proj[0]:proj[1]] if proj else "<destroyed>"
                destroyed_examples.append((k, sp["text"][:30], heard[:30]))

    print(f"ERROR ATTRIBUTION  (split={args.split}, L4 rules + NER)\n")
    print("  Each PII item falls in exactly one cell: found on the reference")
    print("  transcript (what was said) x found on Whisper's transcript.\n")
    print(f"  {'TYPE':<12}{'n':>5}{'both':>8}{'LOST to ASR':>13}{'RESCUED by ASR':>16}{'neither':>9}")
    tot = collections.Counter()
    for k in sorted(stat, key=lambda k: -stat[k]["n"]):
        s_ = stat[k]; n = s_["n"]; tot.update(s_)
        print(f"  {k:<12}{n:>5}{s_['both']/n:>7.0%}{s_['lost']/n:>13.0%}"
              f"{s_['rescued']/n:>16.0%}{s_['neither']/n:>9.0%}")
    n = tot["n"]
    print(f"  {'ALL':<12}{n:>5}{tot['both']/n:>7.0%}{tot['lost']/n:>13.0%}"
          f"{tot['rescued']/n:>16.0%}{tot['neither']/n:>9.0%}")

    fail = tot["lost"] + tot["neither"]
    print(f"\n  Of the {fail} items we fail to redact:")
    print(f"    speech recognition destroyed it  : {tot['lost']/fail:>5.0%}  ({tot['lost']})")
    print(f"    detector could not find it anyway: {tot['neither']/fail:>5.0%}  ({tot['neither']})")

    print("\n  examples lost purely to transcription:")
    for k, said, heard in destroyed_examples:
        print(f"    {k:<11} said {said!r} -> heard {heard!r}")


if __name__ == "__main__":
    main()
