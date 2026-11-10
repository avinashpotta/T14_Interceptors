#!/usr/bin/env python3
"""How much does degraded audio hurt? (§4.2, §5)

Paired across conditions: every condition is the SAME dialogues, so the
difference from clean is measured on matched pairs rather than between
independent samples.
"""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio.align import project_span, token_map, tokenise
from piiaudio.asr import Transcription
from piiaudio.detect.base import merge
from piiaudio.detect.names import NameListDetector
from piiaudio.detect.ner import NerDetector
from piiaudio.detect.regex_rules import RegexDetector
from piiaudio.evaluate import Span, Tally, score_dialogue, paired_delta
from piiaudio.redact import DEFAULT_MARGIN_MS

M = DEFAULT_MARGIN_MS / 1000.0
CONDS = ["clean", "noise20", "noise10", "echo", "phone", "phone_noisy"]


def cache(did, cond):
    name = f"{did}.small.beam5.json" if cond == "clean" else f"{did}.{cond}.small.beam5.json"
    return ROOT / "data" / "asr" / name


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test")
    args = ap.parse_args()
    man = [m for m in json.loads((ROOT/"data"/"manifest.json").read_text())
           if m["split"] == args.split]
    D = [RegexDetector(),
         NameListDetector("both", use_context=True, require_capital=True),
         NerDetector()]

    rows, wer_like, attrib = {}, {}, {}
    for cond in CONDS:
        per, lost, neither, both_, rescued, n = [], 0, 0, 0, 0, 0
        word_err, word_tot = 0, 0
        for m in man:
            g = ROOT/"data"/"audio"/f"{m['id']}.gold.json"
            c = cache(m["id"], cond)
            if not (g.exists() and c.exists()):
                continue
            gold = json.loads(g.read_text())
            asr = Transcription.from_dict(json.loads(c.read_text()))
            pred = []
            for d in merge([x for det in D for x in det.detect(asr.text)]):
                t = asr.char_to_time(d.char_start, d.char_end)
                if t:
                    pred.append(Span(d.pii, max(0.0, t[0]-M), min(gold["duration"], t[1]+M)))
            per.append(([Span(s["pii"], s["start"], s["end"]) for s in gold["spans"]],
                        pred, gold["duration"]))

            # how badly did transcription suffer, restricted to PII words?
            rt, ht = tokenise(gold["transcript"]), tokenise(asr.text)
            mp = token_map(gold["transcript"], asr.text)
            for s in gold["spans"]:
                proj = project_span(gold["transcript"], asr.text,
                                    s["char_start"], s["char_end"], mp, ht, rt)
                word_tot += 1
                if proj is None:
                    word_err += 1
        t = Tally()
        for gs, ps, d in per:
            score_dialogue(gs, ps, d, t)
        rows[cond] = (t.metrics(), per, t)
        wer_like[cond] = word_err / max(1, word_tot)

    print(f"DEGRADED AUDIO  (split={args.split}, L4 rules + NER, {len(rows['clean'][1])} dialogues)\n")
    print(f"  {'condition':<10}{'recall':>9}{'precision':>11}{'F2':>8}{'over-red':>10}"
          f"{'ΔF2 vs clean':>24}{'PII lost by ASR':>17}")
    for cond in CONDS:
        m, per, _ = rows[cond]
        if cond == "clean":
            delta = ""
        else:
            d = paired_delta(rows["clean"][1], per, n=600)
            delta = f"{d['observed']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
        print(f"  {cond:<10}{m['recall']:>9.3f}{m['precision']:>11.3f}{m['f2']:>8.3f}"
              f"{m['over_redaction']:>9.1%}{delta:>24}{wer_like[cond]:>16.0%}")

    print(f"\n  {'PER-TYPE RECALL':<12}" + "".join(f"{c:>10}" for c in CONDS))
    for k in ("GIVEN_NAME","SURNAME","ADDRESS","EMAIL","PHONE","AADHAAR","PAN","LANDMARK"):
        cells = []
        for cond in CONDS:
            t = rows[cond][2].per_type.get(k)
            cells.append(f"{t.tp/(t.tp+t.fn):>9.0%}" if t and (t.tp+t.fn) else "        -")
        print(f"  {k:<12}" + " ".join(cells))
    print("\n  'PII lost by ASR' = gold spans that could not be located in the "
          "transcript at all")


if __name__ == "__main__":
    main()
