#!/usr/bin/env python3
"""Human speech vs its TTS twin (§5.8).

Every human recording reads a script that is already in the TTS corpus, so
this is a PAIRED comparison: identical text, identical PII, identical
detector. The only thing that varies is whether a person or macOS `say`
produced the audio. Any difference is therefore attributable to the voice
source, which is exactly what D1 declared as this corpus's weakness and what
this experiment exists to quantify.

Scored in CHARACTER space on each transcript, with gold spans projected by
token alignment - the same method as the attribution experiment. This needs no
verified timings, so it is available immediately and is unaffected by any
boundary error in the proposed gold.
"""
from __future__ import annotations
import collections, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio.align import project_span, token_map, tokenise
from piiaudio.asr import Transcription
from piiaudio.detect.base import merge
from piiaudio.detect.names import NameListDetector
from piiaudio.detect.ner import NerDetector
from piiaudio.detect.regex_rules import RegexDetector

D = [RegexDetector(),
     NameListDetector("both", use_context=True, require_capital=True),
     NerDetector()]


def evaluate(ref_text: str, hyp_text: str, spans: list[dict]) -> list[tuple[str, bool, bool]]:
    """-> [(type, located_in_transcript, found_by_detector)] per gold span."""
    rt, ht = tokenise(ref_text), tokenise(hyp_text)
    mp = token_map(ref_text, hyp_text)
    dets = merge([d for det in D for d in det.detect(hyp_text)])
    out = []
    for sp in spans:
        proj = project_span(ref_text, hyp_text, sp["char_start"], sp["char_end"], mp, ht, rt)
        located = proj is not None
        found = bool(proj) and any(d.char_end > proj[0] and d.char_start < proj[1] for d in dets)
        out.append((sp["pii"], located, found))
    return out


def main() -> None:
    human_dir = ROOT / "data" / "human"
    props = sorted(human_dir.glob("*.proposed.json"))
    if not props:
        sys.exit("no ingested recordings in data/human/")

    per_type = collections.defaultdict(lambda: collections.Counter())
    per_reader = collections.defaultdict(lambda: collections.Counter())
    overall = collections.Counter()
    pairs = 0

    for p in props:
        rec = json.loads(p.read_text())
        did, reader = rec["id"], rec["reader"]
        gold_p = ROOT / "data" / "audio" / f"{did}.gold.json"
        tts_asr = ROOT / "data" / "asr" / f"{did}.small.beam5.json"
        if not (gold_p.exists() and tts_asr.exists()):
            continue
        gold = json.loads(gold_p.read_text())
        ref = gold["transcript"]

        h = evaluate(ref, rec["asr_transcript"], gold["spans"])
        t = evaluate(ref, Transcription.from_dict(json.loads(tts_asr.read_text())).text,
                     gold["spans"])
        pairs += 1
        for (k, hl, hf), (_, tl, tf) in zip(h, t):
            for scope in (per_type[k], per_reader[reader], overall):
                scope["n"] += 1
                scope["tts_located"] += tl; scope["tts_found"] += tf
                scope["hum_located"] += hl; scope["hum_found"] += hf

    def row(label, c):
        n = c["n"]
        return (f"  {label:<14}{n:>5}{c['tts_found']/n:>11.0%}{c['hum_found']/n:>11.0%}"
                f"{(c['hum_found']-c['tts_found'])/n:>+9.0%}"
                f"{c['tts_located']/n:>12.0%}{c['hum_located']/n:>11.0%}")

    print(f"HUMAN SPEECH vs TTS TWIN  ({pairs} paired recordings, "
          f"{overall['n']} PII spans)\n")
    print(f"  {'':<14}{'n':>5}{'TTS found':>11}{'human':>11}{'delta':>9}"
          f"{'TTS survived':>12}{'human':>11}")
    for k in sorted(per_type, key=lambda k: -per_type[k]["n"]):
        print(row(k, per_type[k]))
    print(row("ALL", overall))

    print(f"\n  per reader:")
    print(f"  {'':<14}{'n':>5}{'TTS found':>11}{'human':>11}{'delta':>9}"
          f"{'TTS survived':>12}{'human':>11}")
    for r in sorted(per_reader):
        print(row(r, per_reader[r]))

    print("\n  'found'    = the detector located the PII in that transcript")
    print("  'survived' = the PII was still locatable in the transcript at all")
    print("               (the gap between them is detector failure, not ASR failure)")


if __name__ == "__main__":
    main()
