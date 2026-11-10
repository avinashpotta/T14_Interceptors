#!/usr/bin/env python3
"""Build the human-recording kit: pick dialogues, assign readers, write sheets.

Design decisions worth knowing.

PAIRED, NOT NEW. Readers read scripts that are ALREADY in the TTS corpus, so
every human recording has a synthetic twin with identical text and identical
PII. The only thing that varies is the voice source, which is exactly the
comparison we want. New dialogues would confound voice with text and the
difference would be unattributable.

NO PII MARKUP ON THE SHEET. Readers see plain dialogue, with no indication of
which words are sensitive. Marking them would invite careful over-enunciation
of precisely the words whose natural pronunciation we are trying to measure.

NUMBERS ARE WRITTEN AS WORDS. The sheet carries the exact spoken form the TTS
used ("nine eight seven six..."), so the speaking style is preserved and the
pair stays comparable. Printing "9876543210" would let each reader invent
their own style.

SHARED CONTROL DIALOGUES. A couple of dialogues go to every reader, giving
the same text in five voices - that separates speaker variation from text
variation.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

N_READERS = 5
PER_READER = 10
N_SHARED = 2          # read by everyone
SEED = 14

INSTRUCTIONS = """\
RECORDING INSTRUCTIONS  —  please read before starting

You are reading {n} short customer-support conversations out loud.
Each one takes about 20 seconds. The whole thing should take 20-30 minutes.

SETUP
  * Find a quiet room. Background hum is fine; a TV or other people talking is not.
  * Use your phone's voice recorder. No app to install.
  * Hold the phone about an arm's length away - roughly 20-30 cm. Do NOT hold it
    right against your mouth, it distorts.

HOW TO READ
  * Each conversation has two people, AGENT and CUSTOMER. You read BOTH parts.
    Just read straight through as if reading a short script aloud.
  * Read at a normal conversational pace. Do NOT slow down or over-pronounce the
    names and numbers - that is the single most important thing. We are measuring
    how speech recognition copes with ordinary speech.
  * Numbers are written out as words. Read them exactly as written.
  * IF YOU STUMBLE, KEEP GOING. Small stumbles, repeated words and "um" are
    genuinely useful to us. Only start a recording over if you lose your place
    completely.

SAVING AND SENDING
  * One recording per conversation - {n} files in total.
  * Rename each file to just its number: 1, 2, 3 ... {n}
    (matching the "RECORDING n" heading below)
  * Send all {n} files to Alex, unedited. Do not trim or clean them up.

Thank you - this takes under half an hour and it is the only part of the
project that cannot be generated automatically.
"""


def dialogue_text(script: dict) -> list[tuple[str, str]]:
    """Flatten a dialogue script to (speaker, line) with no PII markup."""
    out = []
    for turn in script["turns"]:
        line = ""
        for seg in turn["segments"]:
            t = seg["text"].strip()
            if not t:
                continue
            # A PAN spelled out letter-by-letter can contain "D O T", which a
            # reader may well say as "dot". Hyphenate character sequences on
            # the sheet so the grouping is unambiguous. This changes only what
            # the reader SEES - the spoken content is identical, so the pair
            # with the TTS twin still holds.
            if seg.get("style") == "letters_digits":
                t = "-".join(t.split())
            line = t if not line else (line + t if t[0] in ".,!?;:" else line + " " + t)
        out.append((turn["speaker"].upper(), line))
    return out


def score_dialogue(script: dict) -> tuple:
    """Rank for selection: prefer type variety, hard cases, and PII density."""
    types, styles, has_wordlike = set(), set(), script.get("wordlike_name", False)
    n_pii = 0
    for turn in script["turns"]:
        for seg in turn["segments"]:
            if "pii" in seg:
                types.add(seg["pii"]); n_pii += 1
                if seg.get("style"):
                    styles.add(seg["style"])
    return (len(types), len(styles), int(has_wordlike), n_pii)


def main() -> None:
    rng = random.Random(SEED)
    scripts = {p.stem: json.loads(p.read_text())
               for p in sorted((ROOT / "data" / "scripts").glob("*.json"))}

    # Greedy selection by MARGINAL coverage gain. Each step picks the dialogue
    # that most improves whatever is currently least covered, across PII types
    # AND speaking styles. A plain template round-robin gave 13 `with_doubles`
    # against 2 `grouped` - and `grouped`/`paired` are precisely the styles
    # where we found the digit-fragmentation effect, so under-sampling them
    # would leave our headline finding untested on real speech.
    def features(did):
        f = []
        for turn in scripts[did]["turns"]:
            for seg in turn["segments"]:
                if "pii" in seg:
                    f.append(("type", seg["pii"]))
                    if seg.get("style"):
                        f.append(("style", seg["style"]))
        if scripts[did].get("wordlike_name"):
            f.append(("hard", "wordlike"))
        return f

    need = N_SHARED + N_READERS * (PER_READER - N_SHARED)
    have, chosen = Counter(), []
    per_template = Counter()
    pool = sorted(scripts)
    rng.shuffle(pool)

    while len(chosen) < need and pool:
        def gain(did):
            # diminishing returns per feature, plus a penalty for re-using a
            # template so no single phrasing dominates the human set
            g = sum(1.0 / (1 + have[f]) for f in features(did))
            return g - 0.35 * per_template[scripts[did]["template"]]
        best = max(pool, key=gain)
        pool.remove(best)
        chosen.append(best)
        have.update(features(best))
        per_template[scripts[best]["template"]] += 1

    shared, unique = chosen[:N_SHARED], chosen[N_SHARED:]
    rng.shuffle(unique)

    kit = ROOT / "data" / "recording_kit"
    kit.mkdir(parents=True, exist_ok=True)
    for old in kit.glob("*"):
        old.unlink()

    assignment = {}
    per = PER_READER - N_SHARED
    for r in range(N_READERS):
        mine = shared + unique[r * per:(r + 1) * per]
        reader = f"P{r+1}"
        assignment[reader] = mine

        lines = [f"RECORDING SHEET  —  reader {reader}", "=" * 60, "",
                 INSTRUCTIONS.format(n=len(mine)), "", "=" * 60, ""]
        for i, did in enumerate(mine, 1):
            lines += [f"RECORDING {i}          (save this file as: {i})",
                      "-" * 60]
            for spk, text in dialogue_text(scripts[did]):
                lines.append(f"  {spk:<9} {text}")
            lines += ["", ""]
        (kit / f"sheet_{reader}.txt").write_text("\n".join(lines))

    (kit / "assignment.json").write_text(json.dumps(
        {"shared": shared, "readers": assignment}, indent=1))

    # report coverage of the selected set
    types = Counter(); styles = Counter(); wordlike = 0
    for did in set(chosen):
        s = scripts[did]
        wordlike += int(s.get("wordlike_name", False))
        for turn in s["turns"]:
            for seg in turn["segments"]:
                if "pii" in seg:
                    types[seg["pii"]] += 1
                    if seg.get("style"): styles[seg["style"]] += 1

    print(f"{len(set(chosen))} distinct dialogues, {N_READERS} readers x {PER_READER} = "
          f"{N_READERS*PER_READER} recordings")
    print(f"  {N_SHARED} shared (read by all {N_READERS}) as an inter-speaker control")
    print(f"  word-like given names: {wordlike}/{len(set(chosen))} dialogues")
    print(f"  PII spans in the selected set: {sum(types.values())}")
    print("\n  per type:", dict(types.most_common()))
    print("  per style:", dict(styles.most_common()))
    print(f"\nwrote {N_READERS} sheets to data/recording_kit/")


if __name__ == "__main__":
    main()
