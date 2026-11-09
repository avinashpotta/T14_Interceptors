#!/usr/bin/env python3
"""Instantiate dialogue templates into renderable scripts.

Two things here are not incidental.

SPLIT BY TEMPLATE, NEVER BY DIALOGUE. Dialogues from one template share their
phrasing; only the slot fills differ. Split at the dialogue level and the same
sentence appears in both halves, so anything that keys on phrasing - a tuned
threshold, a rule, a model - scores on material it has effectively already
seen. We assign whole templates to a split, so dev and test share no wording.
(We train nothing, but we *tune* the detection threshold and redaction margin
on dev and report on test. Same leakage risk, same fix.)

SEAM ONLY AT PII BOUNDARIES. Each rendered segment carries utterance-final
prosody, so every segment boundary is an unnatural pause - a cue the ASR would
not get in real speech. Plain slots like {{city}} are substituted into the
surrounding literal text so they cost no extra seam. Only PII slots break a
segment, because only those need their own timing.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio import taxonomy, verbalize as V
from piiaudio.tts import VOICES

SLOT_RE = re.compile(r"\{\{([A-Za-z_]+)\}\}")
DIALOGUES_PER_TEMPLATE = 4
DEV_FRACTION = 1 / 3
SEED = 14


def _pick_name(vocab, rng):
    """Bias towards ordinary Indian names, but keep a steady supply of the
    hard case: given names that are also common English words."""
    if rng.random() < 0.3:
        return rng.choice(vocab["GIVEN_NAME_wordlike"]), True
    return rng.choice(vocab["GIVEN_NAME_common"]), False


def _make_address(vocab, rng) -> str:
    building, street, city = (rng.choice(vocab[k]) for k in ("building", "street", "city"))
    return rng.choice([
        f"flat {rng.choice(vocab['flat_no'])}, {building}",
        f"{building}, {street}",
        f"flat {rng.choice(vocab['flat_no'])}, {building}, {street}",
        f"{building} on {street}, {city}",
    ])


def _fill_pii(slot, ctx, vocab, rng, style_cycle):
    """Return (spoken_text, canonical, style) for one PII slot."""
    if slot == "GIVEN_NAME":
        return ctx["given"], ctx["given"], None
    if slot == "SURNAME":
        return ctx["surname"], ctx["surname"], None
    if slot == "ADDRESS":
        return _make_address(vocab, rng), None, None
    if slot == "LANDMARK":
        return rng.choice(vocab["LANDMARK"]), None, None
    if slot == "EMAIL":
        return V.email_to_words(ctx["email"]), ctx["email"], "spoken"
    if slot == "PAN":
        pan = V.make_pan(rng)
        return V.pan_to_words(pan), pan, "letters_digits"
    if slot in ("PHONE", "AADHAAR"):
        digits = V.make_phone(rng) if slot == "PHONE" else V.make_aadhaar(rng)
        style = next(style_cycle)
        return V.digits_to_words(digits, style, rng), digits, style
    raise ValueError(f"no filler for PII slot {slot!r}")


def instantiate(template, vocab, rng, style_cycle, index):
    given, wordlike = _pick_name(vocab, rng)
    surname = rng.choice(vocab["SURNAME"])
    ctx = {
        "given": given,
        "surname": surname,
        "email": f"{given.lower()}.{surname.lower()}@{rng.choice(vocab['mail_domain'])}",
    }
    plain = {
        "brand": rng.choice(vocab["brand"]),
        "agent_name": rng.choice(vocab["agent_name"]),
        "city": rng.choice(vocab["city"]),
        "building": rng.choice(vocab["building"]),
        "street": rng.choice(vocab["street"]),
        "flat_no": rng.choice(vocab["flat_no"]),
    }

    agent_voice, customer_voice = rng.sample(VOICES, 2)
    turns = []
    for turn in template["turns"]:
        voice = agent_voice if turn["role"] == "agent" else customer_voice
        segments, buf = [], ""
        pos = 0
        for m in SLOT_RE.finditer(turn["text"]):
            slot = m.group(1)
            buf += turn["text"][pos:m.start()]
            pos = m.end()
            if taxonomy.is_leaf(slot):
                if buf.strip():
                    segments.append({"text": buf.strip()})
                buf = ""
                spoken, canonical, style = _fill_pii(slot, ctx, vocab, rng, style_cycle)
                seg = {"text": spoken, "pii": slot}
                if canonical:
                    seg["canonical"] = canonical
                if style:
                    seg["style"] = style
                segments.append(seg)
            else:
                buf += plain[slot]  # plain slots cost no seam
        buf += turn["text"][pos:]
        if buf.strip():
            segments.append({"text": buf.strip()})
        turns.append({"voice": voice, "speaker": turn["role"], "segments": segments})

    return {
        "id": f"{template['id']}_{index:02d}",
        "template": template["id"],
        "scenario": template["scenario"],
        "wordlike_name": wordlike,
        "turns": turns,
    }


def main() -> None:
    rng = random.Random(SEED)
    vocab = json.loads((ROOT / "data" / "vocab.json").read_text())
    templates = json.loads((ROOT / "data" / "templates.json").read_text())

    # Balanced cycle over speaking styles so each is comparably represented.
    def cycle():
        i = 0
        while True:
            yield V.STYLES[i % len(V.STYLES)]
            i += 1
    style_cycle = cycle()

    # Template-level split, stratified by scenario so both halves see a
    # similar mix of call types.
    by_scenario: dict[str, list] = {}
    for t in templates:
        by_scenario.setdefault(t["scenario"], []).append(t)
    dev_ids, test_ids = set(), set()
    for scenario, group in sorted(by_scenario.items()):
        shuffled = sorted(group, key=lambda t: t["id"])
        rng.shuffle(shuffled)
        n_dev = max(1, round(len(shuffled) * DEV_FRACTION)) if len(shuffled) > 1 else 0
        for t in shuffled[:n_dev]:
            dev_ids.add(t["id"])
        for t in shuffled[n_dev:]:
            test_ids.add(t["id"])

    out_dir = ROOT / "data" / "scripts"
    for old in out_dir.glob("*.json"):
        old.unlink()

    manifest = []
    for t in templates:
        split = "dev" if t["id"] in dev_ids else "test"
        for i in range(DIALOGUES_PER_TEMPLATE):
            script = instantiate(t, vocab, rng, style_cycle, i)
            script["split"] = split
            (out_dir / f"{script['id']}.json").write_text(json.dumps(script, indent=1))
            manifest.append({
                "id": script["id"], "template": t["id"],
                "scenario": t["scenario"], "split": split,
            })

    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=1))

    n_dev = sum(1 for m in manifest if m["split"] == "dev")
    print(f"{len(manifest)} dialogues from {len(templates)} templates")
    print(f"  dev  {n_dev:>4} dialogues from {len(dev_ids)} templates")
    print(f"  test {len(manifest)-n_dev:>4} dialogues from {len(test_ids)} templates")
    print(f"  template overlap between splits: {len(dev_ids & test_ids)}")


if __name__ == "__main__":
    main()
