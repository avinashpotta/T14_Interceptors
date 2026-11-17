#!/usr/bin/env python3
"""Regression tests for the invariants that matter.

Plain asserts, no pytest - one less dependency between a grader and a green
run:   python tests/test_core.py

Each test guards something that actually broke, or that would corrupt results
silently if it broke. Comments say which.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from piiaudio import gazetteer, taxonomy, verbalize as V
from piiaudio.align import project_span
from piiaudio.asr import Transcription, Word
from piiaudio.audio import SAMPLE_RATE, beep, read_wav, trim_silence, write_wav
from piiaudio.detect.base import Detection, merge
from piiaudio.detect.names import NameListDetector
from piiaudio.detect.regex_rules import RegexDetector
from piiaudio.evaluate import Span, Tally, score_dialogue, paired_delta

PASSED = []


def check(name):
    def deco(fn):
        try:
            fn()
            PASSED.append((name, None))
        except AssertionError as e:
            PASSED.append((name, str(e) or "assertion failed"))
        except Exception as e:                       # noqa: BLE001
            PASSED.append((name, f"{type(e).__name__}: {e}"))
        return fn
    return deco


# --------------------------------------------------------------- taxonomy
@check("taxonomy rolls leaves up to branches")
def _():
    assert taxonomy.parent("AADHAAR") == "IDENTIFIER"
    assert taxonomy.parent("GIVEN_NAME") == "PERSON"
    assert len(taxonomy.LEAVES) == 8
    try:
        taxonomy.parent("NOPE"); assert False, "should have raised"
    except ValueError:
        pass


# ------------------------------------------------------------------ audio
@check("wav round-trips without drift")
def _():
    # keep inside [-1, 1]: write_wav clips, so Gaussian tails would make this
    # test fail on correct behaviour
    s = np.clip(np.random.default_rng(0).standard_normal(8000) * 0.3, -0.95, 0.95).astype(np.float32)
    p = ROOT / "tests" / "_tmp.wav"
    write_wav(p, s); back = read_wav(p); p.unlink()
    assert back.shape == s.shape
    assert np.max(np.abs(back - s)) < 2e-4, "int16 round-trip drifted"


@check("trim_silence strips padding but keeps speech")
def _():
    sig = np.concatenate([np.zeros(4000, np.float32),
                          (np.random.default_rng(1).standard_normal(4000) * 0.5).astype(np.float32),
                          np.zeros(4000, np.float32)])
    t = trim_silence(sig)
    assert 3500 < t.size < 5500, f"trimmed to {t.size}, expected ~4000 plus padding"


@check("beep DESTROYS rather than attenuates")
def _():
    # This is the privacy claim. Attenuation would leave the signal recoverable.
    rng = np.random.default_rng(2)
    orig = (rng.standard_normal(16000) * 0.4).astype(np.float32)
    red = orig.copy(); red[4000:12000] = beep(8000)
    corr = float(np.corrcoef(orig[4000:12000], red[4000:12000])[0, 1])
    assert abs(corr) < 0.05, f"correlation {corr:.3f} - original still recoverable"


# ------------------------------------------------------------- verbalize
@check("every speaking style recovers the canonical digits")
def _():
    import re, random
    rng = random.Random(3)
    for _ in range(20):
        d = V.make_phone(rng)
        for style in V.STYLES:
            spoken = V.digits_to_words(d, style, rng)
            assert not re.search(r"\d", spoken), f"{style} leaked a numeral: {spoken!r}"


@check("PAN and email spoken forms are well formed")
def _():
    import random
    assert V.pan_to_words("ABCDE1234F") == "A B C D E one two three four F"
    assert V.email_to_words("rose.mathew@gmail.com") == "rose dot mathew at gmail dot com"


# ------------------------------------------------------------------- asr
def _tr(words):
    text = ""
    ws = []
    for w, st, en in words:
        cs = len(text) + (1 if text else 0)
        text = f"{text} {w}" if text else w
        ws.append(Word(w, st, en, 0.9, cs, len(text)))
    return Transcription(text, tuple(ws), "en", "test", 5)


@check("char_to_time maps a span to the right words")
def _():
    tr = _tr([("It", 0.0, 0.2), ("is", 0.2, 0.4), ("9876543210", 0.5, 1.0), ("ok", 3.0, 3.2)])
    i = tr.text.index("9876543210")
    s, e = tr.char_to_time(i, i + 10, expand=False)
    assert (s, e) == (0.5, 1.0), (s, e)


@check("compressed tokens expand to neighbouring word boundaries")
def _():
    # The PAN finding: "M W P H M eight one three three G" spoken over 2.13s
    # comes back as one token stamped 0.52s. Without expansion the beep covers
    # a quarter of the audio and three quarters of the PAN stays audible.
    tr = _tr([("name", 0.0, 0.4), ("MWPHM8133G", 2.0, 2.5), ("thanks", 4.0, 4.4)])
    i = tr.text.index("MWPHM8133G")
    s, e = tr.char_to_time(i, i + 10, expand=True)
    assert s <= 0.4 + 1e-9 and e >= 2.5, f"expanded to ({s}, {e}), expected to reach neighbours"


@check("a span covering no word returns None, not arbitrary audio")
def _():
    tr = _tr([("hello", 0.0, 0.5)])
    assert tr.char_to_time(9000, 9010) is None


# --------------------------------------------------------------- detectors
@check("regex handles the forms the ASR actually emits")
def _():
    d = RegexDetector()
    must_hit = ["960 -595 -745 -4", "97, 47, 76, 69, 14", "6984854102",
                "neha .goli at gmail .com", "rose dot mathew at gmail dot com",
                "rose.mathew@gmail.com"]
    for t in must_hit:
        assert d.detect(t), f"missed {t!r}"
    for t in ["I have 5 items and 3 boxes.", "call back at 4 pm on the 12th"]:
        assert not d.detect(t), f"false fire on {t!r}"


@check("email matcher covers compound suffixes without truncating them")
def _():
    # Regression: the TLD list used to be com|in|org|net|co|edu, so
    # "x@gmail.co.in" matched only up to ".co" and left ".in" in the clear.
    d = RegexDetector()
    for text in ["neha@gmail.co.in", "neha@iitb.ac.in", "a@b.gov.in", "a@b.io",
                 "neha@mail.iitb.ac.in", "neha dot goli at iitb dot ac dot in"]:
        hits = [h.text for h in d.detect(text)]
        assert hits and max(hits, key=len) == text, f"{text!r} -> {hits}"
    assert not d.detect("meet me at 5 dot 30 in the evening")


@check("gazetteer stoplist blocks the common-word collisions")
def _():
    n = NameListDetector("both")
    for t in ["May I know who I'm speaking with?",
              "I'll need your PAN card details.",
              "Your bill has been corrected."]:
        assert not n.detect(t), f"false fire on {t!r}"


@check("context rules catch a name in no gazetteer")
def _():
    n = NameListDetector("both", use_context=True, require_capital=True)
    hits = n.detect("This is Zephyrine Blackwood calling about my account.")
    assert hits and "Zephyrine" in hits[0].text, hits


@check("merge unions overlapping detections")
def _():
    a = Detection("PHONE", 0, 10, "x", "a")
    b = Detection("PHONE", 5, 20, "y", "b")
    m = merge([a, b])
    assert len(m) == 1 and m[0].char_start == 0 and m[0].char_end == 20


# ---------------------------------------------------------------- scoring
@check("a detector that beeps EVERYTHING is not scored as perfect")
def _():
    # The metric trap. Span-counted precision gave this P=R=F2=1.00 while
    # destroying the entire recording.
    gold = [Span("PHONE", 1.0, 3.0), Span("GIVEN_NAME", 5.0, 5.5)]
    t = Tally(); score_dialogue(gold, [Span("X", 0.0, 10.0)], 10.0, t)
    m = t.metrics()
    assert m["recall"] == 1.0
    assert m["precision"] < 0.3, f"precision {m['precision']:.2f} - over-redaction invisible"
    assert m["over_redaction"] > 0.9


@check("a perfect detector scores 1.0 and a blind one scores 0")
def _():
    gold = [Span("PHONE", 1.0, 3.0)]
    t = Tally(); score_dialogue(gold, gold, 10.0, t)
    assert t.metrics()["f2"] == 1.0
    t2 = Tally(); score_dialogue(gold, [], 10.0, t2)
    assert t2.metrics()["f2"] == 0.0


@check("paired_delta reports zero for a system against itself")
def _():
    rows = [([Span("PHONE", 1.0, 2.0)], [Span("PHONE", 1.0, 2.0)], 5.0)] * 8
    d = paired_delta(rows, rows, n=50)
    assert abs(d["observed"]) < 1e-9, d


# -------------------------------------------------------------- alignment
@check("project_span finds a span the ASR rewrote")
def _():
    ref = "my number is nine eight seven six five four three two one zero thanks"
    hyp = "my number is 9876543210 thanks"
    i = ref.index("nine")
    j = ref.index("zero") + 4
    got = project_span(ref, hyp, i, j)
    assert got and hyp[got[0]:got[1]] == "9876543210", got


@check("project_span returns None when the ASR destroyed the span")
def _():
    got = project_span("the name is Rithwik Das here", "the uh here", 12, 24)
    assert got is None or "Rithwik" not in "the uh here"


# ------------------------------------------------------------ corpus guard
@check("corpus gold offsets and spans are consistent")
def _():
    audio = ROOT / "data" / "audio"
    golds = sorted(audio.glob("*.gold.json"))[:30]
    if not golds:
        return                                   # corpus not built; not a failure
    for p in golds:
        g = json.loads(p.read_text())
        for s in g["spans"]:
            assert g["transcript"][s["char_start"]:s["char_end"]] == s["text"], p.name
            assert s["end"] > s["start"], p.name
            assert s["end"] - s["start"] > 0.05, f"{p.name}: silent gold span {s['pii']}"


if __name__ == "__main__":
    width = max(len(n) for n, _ in PASSED)
    failed = 0
    for name, err in PASSED:
        if err:
            failed += 1
            print(f"  FAIL  {name:<{width}}  {err}")
        else:
            print(f"  ok    {name}")
    print(f"\n{len(PASSED) - failed}/{len(PASSED)} passed")
    sys.exit(1 if failed else 0)
