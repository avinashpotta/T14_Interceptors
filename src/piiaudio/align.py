"""Token alignment between a reference transcript and an ASR transcript.

Needed for the error-attribution experiment. On the reference transcript a PII
span sits at known character offsets. In the ASR transcript it may have moved,
shrunk ("nine eight seven six five four three two one zero" -> "9876543210"),
been mangled ("Mathew" -> "Matthew") or vanished entirely. To ask "did the
detector find this item?" under both conditions with the SAME criterion, we
need to know where the item ended up.

The `replace` opcode is the interesting one: that is precisely where the ASR
damaged a word, which is precisely where the PII usually is. We map those
positionally rather than discarding them, because a mangled name still
occupies a position in the text.
"""

from __future__ import annotations

import difflib
import re

_TOKEN_RE = re.compile(r"\S+")


def tokenise(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _norm(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def token_map(ref: str, hyp: str) -> dict[int, int | None]:
    """Map reference token index -> hypothesis token index (or None)."""
    r = [_norm(t) for t, _, _ in tokenise(ref)]
    h = [_norm(t) for t, _, _ in tokenise(hyp)]
    out: dict[int, int | None] = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out[i1 + k] = j1 + k
        elif tag == "replace":
            n, m = i2 - i1, j2 - j1
            for k in range(n):
                out[i1 + k] = (j1 + min(m - 1, k * m // max(1, n))) if m else None
        elif tag == "delete":
            for k in range(i1, i2):
                out[k] = None
    return out


def project_span(ref: str, hyp: str, char_start: int, char_end: int,
                 mapping: dict[int, int | None] | None = None,
                 hyp_tokens: list[tuple[str, int, int]] | None = None,
                 ref_tokens: list[tuple[str, int, int]] | None = None,
                 ) -> tuple[int, int] | None:
    """Where does ref[char_start:char_end] land in `hyp`? None if destroyed."""
    rt = ref_tokens if ref_tokens is not None else tokenise(ref)
    ht = hyp_tokens if hyp_tokens is not None else tokenise(hyp)
    mp = mapping if mapping is not None else token_map(ref, hyp)

    idx = [i for i, (_, cs, ce) in enumerate(rt) if ce > char_start and cs < char_end]
    hit = [mp.get(i) for i in idx]
    hit = [h for h in hit if h is not None]
    if not hit:
        return None
    return ht[min(hit)][1], ht[max(hit)][2]
