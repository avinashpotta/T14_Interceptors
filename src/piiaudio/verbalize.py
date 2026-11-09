"""Generate identifiers and render them the way people actually say them.

Why this is its own module
--------------------------
Day 1 finding F1: Whisper silently normalised "nine eight seven six five four
three two one zero" into `9876543210`, which made the regex work rather than
fail. That turns "how was the number read aloud" from an incidental detail
into the independent variable of the experiment: we need to know whether that
normalisation is *consistent*, because unpredictable normalisation is worse
than reliable failure - you cannot write a rule against it.

So every numeric identifier is generated with a known canonical value and
rendered in one of several realistic speaking styles. The corpus then lets us
ask, per style: did Whisper give us digits, words, or a mangled mixture?

Note we can never write bare digits into the TTS text - `say` would read
"9876543210" as "nine billion eight hundred seventy-six million...". The
spoken form has to be spelled out explicitly, which is exactly why the
canonical value must be tracked separately.
"""

from __future__ import annotations

import random

_ONES = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
_TENS = {
    2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
    6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety",
}
_TEENS = {
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
}

# The styles we actually hear on Indian support calls.
STYLES = ("digit_by_digit", "with_doubles", "paired", "grouped")


def _two_digit_word(pair: str) -> str:
    n = int(pair)
    if n < 10:
        return f"zero {_ONES[pair[1]]}" if pair[0] == "0" else _ONES[pair[1]]
    if n in _TEENS:
        return _TEENS[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]} {_ONES[str(ones)]}"


def digits_to_words(digits: str, style: str, rng: random.Random | None = None) -> str:
    """Render a digit string as speech. `style` is one of STYLES."""
    rng = rng or random.Random()

    if style == "digit_by_digit":
        return " ".join(_ONES[d] for d in digits)

    if style == "with_doubles":
        # "double seven" for runs of two - very common, and a real stress test
        out: list[str] = []
        i = 0
        while i < len(digits):
            if i + 1 < len(digits) and digits[i] == digits[i + 1]:
                out.append(f"double {_ONES[digits[i]]}")
                i += 2
            else:
                out.append(_ONES[digits[i]])
                i += 1
        return " ".join(out)

    if style == "paired":
        pairs = [digits[i:i + 2] for i in range(0, len(digits) - 1, 2)]
        words = [_two_digit_word(p) for p in pairs]
        if len(digits) % 2:
            words.append(_ONES[digits[-1]])
        return ", ".join(words)

    if style == "grouped":
        # threes, with a comma pause: "nine eight seven, six five four, ..."
        groups = [digits[i:i + 3] for i in range(0, len(digits), 3)]
        return ", ".join(" ".join(_ONES[d] for d in g) for g in groups)

    raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")


# ---------------------------------------------------------------- generators

def make_phone(rng: random.Random) -> str:
    """Indian mobile: 10 digits, first in 6-9."""
    return rng.choice("6789") + "".join(rng.choice("0123456789") for _ in range(9))


def make_aadhaar(rng: random.Random) -> str:
    """12 digits; the real scheme never starts 0 or 1."""
    return rng.choice("23456789") + "".join(rng.choice("0123456789") for _ in range(11))


def make_pan(rng: random.Random) -> str:
    """5 letters, 4 digits, 1 letter."""
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(rng.choice(L) for _ in range(5))
        + "".join(rng.choice("0123456789") for _ in range(4))
        + rng.choice(L)
    )


def pan_to_words(pan: str) -> str:
    """Letters spoken individually, digits individually."""
    return " ".join(c if c.isalpha() else _ONES[c] for c in pan)


def email_to_words(email: str) -> str:
    """'rose.mathew@gmail.com' -> 'rose dot mathew at gmail dot com'."""
    local, _, domain = email.partition("@")
    spoken_local = local.replace(".", " dot ").replace("_", " underscore ")
    spoken_domain = domain.replace(".", " dot ")
    return f"{spoken_local} at {spoken_domain}".replace("  ", " ")
