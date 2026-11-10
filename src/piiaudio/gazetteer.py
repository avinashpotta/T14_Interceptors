"""Name gazetteers, and the stoplist that makes them usable.

Two lists, deliberately. Section 1 argues that "most ready-made models are
trained mostly on Western names... Indian names are underrepresented". Running
a Western list and an Indian list side by side turns that from a claim
borrowed from the literature into a number we measured.

  western - NLTK's names corpus (US Census given names), ~7.9k
  indian  - Indian-Male/Female-Names, tokenised to given and family names

NEITHER LIST COMES FROM OUR OWN VOCABULARY. Building the gazetteer from
data/vocab.json would give Level 2 perfect prior knowledge of every name in
the corpus - an oracle, not a baseline - and would make Level 4 look bad for
no real reason. A real deployment cannot enumerate its customers' names in
advance, and neither can this one.

THE STOPLIST IS NOT OPTIONAL. Measured on our own transcripts, a raw list
matching ordinary words scored 24% precision: "May I know who I'm speaking
with" (may), "your PAN card details" (card), "your bill has been corrected"
(bill). Common English words are subtracted, and how many is tuned on dev.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent.parent / "data" / "gazetteer"

# How many of the most frequent English words to subtract. Tuned on dev;
# higher means better precision and worse recall on name-like words.
DEFAULT_STOPLIST_N = 2000

# Words we never treat as names regardless of frequency rank, because they
# are structural to this domain and appear constantly in support calls.
DOMAIN_STOPWORDS = {
    "card", "bank", "support", "account", "number", "mobile", "email",
    "address", "customer", "service", "team", "call", "name", "sir", "madam",
    "pan", "aadhaar", "detail", "details", "please", "thank", "thanks",
}


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z]+", name.lower()) if len(t) > 1]


@lru_cache(maxsize=4)
def stoplist(n: int = DEFAULT_STOPLIST_N) -> frozenset[str]:
    words = (DATA / "common_english.txt").read_text().split()
    return frozenset(w.lower() for w in words[:n]) | DOMAIN_STOPWORDS


@lru_cache(maxsize=8)
def load(source: str, stoplist_n: int = DEFAULT_STOPLIST_N) -> frozenset[str]:
    """Return a name set with common English words removed."""
    raw: set[str] = set()

    if source == "western":
        for f in ("male.txt", "female.txt"):
            p = DATA / "names" / f
            raw.update(w.strip().lower() for w in p.read_text().splitlines() if w.strip())

    elif source == "indian":
        for f in ("indian_male.csv", "indian_female.csv"):
            with (DATA / f).open(newline="", encoding="utf-8", errors="ignore") as fh:
                for row in csv.DictReader(fh):
                    # entries are full names ("ramdin verma"); we want both parts
                    raw.update(_tokens(row.get("name") or ""))

    elif source == "both":
        return frozenset(load("western", stoplist_n) | load("indian", stoplist_n))

    else:
        raise ValueError(f"unknown gazetteer source {source!r}")

    stop = stoplist(stoplist_n)
    return frozenset(w for w in raw if len(w) > 2 and w not in stop)


def coverage(names: list[str], source: str, stoplist_n: int = DEFAULT_STOPLIST_N) -> float:
    """Share of `names` present in a gazetteer - the Section 1 measurement."""
    g = load(source, stoplist_n)
    if not names:
        return 0.0
    return sum(1 for n in names if n.lower() in g) / len(names)
