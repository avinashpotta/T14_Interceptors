"""The PII ontology from Section 2 of the proposal.

Categories are a tree, not a flat label set, because the branches genuinely
behave differently and need different machinery:

  IDENTIFIER  - rigid surface form. Regex works on clean text and fails on
                spoken-out digits. This is where the spoken-number problem lives.
  PERSON      - open class. No pattern exists; needs a model or a name list,
                and collides with ordinary English words (Rose, Prince, Happy).
  LOCATION    - semi-structured in theory, landmark-relative in Indian speech
                ("opposite the petrol pump"), so neither regex nor a gazetteer
                is sufficient.

Detectors declare which leaves they can emit; scoring rolls leaves up to
branches so we can report per-category results (Section 5).
"""

from __future__ import annotations

# leaf -> parent branch
_PARENT: dict[str, str] = {
    "GIVEN_NAME": "PERSON",
    "SURNAME": "PERSON",
    "ADDRESS": "LOCATION",
    "LANDMARK": "LOCATION",
    "AADHAAR": "IDENTIFIER",
    "PAN": "IDENTIFIER",
    "PHONE": "IDENTIFIER",
    "EMAIL": "IDENTIFIER",
}

LEAVES: tuple[str, ...] = tuple(_PARENT)
BRANCHES: tuple[str, ...] = ("PERSON", "LOCATION", "IDENTIFIER")


def parent(leaf: str) -> str:
    """Branch that a leaf category rolls up to."""
    try:
        return _PARENT[leaf]
    except KeyError:
        raise ValueError(f"unknown PII category {leaf!r}; expected one of {LEAVES}") from None


def is_leaf(label: str) -> bool:
    return label in _PARENT


def normalise(label: str) -> str:
    """Accept either a leaf or a branch, return it unchanged if valid."""
    if label in _PARENT or label in BRANCHES:
        return label
    raise ValueError(f"unknown PII category {label!r}")
