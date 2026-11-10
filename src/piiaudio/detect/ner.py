"""Neural NER - the model half of Level 4.

Uses spaCy `en_core_web_md`. Not `lg` (560 MB): this machine goes into swap
under memory pressure (D13), and on 200-400 character transcripts the accuracy
difference does not justify 14x the footprint.

LABEL MAPPING, AND THE ORG TRAP
    PERSON        -> PERSON branch
    GPE, LOC, FAC -> LOCATION branch
    ORG           -> DELIBERATELY EXCLUDED

Our dialogues are full of "Nexa Telecom", "Bluestar Bank", "CityFirst Bank" -
the company being *called*. That is not the customer's personal information.
Tagging ORG as PII would beep the brand name in nearly every dialogue and
inflate over-redaction for no privacy gain. (The prior-art repo we looked at
does include ORG in its PII label set, which for this use case is wrong.)

CARDINAL is also excluded: it fires on "five gigabytes", "twelve month",
"between ten and twelve" far more often than on a flat number.

EXPECTATION, TO BE TESTED
Finding F6 showed Western-trained resources cover 11% of our Indian given
names. spaCy is OntoNotes-trained, so we expect it to help substantially on
LOCATION (Bangalore, MG Road are well represented) and much less on Indian
PERSON names. If that is what we measure, it corroborates F6 rather than
contradicting it.
"""

from __future__ import annotations

from functools import lru_cache

from .base import Detection
from ..gazetteer import DOMAIN_STOPWORDS, stoplist

PERSON_LABELS = {"PERSON"}
LOCATION_LABELS = {"GPE", "LOC", "FAC"}
EXCLUDED = {"ORG", "CARDINAL", "DATE", "TIME", "MONEY", "ORDINAL", "PRODUCT",
            "NORP", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "PERCENT", "QUANTITY"}


@lru_cache(maxsize=2)
def _load(model: str):
    import spacy
    return spacy.load(model, disable=["lemmatizer", "textcat"])


class NerDetector:
    name = "ner"
    emits = ("GIVEN_NAME", "ADDRESS")

    def __init__(self, model: str = "en_core_web_md", persons: bool = True,
                 locations: bool = True):
        self.model = model
        self.persons = persons
        self.locations = locations
        self._nlp = None

    def detect(self, text: str) -> list[Detection]:
        if self._nlp is None:
            self._nlp = _load(self.model)
        out: list[Detection] = []
        stop = stoplist()
        for ent in self._nlp(text).ents:
            if ent.label_ in EXCLUDED:
                continue
            # The domain stoplist applies to model output too, not just to the
            # gazetteer. spaCy tags "Aadhaar" and "KYC" as entities because
            # they are capitalised unfamiliar tokens; beeping the word
            # "Aadhaar" protects nobody and destroys the sentence that tells
            # you an Aadhaar number is coming.
            toks = [t.strip(".,?!'").lower() for t in ent.text.split()]
            if toks and all(t in DOMAIN_STOPWORDS or t in stop for t in toks):
                continue
            if ent.label_ in PERSON_LABELS and self.persons:
                pii = "GIVEN_NAME"
            elif ent.label_ in LOCATION_LABELS and self.locations:
                pii = "ADDRESS"
            else:
                continue
            out.append(Detection(
                pii=pii,
                char_start=ent.start_char,
                char_end=ent.end_char,
                text=ent.text,
                source=f"ner:{ent.label_}",
            ))
        return out
