"""Level 3: Microsoft Presidio, off the shelf (§4.3).

CONFIGURED AS CHARITABLY AS IS REASONABLE. Section 4.3 warns that a weak
baseline would be unfair in our favour, and that applies to a competitor as
much as to our own regex level.

  * It keeps every entity type that could land on real PII, including ones
    that match for the wrong reason. A 12-digit Aadhaar comes back as
    US_BANK_NUMBER; we count that as a hit, because our scoring is
    type-agnostic for redaction — audio destroyed is audio destroyed, and the
    person is protected whatever label caused it.
  * Four types are dropped: DATE_TIME, URL, NRP and AGE. None corresponds to
    anything in our gold, so keeping them could only add false positives.
    Dropping them helps Presidio's precision, not ours.
  * It runs on `en_core_web_md`, the same spaCy model our own NER layer uses,
    so the comparison is not confounded by model size.

NOTE ON COVERAGE. This build exposes no India-specific recognizers — there is
no IN_PAN or IN_AADHAAR among its 24 supported entities. That is not a
misconfiguration on our part; it is the observation §3 predicts.
"""

from __future__ import annotations

from functools import lru_cache

from .base import Detection

# Presidio entity -> our taxonomy. Several map to the same leaf because we only
# care whether the audio gets destroyed, not what it was called.
MAPPING = {
    "PERSON": "GIVEN_NAME",
    "LOCATION": "ADDRESS",
    "EMAIL_ADDRESS": "EMAIL",
    "EMAIL": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "AADHAAR",
    "US_BANK_NUMBER": "AADHAAR",
    "US_DRIVER_LICENSE": "PAN",
    "US_SSN": "PAN",
    "US_ITIN": "PAN",
    "US_PASSPORT": "PAN",
    "UK_NHS": "PHONE",
    "IBAN_CODE": "AADHAAR",
    "ID": "PAN",
    "MEDICAL_LICENSE": "PAN",
    # India-specific recognizers. These ship with Presidio but are NOT loaded by
    # load_predefined_recognizers(languages=["en"]) - they have to be registered
    # explicitly. Our first run omitted them, which made Presidio look far worse
    # on exactly the identifiers this project is about. Leaving them off would
    # have been the same weak-baseline error D16 and D24 exist to prevent.
    "IN_PAN": "PAN",
    "IN_AADHAAR": "AADHAAR",
    "IN_PASSPORT": "PAN",
    "IN_VOTER": "PAN",
    "IN_GSTIN": "PAN",
    "IN_VEHICLE_REGISTRATION": "PAN",
}
DROPPED = {"DATE_TIME", "URL", "NRP", "AGE", "ORGANIZATION", "IP_ADDRESS",
           "MAC_ADDRESS", "CRYPTO", "UUID"}


@lru_cache(maxsize=1)
def _engine(model: str):
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    import presidio_analyzer.predefined_recognizers as pr

    prov = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model}],
    })
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=["en"])
    for name in ("InPanRecognizer", "InAadhaarRecognizer", "InPassportRecognizer",
                 "InVoterRecognizer", "InGstinRecognizer",
                 "InVehicleRegistrationRecognizer"):
        cls = getattr(pr, name, None)
        if cls is not None:
            try:
                registry.add_recognizer(cls())
            except Exception:      # a recognizer that will not instantiate is skipped
                pass
    return AnalyzerEngine(nlp_engine=prov.create_engine(), registry=registry,
                          supported_languages=["en"])


class PresidioDetector:
    name = "presidio"
    emits = tuple(sorted(set(MAPPING.values())))

    def __init__(self, model: str = "en_core_web_md", threshold: float = 0.4):
        self.model = model
        self.threshold = threshold

    def detect(self, text: str) -> list[Detection]:
        eng = _engine(self.model)
        out: list[Detection] = []
        for r in eng.analyze(text=text, language="en"):
            if r.entity_type in DROPPED:
                continue
            leaf = MAPPING.get(r.entity_type)
            if leaf is None or r.score < self.threshold:
                continue
            out.append(Detection(
                pii=leaf,
                char_start=r.start,
                char_end=r.end,
                text=text[r.start:r.end],
                source=f"presidio:{r.entity_type}",
                score=float(r.score),
            ))
        return out
