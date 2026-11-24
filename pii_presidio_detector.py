"""
presidio_detector.py

Industry-standard PII detection using Microsoft Presidio + spaCy NER.

Same interface as detector.py (the regex version), so they are easy
to compare and swap:

    detect(text)          -> list[PiiMatch]
    contains_pii(text)    -> bool
    mask(text)            -> (masked_text, mapping)   reversible: [PERSON_1]
    unmask(text, mapping) -> original text restored
    redact(text)          -> masked_text              one-way: <PERSON>

What Presidio adds over regex:
    - PERSON names
      detected with an NER model (spaCy) - regex cannot do this
    - Confidence score for every match, so you can tune strictness
    - Context awareness: "call me on ..." raises the score of a phone number
    - Built-in recognizers for EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD
      (with checksum), IBAN_CODE, IP_ADDRESS, URL, DATE_TIME and more

Install:
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_lg

Run the demo (from the project root):
    python -m app.pii.presidio_detector
"""

from __future__ import annotations

from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import PhoneRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


# Entity types we care about for an AI agent. Presidio supports more
# (e.g. LOCATION, DATE_TIME, URL, NRP), but those cause many false positives in
# normal chat text, so we leave them out by default.
DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
]

# Countries for phone number detection (ISO 3166 codes).
# Presidio's default list uses "UK", but the correct code is "GB", so UK
# numbers in national format were missed. We replace it with a wider list.
# Numbers written with a "+country code" are detected for any country.
PHONE_REGIONS = ["GB", "US", "CA", "IN", "AU", "IE", "DE", "FR", "ES", "IT", "NL", "BR", "IL"]


@dataclass
class PiiMatch:
    type: str
    value: str
    start: int
    end: int
    score: float  # 0.0 - 1.0, how confident Presidio is


class PresidioPiiDetector:
    """
    Usage:
        detector = PresidioPiiDetector()          # slow: loads the NER model
        safe_text, mapping = detector.mask(text)  # fast: reuse the instance
        original = detector.unmask(llm_reply, mapping)

    Create ONE instance at app startup and reuse it - loading the
    spaCy model takes a few seconds and a lot of memory.
    """

    def __init__(
        self,
        entities: list[str] | None = None,
        score_threshold: float = 0.4,
        spacy_model: str = "en_core_web_lg",
        language: str = "en",
    ):
        self.entities = entities or DEFAULT_ENTITIES
        self.score_threshold = score_threshold
        self.language = language

        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": spacy_model}],
        }).create_engine()

        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language])

        # Swap the default phone recognizer for one with the right regions
        self.analyzer.registry.remove_recognizer("PhoneRecognizer")
        self.analyzer.registry.add_recognizer(
            PhoneRecognizer(supported_regions=PHONE_REGIONS, supported_language=language)
        )

        self.anonymizer = AnonymizerEngine()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _analyze(self, text: str) -> list[RecognizerResult]:
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities,
            score_threshold=self.score_threshold,
        )
        return self._remove_overlaps(results)

    @staticmethod
    def _remove_overlaps(results: list[RecognizerResult]) -> list[RecognizerResult]:
        """
        Different recognizers can flag the same text (e.g. a phone number
        inside a card number). Keep the highest-scoring match, and the
        longest one if scores are equal.
        """
        ranked = sorted(results, key=lambda r: (-r.score, -(r.end - r.start)))
        kept: list[RecognizerResult] = []
        for r in ranked:
            if all(r.end <= k.start or r.start >= k.end for k in kept):
                kept.append(r)
        return sorted(kept, key=lambda r: r.start)

    def detect(self, text: str) -> list[PiiMatch]:
        return [
            PiiMatch(r.entity_type, text[r.start:r.end], r.start, r.end, round(r.score, 2))
            for r in self._analyze(text)
        ]

    def contains_pii(self, text: str) -> bool:
        """Quick check - use this to BLOCK a request instead of masking it."""
        return bool(self._analyze(text))

    # ------------------------------------------------------------------
    # Reversible masking (for agents that must restore values later)
    # ------------------------------------------------------------------

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Replace PII with numbered placeholders like [PERSON_1].
        Returns (masked_text, mapping). Keep the mapping on YOUR server -
        never send it to the LLM.
        """
        mapping: dict[str, str] = {}
        value_to_key: dict[str, str] = {}
        counters: dict[str, int] = {}

        parts: list[str] = []
        pos = 0
        for r in self._analyze(text):
            value = text[r.start:r.end]
            if value not in value_to_key:  # same value -> same placeholder
                counters[r.entity_type] = counters.get(r.entity_type, 0) + 1
                key = f"[{r.entity_type}_{counters[r.entity_type]}]"
                value_to_key[value] = key
                mapping[key] = value
            parts.append(text[pos:r.start])
            parts.append(value_to_key[value])
            pos = r.end
        parts.append(text[pos:])

        return "".join(parts), mapping

    @staticmethod
    def unmask(text: str, mapping: dict[str, str]) -> str:
        """Put the real values back into the LLM's reply."""
        for key, value in mapping.items():
            text = text.replace(key, value)
        return text

    # ------------------------------------------------------------------
    # One-way redaction (uses presidio-anonymizer)
    # ------------------------------------------------------------------

    def redact(self, text: str) -> str:
        """
        Permanently replace PII with its type, e.g. <PERSON>, <EMAIL_ADDRESS>.
        Use this for logs, analytics or storing chats - anywhere you never
        need the original value back.
        """
        result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=self._analyze(text),
            operators={"DEFAULT": OperatorConfig("replace", {"new_value": None})},
        )
        return result.text


# --------------------------------------------------------------------------
# Demo:  python -m app.pii.presidio_detector
# --------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Hi, I'm Sarah Johnson. Email me at sarah.j@example.com "
        "or call +44 7911 123456. My card is 4111 1111 1111 1111 and my IBAN is "
        "GB82 WEST 1234 5698 7654 32. Order #12345 came from IP 192.168.0.1. "
        "Please ask Sarah Johnson before shipping."
    )

    print("Loading spaCy model...")
    detector = PresidioPiiDetector()

    print("\nORIGINAL:\n", sample)

    print("\nDETECTED:")
    for m in detector.detect(sample):
        print(f"  {m.type:<14} {m.score:<5} {m.value}")

    masked, mapping = detector.mask(sample)
    print("\nSENT TO LLM (reversible mask):\n", masked)

    print("\nMAPPING (stays on your server):")
    for k, v in mapping.items():
        print(f"  {k:<18} -> {v}")

    reply = "Thanks [PERSON_1], I've emailed the receipt to [EMAIL_ADDRESS_1]."
    print("\nLLM REPLY UNMASKED:\n", detector.unmask(reply, mapping))

    print("\nFOR LOGS (one-way redact):\n", detector.redact(sample))