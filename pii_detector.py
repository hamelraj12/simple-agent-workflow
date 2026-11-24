from __future__ import annotations

import re
from dataclasses import dataclass

@dataclass
class PiiMatch:
    type: str
    value: str
    start: int
    end: int


# --------------------------------------------------------------------------
# Validators (reduce false positives)
# --------------------------------------------------------------------------

def _luhn_valid(number: str) -> bool:
    """Check a card number with the Luhn algorithm."""
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _phone_valid(number: str) -> bool:
    """
    A phone number should have 10-15 digits.
    A plain run of digits (no '+', spaces, dashes or brackets) only counts
    if it starts with 0 (national format, e.g. 0612345678), so order or
    reference numbers like 1234567890123 are not flagged.
    """
    digits = re.sub(r"\D", "", number)
    if not 10 <= len(digits) <= 15:
        return False
    if number.isdigit() and not number.startswith("0"):
        return False
    return True


def _ip_valid(ip: str) -> bool:
    return all(0 <= int(part) <= 255 for part in ip.split("."))

PATTERNS: list[tuple[str, re.Pattern, callable | None]] = [
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        None,
    ),
    (
        "CARD_NUMBER",
        re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
        _luhn_valid,
    ),
    (
        "PHONE",
        re.compile(
            r"(?<![\w.])"
            r"(?:\+\d{1,3}[\s.-]?)?"          # optional country code: +44, +1, +91
            r"\(?\d{1,5}\)?"                  # first group, optional brackets: (020)
            r"(?:[\s.-]?\d{2,5}){1,4}"        # 1-4 more digit groups, any grouping
            r"(?![\w.])"                      # digit count checked by _phone_valid
        ),
        _phone_valid,
    ),
    (
        "IP_ADDRESS",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        _ip_valid,
    ),
]

class PiiDetector:

    def detect(self, text: str) -> list[PiiMatch]:
            """Return all non-overlapping PII matches, in order of position."""
            found: list[PiiMatch] = []
            taken: list[tuple[int, int]] = []

            for pii_type, pattern, validator in PATTERNS:
                for m in pattern.finditer(text):
                    start, end = m.start(), m.end()
                    value = m.group().rstrip(" -.")
                    end = start + len(value)

                    if validator and not validator(value):
                        continue
                    if any(start < t_end and end > t_start for t_start, t_end in taken):
                        continue  # overlaps a higher-priority match

                    found.append(PiiMatch(pii_type, value, start, end))
                    taken.append((start, end))

            return sorted(found, key=lambda x: x.start)

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Replace PII with placeholders.
        Returns (masked_text, mapping). Keep the mapping on YOUR server -
        never send it to the LLM.
        """
        matches = self.detect(text)
        mapping: dict[str, str] = {}
        value_to_key: dict[str, str] = {}
        counters: dict[str, int] = {}

        parts: list[str] = []
        pos = 0
        for m in matches:
            if m.value not in value_to_key:  # same value -> same placeholder
                counters[m.type] = counters.get(m.type, 0) + 1
                key = f"[{m.type}_{counters[m.type]}]"
                value_to_key[m.value] = key
                mapping[key] = m.value
            parts.append(text[pos:m.start])
            parts.append(value_to_key[m.value])
            pos = m.end
        parts.append(text[pos:])

        return "".join(parts), mapping

    @staticmethod
    def unmask(text: str, mapping: dict[str, str]) -> str:
        """Put the real values back into the LLM's reply."""
        for key, value in mapping.items():
            text = text.replace(key, value)
        return text

    def contains_pii(self, text: str) -> bool:
        """Quick check - use this to BLOCK a request instead of masking it."""
        return bool(self.detect(text))

if __name__ == "__main__":
    sample = (
        "Hi, I'm a customer. Email me at john.smith@example.com or call "
        "+44 7700 900123 (or my US office +1 415 555 2671). "
        "My card is 4111 1111 1111 1111. "
        "I live at 221B Baker Street, London. "
        "Order #12345 was placed on 2026-11-20 from IP 192.168.0.1. "
        "Please reply to john.smith@example.com."
    )

    detector = PiiDetector()

    print("ORIGINAL:\n", sample, "\n")

    print("DETECTED:")
    for m in detector.detect(sample):
        print(f"  {m.type:<13} {m.value}")

    masked, mapping = detector.mask(sample)
    print("\nSENT TO LLM:\n", masked, "\n")

    print("MAPPING (stays on your server):")
    for k, v in mapping.items():
        print(f"  {k:<16} -> {v}")

    fake_llm_reply = "I'll send the receipt to [EMAIL_1] and ship to [ADDRESS_1]."
    print("\nLLM REPLY UNMASKED:\n", detector.unmask(fake_llm_reply, mapping))