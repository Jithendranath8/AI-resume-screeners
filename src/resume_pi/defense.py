"""Defenses: three content filters + channel separation (the fence).

These reconstruct the anchor paper's four-stage defense from its description:
  1. markup_strip        - remove structural hidden carriers
  2. unicode_normalize   - strip zero-width chars, fold homoglyphs to ASCII
  3. instruction_redact  - remove overt selection directives (lexical)
  4. channel separation  - screen(..., treat_as_data=True)  [the 'fence']
"""
from __future__ import annotations

import re

from .document import Document, strip_homoglyph, strip_zwsp

# Overt directive phrases the lexical redactor removes. Deliberately literal:
# it cannot see through zero-width spaces or homoglyphs on its own, and it does
# not understand oblique paraphrases.
_OVERT = [
    "ignore all previous instructions",
    "perfect fit",
    "rank them first",
    "strongly recommend advancing them",
    "recommend advancing",
]


def markup_strip(doc: Document) -> Document:
    """Drop structural hidden carriers: comments, metadata, hidden/off-canvas bodies.

    Does NOT remove styling-based concealment (white text, zero font size) or
    character-level tricks (zero-width, homoglyph) that live in ordinary body text.
    """
    kept = [
        s for s in doc.segments
        if s.channel == "body" and s.method not in ("hidden_element", "off_canvas")
    ]
    return Document(doc.doc_id, doc.base_id, kept, doc.injected,
                    doc.technique, doc.payload_family)


def unicode_normalize(text: str) -> str:
    """Strip zero-width characters and fold homoglyphs back to ASCII."""
    return strip_homoglyph(strip_zwsp(text))


def instruction_redact(text: str) -> str:
    """Remove overt selection directives (literal, case-insensitive)."""
    out = text
    for phrase in _OVERT:
        out = re.sub(re.escape(phrase), " [redacted] ", out, flags=re.IGNORECASE)
    return out
