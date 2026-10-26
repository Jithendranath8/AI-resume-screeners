"""Hidden-prompt detector (the detection side of the defense).

Flags a document if ANY of four principled signals fire:
  (a) visibility diff   - text the extractor sees but a human does not
  (b) zero-width chars  - invisible characters used to smuggle text
  (c) homoglyph anomaly - look-alike (non-Latin) characters inside Latin text
  (d) overt directive   - an explicit selection instruction in *visible* text

By construction the only blind spot is an *oblique* instruction left in plain,
fully visible, otherwise-clean text — which is precisely the anchor paper's
"concealment dilemma": to evade detection the payload must stay visible, at
which point it is subject to ordinary human review.
"""
from __future__ import annotations

import re

from .document import Document, has_homoglyph, has_zwsp, strip_zwsp

_OVERT = [
    "ignore all previous", "perfect fit", "rank them first",
    "strongly recommend", "recommend advancing", "top candidate",
    "select this candidate",
]


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z]{3,}", strip_zwsp(text).lower()))


def _visibility_diff(doc: Document) -> bool:
    """Any extractable tokens that do not appear in the rendered (visible) text."""
    return bool(_tokens(doc.raw_extract()) - _tokens(doc.visible_render()))


def _overt_directive(text: str) -> bool:
    low = strip_zwsp(text).lower()
    return any(p in low for p in _OVERT)


def detect(doc: Document) -> bool:
    raw = doc.raw_extract()
    if _visibility_diff(doc):
        return True
    if has_zwsp(raw):
        return True
    if has_homoglyph(raw):
        return True
    if _overt_directive(doc.visible_render()):
        return True
    return False
