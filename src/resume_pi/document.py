"""Document model: a resume whose text renders two ways.

A naive extraction pipeline reads *everything* (``raw_extract``); a human sees
only ``visible_render``. The gap between the two is exactly what hidden-text
injection exploits, and what a visibility-diff detector looks for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

ZWSP = "​"  # zero-width space

# Latin -> Cyrillic look-alikes used by the homoglyph concealment technique.
_HOMO = {
    "a": "а", "c": "с", "e": "е", "i": "і",
    "o": "о", "p": "р", "x": "х", "y": "у",
}
_HOMO_REV = {v: k for k, v in _HOMO.items()}


def to_homoglyph(text: str) -> str:
    """Replace Latin letters with confusable Cyrillic ones (visible to humans)."""
    return "".join(_HOMO.get(ch, ch) for ch in text)


def strip_homoglyph(text: str) -> str:
    """Fold confusable characters back to ASCII (what a normaliser would do)."""
    return "".join(_HOMO_REV.get(ch, ch) for ch in text)


def has_homoglyph(text: str) -> bool:
    return any(ch in _HOMO_REV for ch in text)


def strip_zwsp(text: str) -> str:
    return text.replace(ZWSP, "")


def has_zwsp(text: str) -> bool:
    return ZWSP in text


@dataclass
class Segment:
    """One piece of a document."""
    text: str
    visible: bool = True      # would a human see this on the rendered page?
    channel: str = "body"     # body | comment | metadata
    kind: str = "content"     # content | payload
    method: str = "none"      # concealment method (payload segments only)


@dataclass
class Document:
    doc_id: str
    base_id: str
    segments: List[Segment] = field(default_factory=list)
    injected: bool = False
    technique: str = "clean"
    payload_family: str = "none"

    def raw_extract(self) -> str:
        """What a naive text extractor pulls: every segment, every channel."""
        return "\n".join(s.text for s in self.segments)

    def visible_render(self) -> str:
        """What a human reviewer sees: visible body segments, zero-width removed."""
        return "\n".join(
            strip_zwsp(s.text)
            for s in self.segments
            if s.visible and s.channel == "body"
        )

    def has_hidden_text(self) -> bool:
        """Ground-truth helper (for dataset manifest only, not used by detectors)."""
        return any(s.kind == "payload" and not (s.visible and s.channel == "body")
                   for s in self.segments)
