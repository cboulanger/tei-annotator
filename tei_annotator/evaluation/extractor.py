"""
extractor.py — Extract EvaluationSpans from an annotated or gold-standard XML element.

A span is positioned by its character offsets in the element's *plain text*
(i.e. all tags stripped, just the text content concatenated in document order).
This makes gold and predicted spans directly comparable as long as both are
derived from the same source text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lxml import etree


def _strip_ns(tag: str) -> str:
    """Remove Clark-notation namespace from '{http://...}tag' → 'tag'."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


@dataclass
class EvaluationSpan:
    """A span extracted from XML, positioned in the element's plain text."""

    element: str            # tag name (namespace stripped)
    start: int              # char offset in stripped plain text (inclusive)
    end: int                # char offset in stripped plain text (exclusive)
    text: str               # the spanned text content
    attrs: dict[str, str] = field(default_factory=dict)

    @property
    def normalized_text(self) -> str:
        """Collapse internal whitespace for lenient comparison."""
        return " ".join(self.text.split())


def _extract_recursive(
    element: etree._Element,
    spans: list[EvaluationSpan],
    offset: int,
) -> int:
    """
    Walk *element*'s subtree depth-first, appending an EvaluationSpan for
    every descendant element to *spans*.

    Returns the offset *after* all of element's content (excluding its tail,
    since tail text belongs to the *parent*, not to this element).
    """
    current = offset

    # Text that belongs to this element, before its first child
    if element.text:
        current += len(element.text)

    for child in element:
        child_start = current
        # Recurse: sets current to the end of all content inside child
        current = _extract_recursive(child, spans, current)
        child_end = current

        child_text = "".join(child.itertext())
        spans.append(
            EvaluationSpan(
                element=_strip_ns(child.tag),
                start=child_start,
                end=child_end,
                text=child_text,
                attrs={_strip_ns(k): v for k, v in child.attrib.items()},
            )
        )

        # Tail text belongs to the parent, not the child
        if child.tail:
            current += len(child.tail)

    return current


def extract_spans(element: etree._Element) -> tuple[str, list[EvaluationSpan]]:
    """
    Extract plain text and EvaluationSpans from an lxml element.

    The plain text is ``"".join(element.itertext())``.  Every descendant
    element (at any depth) produces one EvaluationSpan whose ``start``/``end``
    are byte-exact offsets into that plain text.  Nested elements produce
    overlapping (container, child) span pairs — that is fine for matching.

    Returns
    -------
    (plain_text, spans)
        ``plain_text`` is the concatenated text content.
        ``spans`` is a flat list ordered depth-first (children before parents).
    """
    spans: list[EvaluationSpan] = []
    _extract_recursive(element, spans, offset=0)
    plain_text = "".join(element.itertext())
    return plain_text, spans


def spans_from_xml_string(xml_str: str) -> tuple[str, list[EvaluationSpan]]:
    """
    Parse an XML string with a single root element and extract spans.

    Convenience wrapper around :func:`extract_spans`.
    """
    root = etree.fromstring(xml_str.encode())
    return extract_spans(root)
