from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpanDescriptor:
    """Flat span emitted by the LLM or GLiNER — always context-anchored, never nested."""
    element: str
    text: str
    context: str        # must contain text as a substring
    attrs: dict[str, str] = field(default_factory=dict)
    confidence: float | None = None  # passed through from GLiNER


@dataclass
class ResolvedSpan:
    """Span resolved to absolute char offsets in the source text."""
    element: str
    start: int
    end: int
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[ResolvedSpan] = field(default_factory=list)
    fuzzy_match: bool = False   # flagged for human review
