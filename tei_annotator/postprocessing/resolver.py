from __future__ import annotations

from ..models.spans import ResolvedSpan, SpanDescriptor

try:
    from rapidfuzz import fuzz as _fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


def _find_context(
    source: str,
    context: str,
    threshold: float,
) -> tuple[int, bool] | None:
    """
    Locate *context* in *source*.

    Returns (start_pos, is_fuzzy):
    - (pos, False) on exact match
    - (pos, True)  on fuzzy match with score >= threshold
    - None         if not found or below threshold
    """
    pos = source.find(context)
    if pos != -1:
        return pos, False

    if not _HAS_RAPIDFUZZ or not context:
        return None

    win = len(context)
    if win > len(source):
        return None

    best_score = 0.0
    best_pos = -1
    for i in range(len(source) - win + 1):
        score = _fuzz.ratio(context, source[i : i + win]) / 100.0
        if score > best_score:
            best_score = score
            best_pos = i

    if best_score >= threshold:
        return best_pos, True
    return None


def resolve_spans(
    source: str,
    spans: list[SpanDescriptor],
    fuzzy_threshold: float = 0.92,
) -> list[ResolvedSpan]:
    """
    Convert context-anchored SpanDescriptors to char-offset ResolvedSpans.

    Rejects spans whose text cannot be reliably located in *source*.
    Spans that required fuzzy context matching are flagged with fuzzy_match=True.
    """
    resolved: list[ResolvedSpan] = []

    for span in spans:
        result = _find_context(source, span.context, fuzzy_threshold)
        if result is None:
            continue  # context not found → reject

        ctx_start, context_is_fuzzy = result

        # Find span.text within the located context window
        window = source[ctx_start : ctx_start + len(span.context)]
        text_pos = window.find(span.text)
        if text_pos == -1:
            continue  # text not in context window → reject

        abs_start = ctx_start + text_pos
        abs_end = abs_start + len(span.text)

        # Verify verbatim match (should always hold after exact context find,
        # but important guard after fuzzy context find)
        if source[abs_start:abs_end] != span.text:
            continue

        resolved.append(
            ResolvedSpan(
                element=span.element,
                start=abs_start,
                end=abs_end,
                attrs=span.attrs.copy(),
                children=[],
                fuzzy_match=context_is_fuzzy,
            )
        )

    return resolved
