from __future__ import annotations

import html as _html
import logging

from ..models.spans import ResolvedSpan, SpanDescriptor

log = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as _fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


def _align_end(source: str, source_start: int, text: str, max_extra: int = 50) -> int:
    """
    Walk *source* from *source_start* advancing in parallel with *text*, allowing
    up to *max_extra* extra whitespace characters in *source* (e.g. spaces left
    behind by stripped <lb/> tags).  Returns the source position just after the
    last matched character of *text*.
    """
    s = source_start
    t = 0
    extra = 0
    while t < len(text) and s < len(source):
        if source[s] == text[t]:
            s += 1
            t += 1
        elif source[s] in " \t\n\r" and extra < max_extra:
            s += 1  # extra whitespace in source — skip it
            extra += 1
        else:
            break
    return s


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


def _find_in_window(window: str, text: str) -> int:
    """
    Return the start index of *text* in *window*, using a preference order:

    1. Fully isolated — preceded AND followed by whitespace (or at window edges).
    2. Left-boundary only — preceded by whitespace but not followed.
    3. Any first occurrence (fallback).

    This prevents resolving a short label token (e.g. "3") to a position
    inside a larger token (e.g. the "3" in "321." which is preceded by a
    space but followed by "2") when a truly standalone occurrence exists later.
    Returns -1 when *text* is not in *window*.
    """
    pos = 0
    first_match = -1
    left_boundary = -1
    while True:
        pos = window.find(text, pos)
        if pos == -1:
            break
        if first_match == -1:
            first_match = pos
        end = pos + len(text)
        preceded = pos == 0 or window[pos - 1] in " \t\n\r"
        followed = end >= len(window) or window[end] in " \t\n\r"
        if preceded and followed:
            return pos  # fully isolated — best possible match
        if preceded and left_boundary == -1:
            left_boundary = pos
        pos += 1
    if left_boundary != -1:
        return left_boundary
    return first_match


def _decode_entities(span: SpanDescriptor) -> SpanDescriptor | None:
    """
    Return a copy of *span* with HTML/XML entities decoded (e.g. &amp; → &),
    or None if there are no entities to decode.

    Used as a fallback when the LLM re-encodes characters that an XML parser
    already decoded: itertext() gives bare & but the LLM emits &amp;.
    """
    new_text = _html.unescape(span.text)
    new_context = _html.unescape(span.context)
    if new_text == span.text and new_context == span.context:
        return None
    return SpanDescriptor(
        element=span.element,
        text=new_text,
        context=new_context,
        attrs=span.attrs,
    )


def _resolve_one(
    source: str,
    span: SpanDescriptor,
    fuzzy_threshold: float,
) -> ResolvedSpan | None:
    """Try to resolve a single SpanDescriptor against *source*. Returns None on failure."""
    result = _find_context(source, span.context, fuzzy_threshold)
    if result is None:
        exact_pos = source.find(span.context)
        log.debug(
            "resolver REJECT <%s>: context not found in source. "
            "exact_find=%d, context_len=%d, context=%r",
            span.element, exact_pos, len(span.context), span.context[:120],
        )
        return None

    ctx_start, context_is_fuzzy = result

    # Find span.text within the located context window.
    # When multiple occurrences exist, prefer one that starts at a word
    # boundary (preceded by whitespace or start of window) to avoid
    # landing on the same short token embedded inside a URL or word.
    window = source[ctx_start : ctx_start + len(span.context)]
    text_pos = _find_in_window(window, span.text)
    if text_pos == -1:
        # The fuzzy context match may have landed slightly off, or the LLM
        # text has minor whitespace differences from the source (e.g. \n vs
        # \n<space> around stripped <lb/> tags).  Try locating the text
        # with its own fuzzy search, then take the verbatim source slice.
        text_result = _find_context(source, span.text, fuzzy_threshold)
        if text_result is not None:
            text_fuzzy_start, _ = text_result
            abs_start = text_fuzzy_start
            abs_end = text_fuzzy_start + len(span.text)
            # Verify the source slice is a reasonable match (not completely wrong)
            if _HAS_RAPIDFUZZ:
                score = _fuzz.ratio(span.text, source[abs_start:abs_end]) / 100.0
            else:
                score = 1.0 if source[abs_start:abs_end] == span.text else 0.0
            if score >= fuzzy_threshold:
                # Walk char-by-char to find the true end, tolerating extra
                # whitespace in source that the LLM text omitted.
                true_end = _align_end(source, abs_start, span.text)
                log.debug(
                    "resolver FALLBACK <%s>: fuzzy text search at %d (score=%.2f) "
                    "end adjusted %d→%d",
                    span.element, abs_start, score, abs_end, true_end,
                )
                return ResolvedSpan(
                    element=span.element,
                    start=abs_start,
                    end=true_end,
                    attrs=span.attrs.copy(),
                    children=[],
                    fuzzy_match=True,
                )
        log.debug(
            "resolver REJECT <%s>: text not in context window and fuzzy text search failed. "
            "ctx_start=%d, context_len=%d, text_len=%d, text=%r",
            span.element, ctx_start, len(span.context), len(span.text), span.text[:120],
        )
        return None

    abs_start = ctx_start + text_pos
    abs_end = abs_start + len(span.text)

    # Verify verbatim match (should always hold after exact context find,
    # but important guard after fuzzy context find)
    if source[abs_start:abs_end] != span.text:
        log.debug(
            "resolver REJECT <%s>: verbatim verify failed at [%d:%d]. "
            "source=%r span.text=%r",
            span.element, abs_start, abs_end,
            source[abs_start:abs_end][:120], span.text[:120],
        )
        return None

    return ResolvedSpan(
        element=span.element,
        start=abs_start,
        end=abs_end,
        attrs=span.attrs.copy(),
        children=[],
        fuzzy_match=context_is_fuzzy,
    )


def resolve_spans(
    source: str,
    spans: list[SpanDescriptor],
    fuzzy_threshold: float = 0.92,
) -> list[ResolvedSpan]:
    """
    Convert context-anchored SpanDescriptors to char-offset ResolvedSpans.

    Rejects spans whose text cannot be reliably located in *source*.
    Spans that required fuzzy context matching are flagged with fuzzy_match=True.

    Handles entity-encoding mismatches: when an XML parser decodes &amp; → &
    but the LLM re-encodes & → &amp; in its output, a decoded fallback is tried
    automatically after the primary resolution attempt fails.
    """
    resolved: list[ResolvedSpan] = []

    for span in spans:
        rs = _resolve_one(source, span, fuzzy_threshold)
        if rs is None:
            # Fallback: try with HTML entities decoded (handles the case where
            # lxml decoded &amp; → & in the source but the LLM re-encoded & →
            # &amp; in its span text/context).
            decoded = _decode_entities(span)
            if decoded is not None:
                rs = _resolve_one(source, decoded, fuzzy_threshold)
                if rs is not None:
                    log.debug(
                        "resolver ENTITY-FALLBACK <%s>: resolved after decoding HTML entities",
                        span.element,
                    )
        if rs is not None:
            resolved.append(rs)

    return resolved
