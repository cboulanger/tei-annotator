from __future__ import annotations

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
            exact_pos = source.find(span.context)
            log.debug(
                "resolver REJECT <%s>: context not found in source. "
                "exact_find=%d, context_len=%d, context=%r",
                span.element, exact_pos, len(span.context), span.context[:120],
            )
            continue  # context not found → reject

        ctx_start, context_is_fuzzy = result

        # Find span.text within the located context window
        window = source[ctx_start : ctx_start + len(span.context)]
        text_pos = window.find(span.text)
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
                    resolved.append(
                        ResolvedSpan(
                            element=span.element,
                            start=abs_start,
                            end=true_end,
                            attrs=span.attrs.copy(),
                            children=[],
                            fuzzy_match=True,
                        )
                    )
                    continue
            log.debug(
                "resolver REJECT <%s>: text not in context window and fuzzy text search failed. "
                "ctx_start=%d, context_len=%d, text_len=%d, text=%r",
                span.element, ctx_start, len(span.context), len(span.text), span.text[:120],
            )
            continue  # text not in context window → reject

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
