from __future__ import annotations

import warnings

from ..models.spans import ResolvedSpan


def _build_nesting_tree(flat_spans: list[ResolvedSpan]) -> list[ResolvedSpan]:
    """
    Populate ResolvedSpan.children based on offset containment and return root spans.

    Spans are sorted so that outer (longer) spans are processed before inner ones.
    Overlapping (non-nesting) spans are skipped with a warning.
    """
    # Sort: start asc, then end desc so outer spans come before inner at same start
    spans = sorted(flat_spans, key=lambda s: (s.start, -(s.end - s.start)))

    # Clear any children left from a previous call
    for s in spans:
        s.children = []

    roots: list[ResolvedSpan] = []
    stack: list[ResolvedSpan] = []

    for span in spans:
        rejected = False

        # Pop stack entries that are fully before (or incompatibly overlap) this span
        while stack:
            top = stack[-1]
            if top.start <= span.start and span.end <= top.end:
                break  # top properly contains span → it's the parent
            elif span.start >= top.end:
                stack.pop()  # span comes after top → pop and continue
            else:
                # Partial overlap (neither contained nor after) → reject span
                warnings.warn(
                    f"Overlapping spans [{top.start},{top.end}] and "
                    f"[{span.start},{span.end}] cannot be nested; "
                    f"skipping <{span.element}> span.",
                    stacklevel=3,
                )
                rejected = True
                break

        if rejected:
            continue

        if stack:
            stack[-1].children.append(span)
        else:
            roots.append(span)

        stack.append(span)

    return roots


def _inject_recursive(
    text: str,
    spans: list[ResolvedSpan],
    offset: int,
) -> str:
    """
    Insert XML open/close tags for *spans* into *text*.

    *offset* is the absolute position of text[0] in the original source, used
    to translate span.start/end (absolute) to positions within *text*.
    """
    if not spans:
        return text

    result: list[str] = []
    cursor = 0  # relative position within text

    for span in sorted(spans, key=lambda s: s.start):
        rel_start = span.start - offset
        rel_end = span.end - offset

        # Text before this span
        result.append(text[cursor:rel_start])

        # Build tag strings
        attrs_str = " ".join(f'{k}="{v}"' for k, v in span.attrs.items())
        open_tag = f"<{span.element}" + (f" {attrs_str}" if attrs_str else "") + ">"
        close_tag = f"</{span.element}>"

        # Recursively inject children inside this span's content
        inner = text[rel_start:rel_end]
        if span.children:
            inner = _inject_recursive(inner, span.children, offset=span.start)

        result.append(open_tag)
        result.append(inner)
        result.append(close_tag)

        cursor = rel_end

    result.append(text[cursor:])
    return "".join(result)


def inject_xml(source: str, spans: list[ResolvedSpan]) -> str:
    """
    Insert XML tags into *source* at the positions defined by *spans*.

    Nesting is inferred from offset containment via _build_nesting_tree.
    """
    if not spans:
        return source
    root_spans = _build_nesting_tree(spans)
    return _inject_recursive(source, root_spans, offset=0)
