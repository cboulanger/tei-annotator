import pytest

from tei_annotator.models.spans import SpanDescriptor
from tei_annotator.postprocessing.resolver import resolve_spans

SOURCE = "He said John Smith yesterday, and John Smith agreed."


def _span(element, text, context, attrs=None):
    return SpanDescriptor(element=element, text=text, context=context, attrs=attrs or {})


def test_exact_context_match():
    span = _span("persName", "John Smith", "said John Smith yesterday")
    resolved = resolve_spans(SOURCE, [span])
    assert len(resolved) == 1
    rs = resolved[0]
    assert rs.start == SOURCE.index("John Smith")
    assert rs.end == rs.start + len("John Smith")
    assert not rs.fuzzy_match


def test_context_not_found_rejected():
    span = _span("persName", "John Smith", "this context does not exist xyz987")
    assert resolve_spans(SOURCE, [span]) == []


def test_text_not_in_context_window_rejected():
    span = _span("persName", "Jane Doe", "said John Smith yesterday")
    assert resolve_spans(SOURCE, [span]) == []


def test_source_slice_verified():
    span = _span("persName", "John Smith", "said John Smith yesterday")
    resolved = resolve_spans(SOURCE, [span])
    assert len(resolved) == 1
    rs = resolved[0]
    assert SOURCE[rs.start : rs.end] == "John Smith"


def test_attrs_preserved():
    span = _span("persName", "John Smith", "said John Smith yesterday", {"ref": "#js"})
    resolved = resolve_spans(SOURCE, [span])
    assert len(resolved) == 1
    assert resolved[0].attrs == {"ref": "#js"}


def test_multiple_spans_resolved():
    spans = [
        _span("persName", "John Smith", "He said John Smith yesterday"),
        _span("persName", "John Smith", "and John Smith agreed"),
    ]
    resolved = resolve_spans(SOURCE, spans)
    assert len(resolved) == 2
    assert resolved[0].start != resolved[1].start


def test_empty_span_list():
    assert resolve_spans(SOURCE, []) == []


def test_children_start_empty():
    span = _span("persName", "John Smith", "said John Smith yesterday")
    resolved = resolve_spans(SOURCE, [span])
    assert resolved[0].children == []
