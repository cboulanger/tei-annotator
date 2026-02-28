import pytest

from tei_annotator.models.spans import ResolvedSpan
from tei_annotator.postprocessing.injector import _build_nesting_tree, inject_xml


def _span(element, start, end, attrs=None):
    return ResolvedSpan(element=element, start=start, end=end, attrs=attrs or {})


def test_no_spans_returns_source():
    assert inject_xml("hello world", []) == "hello world"


def test_single_span():
    source = "He said John Smith yesterday."
    # "John Smith" = [8:18]
    span = _span("persName", 8, 18)
    result = inject_xml(source, [span])
    assert result == "He said <persName>John Smith</persName> yesterday."


def test_two_non_overlapping_spans():
    source = "John met Mary."
    # "John" = [0:4], "Mary" = [9:13]
    spans = [_span("persName", 0, 4), _span("persName", 9, 13)]
    result = inject_xml(source, spans)
    assert result == "<persName>John</persName> met <persName>Mary</persName>."


def test_nested_spans():
    # "Dr. Smith" = outer, "Dr." = inner
    source = "He met Dr. Smith today."
    # "Dr. Smith" = [7:16], "Dr." = [7:10]
    spans = [_span("persName", 7, 16), _span("roleName", 7, 10)]
    result = inject_xml(source, spans)
    assert "<persName>" in result
    assert "<roleName>" in result
    # roleName must appear inside persName
    assert result.index("<roleName>") > result.index("<persName>")
    assert result.index("</roleName>") < result.index("</persName>")
    # Text is split by the inner tag; check exact output structure
    assert result == "He met <persName><roleName>Dr.</roleName> Smith</persName> today."


def test_attrs_rendered_in_tag():
    source = "Visit Paris."
    span = _span("placeName", 6, 11, {"ref": "http://example.com/paris"})
    result = inject_xml(source, [span])
    assert 'ref="http://example.com/paris"' in result
    assert "<placeName" in result
    assert "Paris" in result


def test_span_at_start_of_text():
    source = "John went home."
    span = _span("persName", 0, 4)
    result = inject_xml(source, [span])
    assert result.startswith("<persName>John</persName>")


def test_span_covering_entire_text():
    source = "John Smith"
    span = _span("persName", 0, 10)
    result = inject_xml(source, [span])
    assert result == "<persName>John Smith</persName>"


def test_span_at_end_of_text():
    source = "He visited Paris"
    span = _span("placeName", 11, 16)
    result = inject_xml(source, [span])
    assert result.endswith("<placeName>Paris</placeName>")


def test_overlapping_spans_warns_and_skips():
    source = "Hello World"
    # Partial overlap: [0,7] and [5,11]
    spans = [_span("a", 0, 7), _span("b", 5, 11)]
    with pytest.warns(UserWarning, match="Overlapping"):
        result = inject_xml(source, spans)
    # Only the first span should be present
    assert "<a>" in result
    assert "<b>" not in result


def test_build_nesting_tree_simple():
    outer = _span("persName", 0, 20)
    inner = _span("roleName", 0, 5)
    roots = _build_nesting_tree([outer, inner])
    assert len(roots) == 1
    assert roots[0].element == "persName"
    assert len(roots[0].children) == 1
    assert roots[0].children[0].element == "roleName"


def test_build_nesting_tree_siblings():
    a = _span("a", 0, 5)
    b = _span("b", 6, 10)
    roots = _build_nesting_tree([a, b])
    assert len(roots) == 2
    assert all(len(r.children) == 0 for r in roots)
