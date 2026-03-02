"""
Unit tests for the evaluation module.

All tests are fully mocked — no real LLM calls, no GLiNER model downloads.
Run with:  uv run pytest tests/test_evaluation.py
"""

from __future__ import annotations

import json

import pytest
from lxml import etree

from tei_annotator.evaluation.extractor import EvaluationSpan, extract_spans
from tei_annotator.evaluation.metrics import (
    ElementMetrics,
    EvaluationResult,
    MatchMode,
    SpanMatch,
    aggregate,
    compute_metrics,
    match_spans,
)
from tei_annotator.evaluation.evaluator import evaluate_element
from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
from tei_annotator.models.schema import TEIElement, TEISchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(xml_str: str) -> etree._Element:
    return etree.fromstring(xml_str.encode())


def _schema(*tags: str) -> TEISchema:
    return TEISchema(
        elements=[TEIElement(tag=t, description=t, allowed_children=[], attributes=[]) for t in tags]
    )


def _mock_endpoint(spans: list[dict]) -> EndpointConfig:
    """Return an EndpointConfig whose call_fn always returns the given spans as JSON."""
    payload = json.dumps(spans)
    return EndpointConfig(
        capability=EndpointCapability.JSON_ENFORCED,
        call_fn=lambda _prompt: payload,
    )


# ---------------------------------------------------------------------------
# extractor — extract_spans
# ---------------------------------------------------------------------------

class TestExtractSpans:
    def test_flat_two_elements(self):
        root = _parse("<bibl><author>Smith</author>, <date>2020</date>.</bibl>")
        plain, spans = extract_spans(root)
        assert plain == "Smith, 2020."
        author = next(s for s in spans if s.element == "author")
        date = next(s for s in spans if s.element == "date")
        assert (author.start, author.end) == (0, 5)
        assert author.text == "Smith"
        assert (date.start, date.end) == (7, 11)
        assert date.text == "2020"

    def test_plain_text_equals_itertext(self):
        root = _parse("<bibl><author>A</author> and <author>B</author>.</bibl>")
        plain, _ = extract_spans(root)
        assert plain == "A and B."

    def test_nested_elements(self):
        root = _parse(
            "<bibl><author><forename>John</forename> <surname>Smith</surname></author></bibl>"
        )
        plain, spans = extract_spans(root)
        assert plain == "John Smith"
        forename = next(s for s in spans if s.element == "forename")
        surname = next(s for s in spans if s.element == "surname")
        author = next(s for s in spans if s.element == "author")
        assert (forename.start, forename.end) == (0, 4)
        assert (surname.start, surname.end) == (5, 10)
        # author contains both children
        assert author.start == 0
        assert author.end == 10
        assert author.text == "John Smith"

    def test_attributes_preserved(self):
        root = _parse('<bibl><title level="a">My Title</title></bibl>')
        _, spans = extract_spans(root)
        title = spans[0]
        assert title.element == "title"
        assert title.attrs == {"level": "a"}

    def test_namespace_stripped(self):
        root = _parse(
            '<bibl xmlns="http://www.tei-c.org/ns/1.0">'
            "<author>X</author></bibl>"
        )
        _, spans = extract_spans(root)
        assert spans[0].element == "author"

    def test_no_children(self):
        root = _parse("<bibl>plain text only</bibl>")
        plain, spans = extract_spans(root)
        assert plain == "plain text only"
        assert spans == []

    def test_element_with_text_and_tail(self):
        # bibl.text = "See ", author.text = "Jones", author.tail = " for details."
        root = _parse("<bibl>See <author>Jones</author> for details.</bibl>")
        plain, spans = extract_spans(root)
        assert plain == "See Jones for details."
        author = spans[0]
        assert (author.start, author.end) == (4, 9)
        assert author.text == "Jones"

    def test_normalized_text_property(self):
        span = EvaluationSpan("author", 0, 10, "Smith  Jr.")
        assert span.normalized_text == "Smith Jr."


# ---------------------------------------------------------------------------
# metrics — match_spans
# ---------------------------------------------------------------------------

class TestMatchSpans:
    def _spans(self, *triples):
        return [EvaluationSpan(elem, s, e, text) for elem, s, e, text in triples]

    def test_exact_perfect_match(self):
        gold = self._spans(("author", 0, 5, "Smith"))
        pred = self._spans(("author", 0, 5, "Smith"))
        matched, ug, up = match_spans(gold, pred, mode=MatchMode.EXACT)
        assert len(matched) == 1 and len(ug) == 0 and len(up) == 0

    def test_exact_wrong_offset(self):
        gold = self._spans(("author", 0, 5, "Smith"))
        pred = self._spans(("author", 1, 6, "Smith"))
        matched, ug, up = match_spans(gold, pred, mode=MatchMode.EXACT)
        assert len(matched) == 0 and len(ug) == 1 and len(up) == 1

    def test_exact_wrong_element(self):
        gold = self._spans(("author", 0, 5, "Smith"))
        pred = self._spans(("editor", 0, 5, "Smith"))
        matched, ug, up = match_spans(gold, pred, mode=MatchMode.EXACT)
        assert len(matched) == 0

    def test_text_mode_matches_despite_offset_difference(self):
        gold = self._spans(("author", 0, 10, "Smith Jr."))
        pred = self._spans(("author", 0, 9, "Smith Jr."))
        matched, ug, up = match_spans(gold, pred, mode=MatchMode.TEXT)
        assert len(matched) == 1

    def test_text_mode_normalises_whitespace(self):
        gold = self._spans(("author", 0, 10, "Smith  Jr"))
        pred = self._spans(("author", 0, 9, "Smith Jr"))
        matched, ug, up = match_spans(gold, pred, mode=MatchMode.TEXT)
        assert len(matched) == 1

    def test_text_mode_different_text_no_match(self):
        gold = self._spans(("author", 0, 5, "Smith"))
        pred = self._spans(("author", 0, 5, "Jones"))
        matched, ug, up = match_spans(gold, pred, mode=MatchMode.TEXT)
        assert len(matched) == 0

    def test_overlap_mode_above_threshold(self):
        # gold [0,10], pred [5,15] → IoU = 5/15 ≈ 0.33 < default 0.5
        gold = self._spans(("author", 0, 10, "foo"))
        pred = self._spans(("author", 5, 15, "foo"))
        matched, _, _ = match_spans(gold, pred, mode=MatchMode.OVERLAP, overlap_threshold=0.3)
        assert len(matched) == 1

    def test_overlap_mode_below_threshold(self):
        gold = self._spans(("author", 0, 10, "foo"))
        pred = self._spans(("author", 5, 15, "foo"))
        matched, _, _ = match_spans(gold, pred, mode=MatchMode.OVERLAP, overlap_threshold=0.5)
        assert len(matched) == 0

    def test_greedy_each_span_matched_once(self):
        # Two gold, one pred that matches both equally — only one match expected
        gold = self._spans(("date", 0, 4, "2020"), ("date", 10, 14, "2020"))
        pred = self._spans(("date", 0, 4, "2020"))
        matched, ug, up = match_spans(gold, pred, mode=MatchMode.TEXT)
        assert len(matched) == 1
        assert len(ug) == 1  # second gold unmatched
        assert len(up) == 0

    def test_empty_pred(self):
        gold = self._spans(("author", 0, 5, "Smith"))
        matched, ug, up = match_spans(gold, [], mode=MatchMode.TEXT)
        assert matched == [] and len(ug) == 1 and up == []

    def test_empty_gold(self):
        pred = self._spans(("author", 0, 5, "Smith"))
        matched, ug, up = match_spans([], pred, mode=MatchMode.TEXT)
        assert matched == [] and ug == [] and len(up) == 1

    def test_both_empty(self):
        matched, ug, up = match_spans([], [], mode=MatchMode.TEXT)
        assert matched == [] and ug == [] and up == []


# ---------------------------------------------------------------------------
# metrics — compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _s(self, elem, text):
        return EvaluationSpan(elem, 0, len(text), text)

    def test_perfect_prediction(self):
        gold = [self._s("author", "Smith"), self._s("date", "2020")]
        pred = [self._s("author", "Smith"), self._s("date", "2020")]
        r = compute_metrics(gold, pred)
        assert r.micro_precision == 1.0
        assert r.micro_recall == 1.0
        assert r.micro_f1 == 1.0

    def test_all_wrong_element(self):
        gold = [self._s("author", "Smith")]
        pred = [self._s("date", "Smith")]
        r = compute_metrics(gold, pred)
        assert r.micro_precision == 0.0
        assert r.micro_recall == 0.0
        assert r.micro_f1 == 0.0

    def test_partial_recall(self):
        gold = [self._s("author", "Smith"), self._s("date", "2020")]
        pred = [self._s("author", "Smith")]
        r = compute_metrics(gold, pred)
        assert r.micro_precision == 1.0
        assert r.micro_recall == pytest.approx(0.5)
        assert r.micro_f1 == pytest.approx(2 / 3)

    def test_partial_precision(self):
        gold = [self._s("author", "Smith")]
        pred = [self._s("author", "Smith"), self._s("date", "2020")]
        r = compute_metrics(gold, pred)
        assert r.micro_precision == pytest.approx(0.5)
        assert r.micro_recall == 1.0

    def test_per_element_breakdown(self):
        gold = [self._s("author", "Smith"), self._s("date", "2020")]
        pred = [self._s("author", "Smith")]  # date missing
        r = compute_metrics(gold, pred)
        assert r.per_element["author"].true_positives == 1
        assert r.per_element["author"].false_negatives == 0
        assert r.per_element["date"].true_positives == 0
        assert r.per_element["date"].false_negatives == 1

    def test_macro_vs_micro_differ_on_imbalanced(self):
        # author: TP=1, FN=0 → recall=1.0
        # date:   TP=0, FN=2 → recall=0.0
        # micro recall = 1/3  ≈ 0.33
        # macro recall = (1.0 + 0.0) / 2 = 0.5
        gold = [
            self._s("author", "Smith"),
            self._s("date", "2020"),
            self._s("date", "2021"),
        ]
        pred = [self._s("author", "Smith")]
        r = compute_metrics(gold, pred)
        assert r.micro_recall == pytest.approx(1 / 3)
        assert r.macro_recall == pytest.approx(0.5)

    def test_empty_gold_and_pred(self):
        r = compute_metrics([], [])
        assert r.micro_f1 == 0.0
        assert r.per_element == {}

    def test_report_returns_string(self):
        gold = [self._s("author", "Smith")]
        pred = [self._s("author", "Smith")]
        r = compute_metrics(gold, pred)
        report = r.report()
        assert "author" in report
        assert "F1=" in report


# ---------------------------------------------------------------------------
# metrics — aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def _result(self, tp, fp, fn, elem="author") -> EvaluationResult:
        return EvaluationResult(
            matched=[SpanMatch(
                gold=EvaluationSpan(elem, 0, 1, "x"),
                pred=EvaluationSpan(elem, 0, 1, "x"),
            )] * tp,
            unmatched_gold=[EvaluationSpan(elem, 0, 1, "x")] * fn,
            unmatched_pred=[EvaluationSpan(elem, 0, 1, "x")] * fp,
            per_element={
                elem: ElementMetrics(
                    element=elem,
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                )
            },
        )

    def test_aggregate_sums_counts(self):
        r1 = self._result(tp=2, fp=1, fn=0)
        r2 = self._result(tp=1, fp=0, fn=1)
        agg = aggregate([r1, r2])
        m = agg.per_element["author"]
        assert m.true_positives == 3
        assert m.false_positives == 1
        assert m.false_negatives == 1

    def test_aggregate_empty(self):
        agg = aggregate([])
        assert agg.per_element == {}
        assert agg.micro_f1 == 0.0

    def test_aggregate_concatenates_lists(self):
        r1 = self._result(tp=1, fp=0, fn=0)
        r2 = self._result(tp=0, fp=1, fn=0)
        agg = aggregate([r1, r2])
        assert len(agg.matched) == 1
        assert len(agg.unmatched_pred) == 1


# ---------------------------------------------------------------------------
# evaluator — evaluate_element (mocked endpoint)
# ---------------------------------------------------------------------------

class TestEvaluateElement:
    def _schema(self, *tags):
        return _schema(*tags)

    def test_perfect_annotation(self):
        """Mock returns exactly the gold spans → F1=1."""
        gold_xml = "<bibl><author>Smith</author>, <date>2020</date>.</bibl>"
        root = _parse(gold_xml)
        schema = self._schema("author", "date")
        endpoint = _mock_endpoint([
            {"element": "author", "text": "Smith", "context": "Smith, 2020.", "attrs": {}},
            {"element": "date", "text": "2020", "context": "Smith, 2020.", "attrs": {}},
        ])
        result = evaluate_element(root, schema, endpoint, gliner_model=None)
        assert result.micro_f1 == pytest.approx(1.0)
        assert result.micro_precision == 1.0
        assert result.micro_recall == 1.0

    def test_missing_span_reduces_recall(self):
        gold_xml = "<bibl><author>Smith</author>, <date>2020</date>.</bibl>"
        root = _parse(gold_xml)
        schema = self._schema("author", "date")
        # Annotator only finds author, misses date
        endpoint = _mock_endpoint([
            {"element": "author", "text": "Smith", "context": "Smith, 2020.", "attrs": {}},
        ])
        result = evaluate_element(root, schema, endpoint, gliner_model=None)
        assert result.micro_precision == 1.0
        assert result.micro_recall == pytest.approx(0.5)

    def test_spurious_span_reduces_precision(self):
        gold_xml = "<bibl><author>Smith</author>.</bibl>"
        root = _parse(gold_xml)
        schema = self._schema("author", "date")
        # Annotator invents a date that isn't in gold
        endpoint = _mock_endpoint([
            {"element": "author", "text": "Smith", "context": "Smith.", "attrs": {}},
            {"element": "date", "text": "Smith", "context": "Smith.", "attrs": {}},
        ])
        result = evaluate_element(root, schema, endpoint, gliner_model=None)
        assert result.micro_recall == 1.0
        assert result.micro_precision == pytest.approx(0.5)

    def test_empty_annotation(self):
        gold_xml = "<bibl><author>Smith</author>.</bibl>"
        root = _parse(gold_xml)
        schema = self._schema("author")
        endpoint = _mock_endpoint([])
        result = evaluate_element(root, schema, endpoint, gliner_model=None)
        assert result.micro_precision == 0.0
        assert result.micro_recall == 0.0
        assert result.micro_f1 == 0.0

    def test_wrong_element_type(self):
        gold_xml = "<bibl><author>Smith</author>.</bibl>"
        root = _parse(gold_xml)
        schema = self._schema("author", "editor")
        # Annotator labels "Smith" as editor instead of author
        endpoint = _mock_endpoint([
            {"element": "editor", "text": "Smith", "context": "Smith.", "attrs": {}},
        ])
        result = evaluate_element(root, schema, endpoint, gliner_model=None)
        assert result.micro_f1 == 0.0

    def test_attributes_not_required_for_text_match(self):
        """With TEXT match mode, attribute differences don't affect matching."""
        gold_xml = '<bibl><title level="a">My Title</title>.</bibl>'
        root = _parse(gold_xml)
        schema = self._schema("title")
        # Annotator returns same text but wrong/missing attribute
        endpoint = _mock_endpoint([
            {"element": "title", "text": "My Title", "context": "My Title.", "attrs": {}},
        ])
        result = evaluate_element(root, schema, endpoint, gliner_model=None, match_mode=MatchMode.TEXT)
        assert result.micro_f1 == pytest.approx(1.0)

    def test_exact_match_mode(self):
        """EXACT mode requires identical character offsets."""
        gold_xml = "<bibl><author>Smith</author>.</bibl>"
        root = _parse(gold_xml)
        schema = self._schema("author")
        # plain text = "Smith." → author is at [0,5]
        endpoint = _mock_endpoint([
            {"element": "author", "text": "Smith", "context": "Smith.", "attrs": {}},
        ])
        result = evaluate_element(root, schema, endpoint, gliner_model=None, match_mode=MatchMode.EXACT)
        # Offsets align (same plain text) → should match
        assert result.micro_f1 == pytest.approx(1.0)

    def test_plain_text_element(self):
        """An element with no child tags → no gold spans, annotation may or may not add spans."""
        gold_xml = "<bibl>Plain text without any markup.</bibl>"
        root = _parse(gold_xml)
        schema = self._schema("author")
        endpoint = _mock_endpoint([])
        result = evaluate_element(root, schema, endpoint, gliner_model=None)
        # No gold spans, no pred spans → vacuously perfect
        assert result.micro_tp == 0
        assert result.micro_fp == 0
        assert result.micro_fn == 0

    def test_ampersand_in_text_is_escaped(self):
        """Bare & in annotated output must be escaped to &amp; so lxml can parse it."""
        # lxml decodes &amp; → & when reading the gold file, and the annotator
        # passes that raw & through into its XML output.  _escape_nonschema_brackets
        # must escape it before we wrap the fragment in a synthetic root element.
        gold_xml = "<bibl><author>A &amp; B</author>.</bibl>"
        root = _parse(gold_xml)
        schema = self._schema("author")
        # plain text = "A & B." — the annotator sees the literal ampersand
        endpoint = _mock_endpoint([
            {"element": "author", "text": "A & B", "context": "A & B.", "attrs": {}},
        ])
        result = evaluate_element(root, schema, endpoint, gliner_model=None)
        # Should parse successfully and match the gold author span
        assert result.micro_f1 == pytest.approx(1.0)
