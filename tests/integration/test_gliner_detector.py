"""
Integration tests for GLiNER detection.

These tests download a real HuggingFace model (~400 MB) on first run.
Run with: pytest -m integration
"""

import pytest

pytestmark = pytest.mark.integration


def test_gliner_detects_person_name():
    from tei_annotator.detection.gliner_detector import detect_spans
    from tei_annotator.models.schema import TEIElement, TEISchema

    schema = TEISchema(
        elements=[
            TEIElement(tag="persName", description="a person's name", attributes=[]),
        ]
    )
    text = "Albert Einstein was born in Ulm in 1879."
    spans = detect_spans(text, schema, model_id="numind/NuNER_Zero")
    assert any(s.element == "persName" and "Einstein" in s.text for s in spans), (
        f"Expected a persName span containing 'Einstein'; got: {spans}"
    )


def test_gliner_confidence_scores_present():
    from tei_annotator.detection.gliner_detector import detect_spans
    from tei_annotator.models.schema import TEIElement, TEISchema

    schema = TEISchema(
        elements=[
            TEIElement(tag="persName", description="a person's name", attributes=[]),
        ]
    )
    text = "Marie Curie discovered polonium."
    spans = detect_spans(text, schema, model_id="numind/NuNER_Zero")
    for span in spans:
        if span.confidence is not None:
            assert 0.0 <= span.confidence <= 1.0


def test_gliner_context_contains_text():
    from tei_annotator.detection.gliner_detector import detect_spans
    from tei_annotator.models.schema import TEIElement, TEISchema

    schema = TEISchema(
        elements=[
            TEIElement(tag="persName", description="a person's name", attributes=[]),
        ]
    )
    text = "Charles Darwin published On the Origin of Species."
    spans = detect_spans(text, schema, model_id="numind/NuNER_Zero")
    for span in spans:
        assert span.text in span.context, (
            f"span.text {span.text!r} not found in context {span.context!r}"
        )
