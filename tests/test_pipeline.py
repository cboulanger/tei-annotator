import json

import pytest

from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
from tei_annotator.models.schema import TEIElement, TEISchema
from tei_annotator.pipeline import annotate


def _schema():
    return TEISchema(
        elements=[
            TEIElement(
                tag="persName",
                description="a person's name",
                allowed_children=[],
                attributes=[],
            )
        ]
    )


def _mock_call_fn(prompt: str) -> str:
    return json.dumps(
        [
            {
                "element": "persName",
                "text": "John Smith",
                "context": "said John Smith yesterday",
                "attrs": {},
            }
        ]
    )


def test_annotate_smoke():
    result = annotate(
        text="He said John Smith yesterday.",
        schema=_schema(),
        endpoint=EndpointConfig(
            capability=EndpointCapability.JSON_ENFORCED,
            call_fn=_mock_call_fn,
        ),
        gliner_model=None,
    )
    assert "persName" in result.xml
    assert "John Smith" in result.xml
    assert result.xml.count("John Smith") == 1  # text not duplicated


def test_annotate_empty_response():
    result = annotate(
        text="No entities here.",
        schema=_schema(),
        endpoint=EndpointConfig(
            capability=EndpointCapability.JSON_ENFORCED,
            call_fn=lambda _: "[]",
        ),
        gliner_model=None,
    )
    assert result.xml == "No entities here."
    assert result.fuzzy_spans == []


def test_annotate_preserves_existing_xml():
    # Pre-existing <b> tag must survive
    def call_fn(prompt: str) -> str:
        return json.dumps(
            [
                {
                    "element": "persName",
                    "text": "John Smith",
                    "context": "said John Smith yesterday",
                    "attrs": {},
                }
            ]
        )

    result = annotate(
        text="He said <b>John Smith</b> yesterday.",
        schema=_schema(),
        endpoint=EndpointConfig(
            capability=EndpointCapability.JSON_ENFORCED, call_fn=call_fn
        ),
        gliner_model=None,
    )
    assert "<b>" in result.xml
    assert "John Smith" in result.xml


def test_annotate_fuzzy_spans_surfaced():
    """Spans flagged as fuzzy appear in AnnotationResult.fuzzy_spans."""
    # We cannot force a fuzzy match easily without mocking internals,
    # so we just verify the field exists and is a list.
    result = annotate(
        text="He said John Smith yesterday.",
        schema=_schema(),
        endpoint=EndpointConfig(
            capability=EndpointCapability.JSON_ENFORCED,
            call_fn=_mock_call_fn,
        ),
        gliner_model=None,
    )
    assert isinstance(result.fuzzy_spans, list)


def test_annotate_text_generation_endpoint():
    """TEXT_GENERATION capability path (with retry logic enabled) works end-to-end."""
    result = annotate(
        text="He said John Smith yesterday.",
        schema=_schema(),
        endpoint=EndpointConfig(
            capability=EndpointCapability.TEXT_GENERATION,
            call_fn=_mock_call_fn,
        ),
        gliner_model=None,
    )
    assert "persName" in result.xml


def test_annotate_no_text_modification():
    """The original text characters must all appear in the output (no hallucination)."""
    original = "He said John Smith yesterday."
    result = annotate(
        text=original,
        schema=_schema(),
        endpoint=EndpointConfig(
            capability=EndpointCapability.JSON_ENFORCED,
            call_fn=_mock_call_fn,
        ),
        gliner_model=None,
    )
    # Strip all tags from output; plain text should equal original
    import re

    plain = re.sub(r"<[^>]+>", "", result.xml)
    assert plain == original
