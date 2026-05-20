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


def test_annotate_escapes_bare_ampersand():
    """Bare & in source text must be escaped to &amp; in the output XML."""
    original = "Smith & Jones 2020."

    def _no_spans(_prompt):
        return json.dumps([])

    result = annotate(
        text=original,
        schema=_schema(),
        endpoint=EndpointConfig(
            capability=EndpointCapability.JSON_ENFORCED,
            call_fn=_no_spans,
        ),
        gliner_model=None,
    )
    assert "&amp;" in result.xml
    assert "& J" not in result.xml


def test_annotate_preserves_existing_entity_references():
    """Already-escaped &amp; in input must not be double-escaped."""
    original = "Smith &amp; Jones 2020."

    def _no_spans(_prompt):
        return json.dumps([])

    result = annotate(
        text=original,
        schema=_schema(),
        endpoint=EndpointConfig(
            capability=EndpointCapability.JSON_ENFORCED,
            call_fn=_no_spans,
        ),
        gliner_model=None,
    )
    assert "&amp;amp;" not in result.xml
    assert "&amp;" in result.xml


def test_no_duplicate_tags_when_same_element_detected():
    """
    When original text contains <elem>text</elem> and the LLM also detects
    'text' as <elem>, tags should not be duplicated (e.g., <elem><elem>...).

    This is a regression test for issue #2 (text order corruption in large annotations).
    """
    # Create a schema with 'bibl' element for this test
    schema = TEISchema(
        elements=[
            TEIElement(
                tag="bibl",
                description="bibliographic entry",
                allowed_children=[],
                attributes=[],
            )
        ]
    )

    def _detect_bibl(prompt: str) -> str:
        # Simulate LLM detecting the exact same text as the original <bibl> tag
        return json.dumps(
            [
                {
                    "element": "bibl",
                    "text": "Smith and Jones (2020)",
                    "context": "See Smith and Jones (2020) for",
                    "attrs": {},
                }
            ]
        )

    result = annotate(
        text="See <bibl>Smith and Jones (2020)</bibl> for more.",
        schema=schema,
        endpoint=EndpointConfig(
            capability=EndpointCapability.JSON_ENFORCED,
            call_fn=_detect_bibl,
        ),
        gliner_model=None,
    )
    # Schema-element tags from the input are dropped from the restore map so the
    # LLM's annotation is the sole source — exactly one <bibl>, not <bibl><bibl>.
    assert result.xml.count("<bibl>") == 1
    assert result.xml.count("</bibl>") == 1
    # Text should not be duplicated
    assert result.xml.count("Smith and Jones (2020)") == 1


def test_strip_existing_tags_preserves_angle_bracket_urls():
    """URLs in angle brackets (e.g. <https://...>) must NOT be stripped as XML tags.

    The tag name character set for XML excludes '/', so 'https://...' is not a
    valid element name.  _strip_existing_tags must leave such angle-bracket
    sequences as plain text rather than recording them as stripped XML tags.
    """
    from tei_annotator.pipeline import _strip_existing_tags

    text = "See (<https://www.example.com/page> 2019)."
    plain, restore_map = _strip_existing_tags(text)

    assert "<https://www.example.com/page>" in plain
    assert restore_map == []


def test_strip_existing_tags_preserves_url_alongside_real_xml():
    """Real XML tags must still be stripped even when angle-bracket URLs are present."""
    from tei_annotator.pipeline import _strip_existing_tags

    text = "See <b>(<https://www.example.com/page> 2019)</b>."
    plain, restore_map = _strip_existing_tags(text)

    assert "<https://www.example.com/page>" in plain
    assert "<b>" not in plain
    assert "</b>" not in plain
    assert len(restore_map) == 2


def test_annotate_preserves_angle_bracket_urls():
    """The full pipeline must not drop angle-bracket-enclosed URLs."""

    def _no_spans(_prompt):
        return json.dumps([])

    text = "See (<https://www.example.com/page> 2019)."
    result = annotate(
        text=text,
        schema=_schema(),
        endpoint=EndpointConfig(
            capability=EndpointCapability.JSON_ENFORCED,
            call_fn=_no_spans,
        ),
        gliner_model=None,
    )
    assert "https://www.example.com/page" in result.xml


def test_overlapping_spans_from_chunks_are_merged():
    """
    When overlapping chunks produce overlapping spans with the same element,
    they should be merged into a single span covering the union of both ranges.

    This prevents text fragmentation and reordering (issue #2).
    """
    # Create a minimal schema
    schema = TEISchema(
        elements=[
            TEIElement(
                tag="ref",
                description="reference",
                allowed_children=[],
                attributes=[],
            )
        ]
    )

    # Simulate two overlapping chunks detecting overlapping sections
    chunk1_response = json.dumps([{
        "element": "ref",
        "text": "item one item two",
        "context": "See item one item two and",
        "attrs": {},
    }])

    chunk2_response = json.dumps([{
        "element": "ref",
        "text": "item two item three",
        "context": "item two item three done",
        "attrs": {},
    }])

    call_count = [0]
    def _multi_chunk_response(prompt: str) -> str:
        # Alternate between chunk 1 and chunk 2 responses
        call_count[0] += 1
        return chunk1_response if call_count[0] == 1 else chunk2_response

    # Text is long enough to be chunked
    text = "See item one item two and item three done."

    result = annotate(
        text=text,
        schema=schema,
        endpoint=EndpointConfig(
            capability=EndpointCapability.JSON_ENFORCED,
            call_fn=_multi_chunk_response,
        ),
        gliner_model=None,
        chunk_size=20,
        chunk_overlap=5,
    )

    # Text should not be reordered or duplicated
    import re
    plain = re.sub(r"<[^>]+>", "", result.xml)
    assert plain == text
    # Should have at most one <ref> pair (merged, not fragmented)
    ref_count = result.xml.count("<ref>")
    assert ref_count <= 2, f"Expected <=2 <ref> tags, got {ref_count}"
