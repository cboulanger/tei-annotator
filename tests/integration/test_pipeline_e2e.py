"""
End-to-end integration tests: real GLiNER model + mocked call_fn.

Tests that only use mocked call_fn (gliner_model=None) are also here because
they exercise the full pipeline with non-trivial context resolution scenarios.

Run with: pytest -m integration
"""

from __future__ import annotations

import json
import re

import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_tags(xml: str) -> str:
    return re.sub(r"<[^>]+>", "", xml)


def _schema(*tags: tuple[str, str]):
    """Build a TEISchema from (tag, description) pairs."""
    from tei_annotator.models.schema import TEIAttribute, TEIElement, TEISchema

    elements = []
    for tag, desc in tags:
        if tag == "persName":
            elements.append(
                TEIElement(
                    tag="persName",
                    description=desc,
                    attributes=[
                        TEIAttribute(name="ref", description="URI reference"),
                        TEIAttribute(name="cert", description="certainty", allowed_values=["high", "low"]),
                    ],
                )
            )
        else:
            elements.append(TEIElement(tag=tag, description=desc, attributes=[]))
    return TEISchema(elements=elements)


def _endpoint(call_fn, capability="json_enforced"):
    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig

    cap = {
        "json_enforced": EndpointCapability.JSON_ENFORCED,
        "text_generation": EndpointCapability.TEXT_GENERATION,
    }[capability]
    return EndpointConfig(capability=cap, call_fn=call_fn)


def _annotate(text, schema, call_fn, capability="json_enforced", gliner_model=None, **kw):
    from tei_annotator.pipeline import annotate

    return annotate(
        text=text,
        schema=schema,
        endpoint=_endpoint(call_fn, capability),
        gliner_model=gliner_model,
        **kw,
    )


# ---------------------------------------------------------------------------
# 1. Exact context longer than span text
# ---------------------------------------------------------------------------


def test_context_longer_than_span_text():
    """Resolver must locate span.text inside a longer context window."""
    source = "The treaty was signed by Cardinal Richelieu in Paris."
    schema = _schema(("persName", "a person's name"), ("placeName", "a place name"))

    def call_fn(_):
        return json.dumps([
            {
                "element": "persName",
                "text": "Cardinal Richelieu",
                "context": "was signed by Cardinal Richelieu in Paris",
                "attrs": {},
            },
            {
                "element": "placeName",
                "text": "Paris",
                "context": "Cardinal Richelieu in Paris.",
                "attrs": {},
            },
        ])

    result = _annotate(source, schema, call_fn)
    assert "<persName>Cardinal Richelieu</persName>" in result.xml
    assert "<placeName>Paris</placeName>" in result.xml
    assert _strip_tags(result.xml) == source


# ---------------------------------------------------------------------------
# 2. Same span text appears twice — context disambiguates
# ---------------------------------------------------------------------------


def test_multiple_occurrences_disambiguated_by_context():
    """
    'John Smith' appears twice.  LLM returns two spans with distinct contexts
    pointing at each occurrence.  Both must be annotated at the correct offset.
    """
    source = "John Smith arrived early. Later, John Smith left."
    schema = _schema(("persName", "a person's name"))

    def call_fn(_):
        return json.dumps([
            {
                "element": "persName",
                "text": "John Smith",
                "context": "John Smith arrived early.",
                "attrs": {},
            },
            {
                "element": "persName",
                "text": "John Smith",
                "context": "Later, John Smith left.",
                "attrs": {},
            },
        ])

    result = _annotate(source, schema, call_fn)
    assert result.xml.count("<persName>") == 2
    assert result.xml.count("John Smith") == 2
    assert _strip_tags(result.xml) == source


# ---------------------------------------------------------------------------
# 3. Long text requiring chunking — global offset calculation
# ---------------------------------------------------------------------------


def test_long_text_entity_in_second_chunk():
    """
    Entity is far into a long text; its LLM context is relative to a later chunk.
    Offset must be shifted by chunk.start_offset to land at the correct global position.
    """
    # Build a ~2500-char text; entity sits well past the first 1500-char chunk
    filler = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 25  # ~1425 chars
    target_sentence = "Napoleon Bonaparte was exiled to Saint Helena."
    source = filler + target_sentence

    schema = _schema(
        ("persName", "a person's name"),
        ("placeName", "a place name"),
    )

    def call_fn(prompt):
        # The LLM sees either a filler chunk (returns []) or the chunk containing
        # the target sentence and returns the two spans.
        if "Napoleon" not in prompt:
            return "[]"
        return json.dumps([
            {
                "element": "persName",
                "text": "Napoleon Bonaparte",
                "context": "Napoleon Bonaparte was exiled to Saint Helena.",
                "attrs": {},
            },
            {
                "element": "placeName",
                "text": "Saint Helena",
                "context": "was exiled to Saint Helena.",
                "attrs": {},
            },
        ])

    result = _annotate(source, schema, call_fn, chunk_size=1500, chunk_overlap=200)

    assert "<persName>Napoleon Bonaparte</persName>" in result.xml
    assert "<placeName>Saint Helena</placeName>" in result.xml
    assert _strip_tags(result.xml) == source

    # Verify the annotated positions are truly within the target sentence
    napoleon_start = result.xml.index("Napoleon Bonaparte")
    assert napoleon_start > 1400, (
        f"Napoleon offset {napoleon_start} is too early — chunk offset was not applied"
    )


# ---------------------------------------------------------------------------
# 4. Nested spans resolved end-to-end
# ---------------------------------------------------------------------------


def test_nested_spans_end_to_end():
    """
    LLM emits an outer persName and inner forename / surname spans.
    Both are resolved separately and then nested by the injector.
    """
    source = "He met John Smith today."
    schema = _schema(
        ("persName", "a person's full name"),
        ("forename", "a forename"),
        ("surname", "a surname"),
    )

    def call_fn(_):
        return json.dumps([
            {"element": "persName", "text": "John Smith", "context": "met John Smith today.", "attrs": {}},
            {"element": "forename", "text": "John", "context": "met John Smith today.", "attrs": {}},
            {"element": "surname", "text": "Smith", "context": "John Smith today.", "attrs": {}},
        ])

    result = _annotate(source, schema, call_fn)

    assert "<persName>" in result.xml
    assert "<forename>" in result.xml
    assert "<surname>" in result.xml
    # forename and surname must be inside persName
    p_open = result.xml.index("<persName>")
    p_close = result.xml.index("</persName>")
    fn_open = result.xml.index("<forename>")
    sn_close = result.xml.index("</surname>")
    assert p_open < fn_open < sn_close < p_close
    assert _strip_tags(result.xml) == source


# ---------------------------------------------------------------------------
# 5. Pre-existing XML preserved after annotation
# ---------------------------------------------------------------------------


def test_preexisting_xml_preserved():
    """
    Source already has markup (<note> tags).  After annotation the original
    markup must still be present alongside the new TEI annotations.
    """
    source = "He met <note>allegedly</note> John Smith yesterday."
    schema = _schema(("persName", "a person's name"))

    def call_fn(_):
        # The LLM sees stripped plain text: "He met allegedly John Smith yesterday."
        return json.dumps([
            {
                "element": "persName",
                "text": "John Smith",
                "context": "allegedly John Smith yesterday.",
                "attrs": {},
            }
        ])

    result = _annotate(source, schema, call_fn)

    assert "<note>" in result.xml
    assert "</note>" in result.xml
    assert "<persName>John Smith</persName>" in result.xml
    # Plain text must be unchanged
    assert _strip_tags(result.xml) == _strip_tags(source)


# ---------------------------------------------------------------------------
# 6. Attributes preserved end-to-end
# ---------------------------------------------------------------------------


def test_attributes_preserved_end_to_end():
    """Attribute values returned by the LLM must appear verbatim in the output tag."""
    source = "The emperor Napoleon was defeated at Waterloo."
    schema = _schema(("persName", "a person's name"))

    def call_fn(_):
        return json.dumps([
            {
                "element": "persName",
                "text": "Napoleon",
                "context": "emperor Napoleon was defeated",
                "attrs": {"ref": "http://viaf.org/viaf/106964661", "cert": "high"},
            }
        ])

    result = _annotate(source, schema, call_fn)
    assert 'ref="http://viaf.org/viaf/106964661"' in result.xml
    assert 'cert="high"' in result.xml
    assert _strip_tags(result.xml) == source


# ---------------------------------------------------------------------------
# 7. Hallucinated context → span silently rejected
# ---------------------------------------------------------------------------


def test_hallucinated_context_span_rejected():
    """
    LLM returns a plausible-looking but non-existent context.
    The resolver must reject the span; the source text is returned unmodified.
    """
    source = "Marie Curie discovered polonium."
    schema = _schema(("persName", "a person's name"))

    def call_fn(_):
        return json.dumps([
            {
                "element": "persName",
                "text": "Marie Curie",
                "context": "Dr. Marie Curie discovered polonium",  # "Dr. " not in source
                "attrs": {},
            }
        ])

    result = _annotate(source, schema, call_fn)
    assert "<persName>" not in result.xml
    assert result.xml == source


# ---------------------------------------------------------------------------
# 8. Fuzzy context match → span annotated and flagged
# ---------------------------------------------------------------------------


def test_fuzzy_context_match_flags_span():
    """
    A context with a single-character typo should still resolve via fuzzy
    matching (score > 0.92) and be included with fuzzy_match=True.
    """
    source = "Galileo Galilei observed the moons of Jupiter."
    schema = _schema(("persName", "a person's name"))

    def call_fn(_):
        return json.dumps([
            {
                "element": "persName",
                "text": "Galileo Galilei",
                # One character different from the source — should trigger fuzzy
                "context": "Galileo Galilei observd the moons of Jupiter.",
                "attrs": {},
            }
        ])

    result = _annotate(source, schema, call_fn)
    # The span should still be annotated
    assert "<persName>Galileo Galilei</persName>" in result.xml
    # And flagged as fuzzy
    assert len(result.fuzzy_spans) == 1
    assert result.fuzzy_spans[0].element == "persName"
    assert _strip_tags(result.xml) == source


# ---------------------------------------------------------------------------
# 9. Source text never modified (plain-text invariant)
# ---------------------------------------------------------------------------


def test_plain_text_invariant_with_multiple_entities():
    """Stripping all tags from the output must yield exactly the input text."""
    source = (
        "Leonardo da Vinci was born in Vinci, Tuscany, "
        "and later worked in Milan and Florence."
    )
    schema = _schema(
        ("persName", "a person's name"),
        ("placeName", "a place name"),
    )

    def call_fn(_):
        return json.dumps([
            {"element": "persName", "text": "Leonardo da Vinci",
             "context": "Leonardo da Vinci was born in Vinci", "attrs": {}},
            {"element": "placeName", "text": "Vinci",
             "context": "born in Vinci, Tuscany", "attrs": {}},
            {"element": "placeName", "text": "Tuscany",
             "context": "Vinci, Tuscany, and later", "attrs": {}},
            {"element": "placeName", "text": "Milan",
             "context": "later worked in Milan and Florence", "attrs": {}},
            {"element": "placeName", "text": "Florence",
             "context": "in Milan and Florence.", "attrs": {}},
        ])

    result = _annotate(source, schema, call_fn)
    assert _strip_tags(result.xml) == source
    assert result.xml.count("<placeName>") == 4
    assert "<persName>Leonardo da Vinci</persName>" in result.xml


# ---------------------------------------------------------------------------
# 10. Real GLiNER model (requires HuggingFace download)
# ---------------------------------------------------------------------------


def test_pipeline_with_real_gliner():
    """Full pipeline: real GLiNER pre-detection + mocked LLM call_fn."""
    schema = _schema(("persName", "a person's name"))

    def mock_llm(_: str) -> str:
        return json.dumps([
            {
                "element": "persName",
                "text": "Albert Einstein",
                "context": "Albert Einstein was born",
                "attrs": {},
            }
        ])

    result = _annotate(
        "Albert Einstein was born in Ulm in 1879.",
        schema,
        mock_llm,
        gliner_model="numind/NuNER_Zero",
    )
    assert "persName" in result.xml
    assert "Albert Einstein" in result.xml
    assert result.xml.count("Albert Einstein") == 1
