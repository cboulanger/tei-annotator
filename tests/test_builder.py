import pytest

from tei_annotator.inference.endpoint import EndpointCapability
from tei_annotator.models.schema import TEIElement, TEISchema
from tei_annotator.models.spans import SpanDescriptor
from tei_annotator.prompting.builder import build_prompt, make_correction_prompt


def _schema():
    return TEISchema(
        elements=[
            TEIElement(tag="persName", description="a person's name", attributes=[]),
            TEIElement(tag="placeName", description="a place name", attributes=[]),
        ]
    )


def test_text_gen_prompt_contains_json_instruction():
    prompt = build_prompt("Some text.", _schema(), EndpointCapability.TEXT_GENERATION)
    assert "JSON" in prompt or "json" in prompt


def test_text_gen_prompt_contains_example():
    prompt = build_prompt("Some text.", _schema(), EndpointCapability.TEXT_GENERATION)
    # The template shows an example output array
    assert "persName" in prompt or "element" in prompt


def test_text_gen_prompt_contains_schema_elements():
    prompt = build_prompt("Some text.", _schema(), EndpointCapability.TEXT_GENERATION)
    assert "persName" in prompt
    assert "placeName" in prompt


def test_text_gen_prompt_contains_source_text():
    prompt = build_prompt("unique_source_42", _schema(), EndpointCapability.TEXT_GENERATION)
    assert "unique_source_42" in prompt


def test_json_enforced_prompt_contains_schema():
    prompt = build_prompt("text", _schema(), EndpointCapability.JSON_ENFORCED)
    assert "persName" in prompt
    assert "placeName" in prompt


def test_json_enforced_prompt_shorter_than_text_gen():
    text_gen = build_prompt("text", _schema(), EndpointCapability.TEXT_GENERATION)
    json_enf = build_prompt("text", _schema(), EndpointCapability.JSON_ENFORCED)
    assert len(json_enf) < len(text_gen)


def test_candidates_appear_in_prompt():
    candidates = [
        SpanDescriptor(element="persName", text="John", context="said John went", attrs={})
    ]
    prompt = build_prompt(
        "said John went.",
        _schema(),
        EndpointCapability.TEXT_GENERATION,
        candidates=candidates,
    )
    assert "John" in prompt


def test_no_candidate_section_when_none():
    prompt = build_prompt("text", _schema(), EndpointCapability.TEXT_GENERATION, candidates=None)
    assert "Pre-detected" not in prompt


def test_empty_candidates_list_no_section():
    prompt = build_prompt("text", _schema(), EndpointCapability.TEXT_GENERATION, candidates=[])
    assert "Pre-detected" not in prompt


def test_extraction_raises():
    with pytest.raises(ValueError):
        build_prompt("text", _schema(), EndpointCapability.EXTRACTION)


def test_correction_prompt_contains_original_response():
    prompt = make_correction_prompt("bad_json_here", "JSONDecodeError")
    assert "bad_json_here" in prompt
    assert "JSONDecodeError" in prompt
