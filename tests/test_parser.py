import json

import pytest

from tei_annotator.postprocessing.parser import _strip_fences, parse_response

VALID_JSON = json.dumps(
    [{"element": "persName", "text": "John Smith", "context": "said John Smith", "attrs": {}}]
)


# ---- _strip_fences ----------------------------------------------------------


def test_strip_fences_json_lang():
    fenced = f"```json\n{VALID_JSON}\n```"
    assert _strip_fences(fenced) == VALID_JSON


def test_strip_fences_no_lang():
    fenced = f"```\n{VALID_JSON}\n```"
    assert _strip_fences(fenced) == VALID_JSON


def test_strip_fences_no_fences():
    assert _strip_fences(VALID_JSON) == VALID_JSON


def test_strip_fences_with_preamble():
    text = f"Here is the JSON:\n```json\n{VALID_JSON}\n```"
    assert _strip_fences(text) == VALID_JSON


# ---- parse_response ---------------------------------------------------------


def test_valid_json_parsed_directly():
    spans = parse_response(VALID_JSON)
    assert len(spans) == 1
    assert spans[0].element == "persName"
    assert spans[0].text == "John Smith"


def test_markdown_fenced_json_parsed():
    spans = parse_response(f"```json\n{VALID_JSON}\n```")
    assert len(spans) == 1


def test_invalid_json_no_retry_raises():
    with pytest.raises(ValueError):
        parse_response("not json at all")


def test_retry_triggered_on_first_failure():
    call_count = [0]

    def retry_fn(prompt: str) -> str:
        call_count[0] += 1
        return VALID_JSON

    def correction_fn(bad: str, err: str) -> str:
        return f"fix: {bad}"

    spans = parse_response("bad json", call_fn=retry_fn, make_correction_prompt=correction_fn)
    assert call_count[0] == 1
    assert len(spans) == 1


def test_retry_still_invalid_raises():
    def retry_fn(prompt: str) -> str:
        return "still bad"

    def correction_fn(bad: str, err: str) -> str:
        return "fix it"

    with pytest.raises(ValueError):
        parse_response("bad", call_fn=retry_fn, make_correction_prompt=correction_fn)


def test_missing_fields_items_skipped():
    raw = json.dumps(
        [
            {"element": "persName"},  # missing text and context → skip
            {"element": "persName", "text": "John", "context": "John went"},  # valid
        ]
    )
    spans = parse_response(raw)
    assert len(spans) == 1
    assert spans[0].text == "John"


def test_non_list_response_raises():
    with pytest.raises(ValueError):
        parse_response(json.dumps({"element": "persName"}))


def test_attrs_defaults_to_empty_dict():
    raw = json.dumps([{"element": "persName", "text": "x", "context": "x"}])
    spans = parse_response(raw)
    assert spans[0].attrs == {}
