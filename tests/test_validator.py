import pytest

from tei_annotator.models.schema import TEIAttribute, TEIElement, TEISchema
from tei_annotator.models.spans import ResolvedSpan
from tei_annotator.postprocessing.validator import validate_output, validate_spans

SOURCE = "He met John Smith."


def _schema():
    return TEISchema(
        elements=[
            TEIElement(
                tag="persName",
                description="a person's name",
                attributes=[
                    TEIAttribute(name="ref", description="reference URI"),
                    TEIAttribute(
                        name="cert",
                        description="certainty",
                        allowed_values=["high", "low"],
                    ),
                ],
            )
        ]
    )


def _span(element, start, end, attrs=None):
    return ResolvedSpan(element=element, start=start, end=end, attrs=attrs or {})


# SOURCE: "He met John Smith."
# positions: H=0 e=1 ' '=2 m=3 e=4 t=5 ' '=6 J=7 o=8 h=9 n=10 ' '=11 S=12 m=13 i=14 t=15 h=16 .=17
# "John Smith" => [7:17]


def test_valid_span_passes():
    result = validate_spans([_span("persName", 7, 17)], _schema(), SOURCE)
    assert len(result) == 1


def test_unknown_element_rejected():
    result = validate_spans([_span("orgName", 7, 17)], _schema(), SOURCE)
    assert len(result) == 0


def test_unknown_attribute_rejected():
    result = validate_spans(
        [_span("persName", 7, 17, {"unknown_attr": "val"})], _schema(), SOURCE
    )
    assert len(result) == 0


def test_invalid_attribute_value_rejected():
    result = validate_spans(
        [_span("persName", 7, 17, {"cert": "medium"})], _schema(), SOURCE
    )
    assert len(result) == 0


def test_valid_constrained_attribute_passes():
    result = validate_spans(
        [_span("persName", 7, 17, {"cert": "high"})], _schema(), SOURCE
    )
    assert len(result) == 1


def test_free_string_attribute_passes():
    result = validate_spans(
        [_span("persName", 7, 17, {"ref": "http://example.com/p/1"})], _schema(), SOURCE
    )
    assert len(result) == 1


def test_out_of_bounds_span_rejected():
    result = validate_spans([_span("persName", -1, 5)], _schema(), SOURCE)
    assert len(result) == 0
    result2 = validate_spans([_span("persName", 5, 200)], _schema(), SOURCE)
    assert len(result2) == 0


def test_empty_span_list():
    assert validate_spans([], _schema(), SOURCE) == []


# --- validate_output ---


def test_validate_output_plain_source_passes():
    validate_output("He met John Smith.", "He met John Smith.")


def test_validate_output_tags_injected_passes():
    validate_output("He met <persName>John Smith</persName>.", "He met John Smith.")


def test_validate_output_multiple_tags_passes():
    validate_output(
        "<s>He met <persName>John Smith</persName>.</s>",
        "He met John Smith.",
    )


def test_validate_output_whitespace_difference_passes():
    # newline in source normalises the same as a space
    validate_output("He met\n<persName>John Smith</persName>.", "He met\nJohn Smith.")


def test_validate_output_empty_passes():
    validate_output("", "")


def test_validate_output_dropped_word_raises():
    with pytest.raises(ValueError, match="mismatch"):
        validate_output("He met <persName>Smith</persName>.", "He met John Smith.")


def test_validate_output_duplicated_word_raises():
    with pytest.raises(ValueError, match="mismatch"):
        validate_output(
            "He met <persName>John John Smith</persName>.", "He met John Smith."
        )


def test_validate_output_error_contains_diff():
    with pytest.raises(ValueError) as exc_info:
        validate_output("He met Smith.", "He met John Smith.")
    assert "John" in str(exc_info.value)


# --- whitespace normalisation: reformatting and annotation displacement ---


def test_multiline_source_normalises_same():
    # newline between fields (common in TEI records)
    validate_output(
        "Müller,\n<persName>Wilhelm</persName>",
        "Müller,\nWilhelm",
    )


def test_multiple_spaces_in_source_normalise():
    # double space in source is fine — both sides normalise to single space
    validate_output(
        "He  met <persName>John Smith</persName>",
        "He  met John Smith",
    )


def test_tab_in_source_normalises():
    validate_output(
        "Author:\t<persName>John Smith</persName>",
        "Author:\tJohn Smith",
    )


def test_leading_space_absorbed_into_span_boundary_normalises():
    # span start offset is one character too early, capturing the preceding space;
    # tag wrapping becomes "met<persName> John" — whitespace normalisation recovers
    validate_output(
        "He met<persName> John Smith</persName>",
        "He met John Smith",
    )


def test_trailing_space_shifted_outside_span_normalises():
    # trailing space of the span text ends up outside the closing tag
    validate_output(
        "<persName>John Smith </persName>goes",
        "John Smith goes",
    )


def test_space_dropped_between_words_raises():
    # injector merges two tokens by omitting the separating space — must be caught
    with pytest.raises(ValueError, match="mismatch"):
        validate_output(
            "He met<persName>John Smith</persName>",
            "He met John Smith",
        )


def test_leading_trailing_whitespace_stripped_both_sides():
    validate_output(
        "  <persName>John Smith</persName>  ",
        "  John Smith  ",
    )
