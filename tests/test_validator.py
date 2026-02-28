from tei_annotator.models.schema import TEIAttribute, TEIElement, TEISchema
from tei_annotator.models.spans import ResolvedSpan
from tei_annotator.postprocessing.validator import validate_spans

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
