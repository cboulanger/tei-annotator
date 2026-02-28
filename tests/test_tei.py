"""Tests for tei_annotator.tei.create_schema."""

from pathlib import Path

import pytest

from tei_annotator.tei import create_schema

# Path to the bundled test schema — adjust if the schema moves.
_SCHEMA = Path(__file__).parent.parent / "schema" / "tei-bib.rng"


# ---------------------------------------------------------------------------
# create_schema basic contract
# ---------------------------------------------------------------------------


def test_create_schema_returns_tei_schema():
    """create_schema returns a TEISchema instance."""
    from tei_annotator.models.schema import TEISchema

    schema = create_schema(_SCHEMA, element="idno", depth=1)
    assert isinstance(schema, TEISchema)


def test_create_schema_unknown_element_raises():
    """An element name not in the schema raises ValueError."""
    with pytest.raises(ValueError, match="not found in schema"):
        create_schema(_SCHEMA, element="__nonexistent__", depth=1)


# ---------------------------------------------------------------------------
# idno — a leaf-like element with enumerated attribute values
# idno's content model in tei-bib.rng: text | model.gLike (notAllowed) | idno
# → only child element reachable by name is idno itself (self-nesting).
# ---------------------------------------------------------------------------


def test_idno_in_schema():
    schema = create_schema(_SCHEMA, element="idno", depth=1)
    idno = schema.get("idno")
    assert idno is not None
    assert idno.tag == "idno"


def test_idno_description():
    schema = create_schema(_SCHEMA, element="idno", depth=1)
    idno = schema.get("idno")
    # The TEI documentation says "(identifier) supplies any form of identifier…"
    assert "identifier" in idno.description.lower()


def test_idno_type_attribute_with_allowed_values():
    schema = create_schema(_SCHEMA, element="idno", depth=1)
    idno = schema.get("idno")

    type_attr = next((a for a in idno.attributes if a.name == "type"), None)
    assert type_attr is not None, "Expected a 'type' attribute on idno"
    assert type_attr.allowed_values is not None
    for expected in ("ISBN", "ISSN", "DOI", "URI"):
        assert expected in type_attr.allowed_values, (
            f"Expected '{expected}' in type allowed_values, got {type_attr.allowed_values}"
        )


def test_idno_self_referential_child():
    """idno can contain idno — should appear in allowed_children."""
    schema = create_schema(_SCHEMA, element="idno", depth=1)
    idno = schema.get("idno")
    assert "idno" in idno.allowed_children


# ---------------------------------------------------------------------------
# biblStruct — element with a small, explicit set of named child elements
# ---------------------------------------------------------------------------
# biblStruct content model (tei-bib.rng):
#   analytic*, (monogr, series*)+ ,
#   (model.noteLike | model.ptrLike | relatedItem | citedRange)*
# model.noteLike → note, noteGrp
# model.ptrLike  → ptr, ref
# ---------------------------------------------------------------------------


def test_biblstruct_in_schema():
    schema = create_schema(_SCHEMA, element="biblStruct", depth=1)
    bib = schema.get("biblStruct")
    assert bib is not None
    assert bib.tag == "biblStruct"


def test_biblstruct_description():
    schema = create_schema(_SCHEMA, element="biblStruct", depth=1)
    bib = schema.get("biblStruct")
    assert "bibliographic" in bib.description.lower()


def test_biblstruct_direct_children_present():
    """analytic, monogr, series, relatedItem, citedRange must be in allowed_children."""
    schema = create_schema(_SCHEMA, element="biblStruct", depth=1)
    bib = schema.get("biblStruct")
    for expected in ("analytic", "monogr", "series", "relatedItem", "citedRange"):
        assert expected in bib.allowed_children, (
            f"Expected '{expected}' in biblStruct.allowed_children"
        )


def test_biblstruct_model_group_children_expanded():
    """Children from model groups (note, ptr) should be expanded into allowed_children."""
    schema = create_schema(_SCHEMA, element="biblStruct", depth=1)
    bib = schema.get("biblStruct")
    for expected in ("note", "ptr"):
        assert expected in bib.allowed_children, (
            f"Expected '{expected}' (from model group) in biblStruct.allowed_children"
        )


def test_biblstruct_has_type_attribute():
    schema = create_schema(_SCHEMA, element="biblStruct", depth=1)
    bib = schema.get("biblStruct")
    attr_names = [a.name for a in bib.attributes]
    assert "type" in attr_names


def test_biblstruct_depth1_includes_children():
    """With depth=1, the children of biblStruct are also present in the schema."""
    schema = create_schema(_SCHEMA, element="biblStruct", depth=1)
    # 'analytic' is a direct child of biblStruct
    analytic = schema.get("analytic")
    assert analytic is not None, "depth=1 should include analytic in the TEISchema"


def test_biblstruct_depth0_excludes_children():
    """With depth=0, only biblStruct itself is in the schema (no children added)."""
    schema = create_schema(_SCHEMA, element="biblStruct", depth=0)
    assert schema.get("biblStruct") is not None
    assert schema.get("analytic") is None, "depth=0 should not include analytic"


def test_no_duplicate_elements_in_schema():
    """Each element name appears at most once in the TEISchema."""
    schema = create_schema(_SCHEMA, element="biblStruct", depth=1)
    tags = [e.tag for e in schema.elements]
    assert len(tags) == len(set(tags)), f"Duplicate elements: {tags}"


def test_no_duplicate_attributes_on_element():
    """Each attribute name appears at most once per TEIElement."""
    schema = create_schema(_SCHEMA, element="idno", depth=0)
    idno = schema.get("idno")
    names = [a.name for a in idno.attributes]
    assert len(names) == len(set(names)), f"Duplicate attributes: {names}"
