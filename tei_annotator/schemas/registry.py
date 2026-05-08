"""
Schema registry for tei-annotator.

To add a new schema, register it in SCHEMA_REGISTRY:
    "schema-name": {
        "build": <callable returning TEISchema>,
        "root_element": "<container element name>",
        "child_element": "<record element name>",
    }
"""

from __future__ import annotations


def _build_bibl():
    from tei_annotator.schemas.bibl import build_bibl_schema
    return build_bibl_schema()


def _build_bibl_reference_segmenter():
    from tei_annotator.schemas.bibl_reference_segmenter import (
        build_bibl_reference_segmenter_schema,
    )
    return build_bibl_reference_segmenter_schema()


SCHEMA_REGISTRY: dict[str, dict] = {
    "bibl": {
        "build": _build_bibl,
        "root_element": "listBibl",
        "child_element": "bibl",
    },
    "bibl-reference-segmenter": {
        "build": _build_bibl_reference_segmenter,
        "root_element": "text",
        "child_element": "listBibl",
    },
}


def get_schema_names() -> list[str]:
    """Return all registered schema names."""
    return list(SCHEMA_REGISTRY)


def get_schema_config(name: str) -> dict:
    """Return the config dict for *name*. Raises KeyError if unknown."""
    return SCHEMA_REGISTRY[name]


def build_schema(name: str):
    """Build and return the TEISchema for *name*."""
    return SCHEMA_REGISTRY[name]["build"]()
