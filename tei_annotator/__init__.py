"""
tei-annotator: TEI XML annotation library using a two-stage LLM pipeline.
"""

from .inference.endpoint import EndpointCapability, EndpointConfig
from .models.schema import TEIAttribute, TEIElement, TEISchema
from .models.spans import ResolvedSpan, SpanDescriptor
from .pipeline import AnnotationResult, annotate, preload_gliner_model
from .tei import create_schema
from .evaluation import (
    EvaluationSpan,
    EvaluationResult,
    ElementMetrics,
    MatchMode,
    SpanMatch,
    aggregate,
    compute_metrics,
    evaluate_element,
    evaluate_file,
    extract_spans,
    match_spans,
    spans_from_xml_string,
)

__all__ = [
    "annotate",
    "AnnotationResult",
    "preload_gliner_model",
    "create_schema",
    "TEISchema",
    "TEIElement",
    "TEIAttribute",
    "SpanDescriptor",
    "ResolvedSpan",
    "EndpointConfig",
    "EndpointCapability",
    # evaluation
    "EvaluationSpan",
    "EvaluationResult",
    "ElementMetrics",
    "MatchMode",
    "SpanMatch",
    "aggregate",
    "compute_metrics",
    "evaluate_element",
    "evaluate_file",
    "extract_spans",
    "match_spans",
    "spans_from_xml_string",
]
