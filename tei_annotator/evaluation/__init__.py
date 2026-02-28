"""
tei_annotator.evaluation — Evaluate annotation quality against a gold standard.

Typical usage::

    from tei_annotator.evaluation import evaluate_file, MatchMode

    per_record, overall = evaluate_file(
        gold_xml_path="tests/fixtures/blbl-examples.tei.xml",
        schema=my_schema,
        endpoint=my_endpoint,
        match_mode=MatchMode.TEXT,
    )
    print(overall.report())
"""

from .evaluator import evaluate_bibl, evaluate_file
from .extractor import EvaluationSpan, extract_spans, spans_from_xml_string
from .metrics import (
    ElementMetrics,
    EvaluationResult,
    MatchMode,
    SpanMatch,
    aggregate,
    compute_metrics,
    match_spans,
)

__all__ = [
    # Extractor
    "EvaluationSpan",
    "extract_spans",
    "spans_from_xml_string",
    # Metrics
    "MatchMode",
    "SpanMatch",
    "ElementMetrics",
    "EvaluationResult",
    "match_spans",
    "compute_metrics",
    "aggregate",
    # Evaluator
    "evaluate_bibl",
    "evaluate_file",
]
