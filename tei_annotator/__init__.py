"""
tei-annotator: TEI XML annotation library using a two-stage LLM pipeline.
"""

from .inference.endpoint import EndpointCapability, EndpointConfig
from .models.schema import TEIAttribute, TEIElement, TEISchema
from .models.spans import ResolvedSpan, SpanDescriptor
from .pipeline import AnnotationResult, annotate

__all__ = [
    "annotate",
    "AnnotationResult",
    "TEISchema",
    "TEIElement",
    "TEIAttribute",
    "SpanDescriptor",
    "ResolvedSpan",
    "EndpointConfig",
    "EndpointCapability",
]
