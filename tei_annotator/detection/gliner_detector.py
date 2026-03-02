from __future__ import annotations

import warnings as _warnings

from ..models.schema import TEISchema
from ..models.spans import SpanDescriptor

try:
    from gliner import GLiNER as _GLiNER
except ImportError as _e:
    raise ImportError(
        "The 'gliner' package is required for GLiNER detection. "
        "Install it with: pip install tei-annotator[gliner]"
    ) from _e

_model_cache: dict[str, _GLiNER] = {}


def _load_model(model_id: str) -> _GLiNER:
    if model_id not in _model_cache:
        with _warnings.catch_warnings():
            _warnings.filterwarnings(
                "ignore",
                message=".*resume_download.*",
                category=UserWarning,
            )
            _model_cache[model_id] = _GLiNER.from_pretrained(model_id)
    return _model_cache[model_id]


def preload_model(model_id: str) -> None:
    """Load and cache the GLiNER model; safe to call multiple times for the same ID."""
    _load_model(model_id)


def detect_spans(
    text: str,
    schema: TEISchema,
    model_id: str = "numind/NuNER_Zero",
) -> list[SpanDescriptor]:
    """
    Detect entity spans in *text* using a GLiNER model.

    Model weights are fetched from HuggingFace Hub on first use and cached in
    ~/.cache/huggingface/.  All listed models run on CPU; no GPU required.

    Recommended model_id values:
        - "numind/NuNER_Zero"                         (MIT, default)
        - "urchade/gliner_medium-v2.1"                (Apache-2.0, balanced)
        - "knowledgator/gliner-multitask-large-v0.5"  (adds relation extraction)
    """
    model = _load_model(model_id)

    # Map TEI element descriptions to their tags
    labels = [elem.description for elem in schema.elements]
    tag_for_label = {elem.description: elem.tag for elem in schema.elements}

    entities = model.predict_entities(text, labels)

    spans: list[SpanDescriptor] = []
    for entity in entities:
        ctx_start = max(0, entity["start"] - 60)
        ctx_end = min(len(text), entity["end"] + 60)
        context = text[ctx_start:ctx_end]

        tag = tag_for_label.get(entity["label"], entity["label"])
        spans.append(
            SpanDescriptor(
                element=tag,
                text=entity["text"],
                context=context,
                attrs={},
                confidence=entity.get("score"),
            )
        )

    return spans
