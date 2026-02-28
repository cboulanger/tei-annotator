from __future__ import annotations

from ..models.schema import TEISchema
from ..models.spans import SpanDescriptor

try:
    from gliner import GLiNER as _GLiNER
except ImportError as _e:
    raise ImportError(
        "The 'gliner' package is required for GLiNER detection. "
        "Install it with: pip install tei-annotator[gliner]"
    ) from _e


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
    model = _GLiNER.from_pretrained(model_id)

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
