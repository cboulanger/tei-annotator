from __future__ import annotations

from ..models.schema import TEISchema
from ..models.spans import ResolvedSpan


def validate_spans(
    spans: list[ResolvedSpan],
    schema: TEISchema,
    source: str,
) -> list[ResolvedSpan]:
    """
    Filter out spans that fail schema validation.

    Rejected when:
    - element is not in the schema
    - an attribute name is not listed for that element
    - an attribute value is not in the element's allowed_values (when constrained)
    - span bounds are out of range
    """
    valid: list[ResolvedSpan] = []

    for span in spans:
        # Bounds sanity check
        if span.start < 0 or span.end > len(source) or span.start >= span.end:
            continue

        elem = schema.get(span.element)
        if elem is None:
            continue  # element not in schema

        allowed_names = {a.name for a in elem.attributes}
        attr_ok = True
        for attr_name, attr_value in span.attrs.items():
            if attr_name not in allowed_names:
                attr_ok = False
                break
            attr_def = next((a for a in elem.attributes if a.name == attr_name), None)
            if attr_def and attr_def.allowed_values is not None:
                if attr_value not in attr_def.allowed_values:
                    attr_ok = False
                    break

        if not attr_ok:
            continue

        valid.append(span)

    return valid
