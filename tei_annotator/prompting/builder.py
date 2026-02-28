from __future__ import annotations

import json
from pathlib import Path

from ..inference.endpoint import EndpointCapability
from ..models.schema import TEISchema
from ..models.spans import SpanDescriptor

try:
    from jinja2 import Environment, FileSystemLoader

    _HAS_JINJA = True
except ImportError:
    _HAS_JINJA = False

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _get_env() -> "Environment":
    if not _HAS_JINJA:
        raise ImportError(
            "jinja2 is required for prompt building. Install it with: pip install jinja2"
        )
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), keep_trailing_newline=True)
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
    return env


def build_prompt(
    source_text: str,
    schema: TEISchema,
    capability: EndpointCapability,
    candidates: list[SpanDescriptor] | None = None,
) -> str:
    """
    Build an LLM prompt for the given endpoint capability.

    Raises ValueError for EXTRACTION endpoints (they don't use text prompts).
    """
    if capability == EndpointCapability.EXTRACTION:
        raise ValueError(
            "EXTRACTION endpoints use their own native format; no text prompt needed."
        )

    env = _get_env()
    template_name = (
        "text_gen.jinja2"
        if capability == EndpointCapability.TEXT_GENERATION
        else "json_enforced.jinja2"
    )
    template = env.get_template(template_name)

    candidate_dicts: list[dict] | None = None
    if candidates:
        candidate_dicts = [
            {
                "element": c.element,
                "text": c.text,
                "context": c.context,
                "attrs": c.attrs,
                **({"confidence": c.confidence} if c.confidence is not None else {}),
            }
            for c in candidates
        ]

    return template.render(
        schema=schema,
        source_text=source_text,
        candidates=candidate_dicts,
    )


def make_correction_prompt(original_response: str, error_message: str) -> str:
    """Build a self-correction retry prompt that includes the bad response and the error."""
    return (
        "Your previous response could not be parsed as JSON.\n"
        f"Error: {error_message}\n\n"
        f"Your previous response was:\n{original_response}\n\n"
        "Please fix the JSON and return only a valid JSON array of span objects. "
        "Do not include any markdown formatting or explanation."
    )
