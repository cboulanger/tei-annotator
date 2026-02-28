from __future__ import annotations

import json
import re
from typing import Callable

from ..models.spans import SpanDescriptor


def _strip_fences(text: str) -> str:
    """Remove markdown code fences, even if preceded by explanatory text."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _parse_json_list(text: str) -> list[dict] | None:
    """Parse text as a JSON list; return None on failure."""
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        return None


def _dicts_to_spans(raw: list[dict]) -> list[SpanDescriptor]:
    spans: list[SpanDescriptor] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        element = item.get("element", "")
        text = item.get("text", "")
        context = item.get("context", "")
        attrs = item.get("attrs", {})
        if not (element and text and context):
            continue
        spans.append(
            SpanDescriptor(
                element=element,
                text=text,
                context=context,
                attrs=attrs if isinstance(attrs, dict) else {},
            )
        )
    return spans


def parse_response(
    response: str,
    call_fn: Callable[[str], str] | None = None,
    make_correction_prompt: Callable[[str, str], str] | None = None,
) -> list[SpanDescriptor]:
    """
    Parse an LLM response string into a list of SpanDescriptors.

    - Strips markdown code fences automatically.
    - If parsing fails and *call_fn* + *make_correction_prompt* are provided,
      retries once with a self-correction prompt that includes the bad response.
    - Raises ValueError if parsing fails after the retry (or if no retry is configured).
    """
    cleaned = _strip_fences(response)
    raw = _parse_json_list(cleaned)
    if raw is not None:
        return _dicts_to_spans(raw)

    if call_fn is None or make_correction_prompt is None:
        raise ValueError(f"Failed to parse JSON from response: {response[:300]!r}")

    error_msg = "Response is not valid JSON"
    correction_prompt = make_correction_prompt(response, error_msg)
    retry_response = call_fn(correction_prompt)
    retry_cleaned = _strip_fences(retry_response)
    raw = _parse_json_list(retry_cleaned)
    if raw is None:
        raise ValueError(f"Failed to parse JSON after retry: {retry_response[:300]!r}")
    return _dicts_to_spans(raw)
