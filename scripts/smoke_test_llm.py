#!/usr/bin/env python
"""
End-to-end smoke test: tei-annotator pipeline with real LLM endpoints.

Providers tested:
  • Google Gemini 2.0 Flash
  • KISSKI (OpenAI-compatible API, llama-3.3-70b-instruct)

Reads API keys from .env in the project root.

Usage:
    uv run scripts/smoke_test_llm.py
    python scripts/smoke_test_llm.py      # if venv is already activated
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# HTTP helper (stdlib urllib)
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


# ---------------------------------------------------------------------------
# call_fn factories
# ---------------------------------------------------------------------------

def make_gemini_call_fn(api_key: str, model: str = "gemini-2.0-flash") -> ...:
    """Return a call_fn that sends a prompt to Gemini and returns the text reply."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models"
        f"/{model}:generateContent?key={api_key}"
    )

    def call_fn(prompt: str) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        result = _post_json(url, payload, {"Content-Type": "application/json"})
        return result["candidates"][0]["content"]["parts"][0]["text"]

    call_fn.__name__ = f"gemini/{model}"
    return call_fn


def make_kisski_call_fn(
    api_key: str,
    base_url: str = "https://chat-ai.academiccloud.de/v1",
    model: str = "llama-3.3-70b-instruct",
) -> ...:
    """Return a call_fn that sends a prompt to a KISSKI-hosted OpenAI-compatible model."""
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    def call_fn(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        result = _post_json(url, payload, headers)
        return result["choices"][0]["message"]["content"]

    call_fn.__name__ = f"kisski/{model}"
    return call_fn


# ---------------------------------------------------------------------------
# Test scenario
# ---------------------------------------------------------------------------

TEST_TEXT = (
    "Marie Curie was born in Warsaw, Poland, and later conducted her research "
    "in Paris, France. Together with her husband Pierre Curie, she discovered "
    "polonium and radium."
)

# We just check that the pipeline runs and produces *some* annotation.
# Whether the LLM chose the right entities is not asserted here.
EXPECTED_TAGS = ["persName", "placeName"]


def _build_schema():
    from tei_annotator.models.schema import TEIAttribute, TEIElement, TEISchema

    return TEISchema(
        elements=[
            TEIElement(
                tag="persName",
                description="a person's name",
                attributes=[TEIAttribute(name="ref", description="authority URI")],
            ),
            TEIElement(
                tag="placeName",
                description="a geographical place name",
                attributes=[TEIAttribute(name="ref", description="authority URI")],
            ),
        ]
    )


def run_smoke_test(provider_name: str, call_fn) -> bool:
    """
    Run the full annotate() pipeline with *call_fn* and print results.
    Returns True on success, False on failure.
    """
    import re

    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.pipeline import annotate

    print(f"\n{'─' * 60}")
    print(f"  Provider : {provider_name}")
    print(f"  Input    : {TEST_TEXT[:80]}…")
    print(f"{'─' * 60}")

    try:
        result = annotate(
            text=TEST_TEXT,
            schema=_build_schema(),
            endpoint=EndpointConfig(
                capability=EndpointCapability.TEXT_GENERATION,
                call_fn=call_fn,
            ),
            gliner_model=None,  # skip GLiNER for speed
        )
    except Exception as exc:
        print(f"  ✗ FAILED  — exception during annotate(): {exc}")
        return False

    # Verify plain text is unmodified
    plain = re.sub(r"<[^>]+>", "", result.xml)
    if plain != TEST_TEXT:
        print(f"  ✗ FAILED  — plain text was modified by the pipeline")
        print(f"    Expected : {TEST_TEXT!r}")
        print(f"    Got      : {plain!r}")
        return False

    # Verify at least one annotation was injected (LLM must have found something)
    has_any_tag = any(f"<{t}>" in result.xml for t in EXPECTED_TAGS)
    if not has_any_tag:
        print(f"  ✗ FAILED  — no annotation tags found in output")
        print(f"    Output XML: {result.xml}")
        return False

    # Pretty-print the result
    tags_found = [t for t in EXPECTED_TAGS if f"<{t}>" in result.xml]
    print(f"  ✓ PASSED")
    print(f"  Tags found : {', '.join(tags_found)}")
    if result.fuzzy_spans:
        print(f"  Fuzzy spans: {[TEST_TEXT[s.start:s.end] for s in result.fuzzy_spans]}")
    print(f"  Output XML :")
    for line in textwrap.wrap(result.xml, width=72, subsequent_indent="    "):
        print(f"    {line}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    kisski_key = os.environ.get("KISSKI_API_KEY", "")

    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not set (check .env)", file=sys.stderr)
        return 1
    if not kisski_key:
        print("ERROR: KISSKI_API_KEY not set (check .env)", file=sys.stderr)
        return 1

    providers: list[tuple[str, object]] = [
        ("Gemini 2.0 Flash", make_gemini_call_fn(gemini_key)),
        ("KISSKI / llama-3.3-70b-instruct", make_kisski_call_fn(kisski_key)),
    ]

    results: list[bool] = []
    for name, fn in providers:
        results.append(run_smoke_test(name, fn))

    print(f"\n{'═' * 60}")
    passed = sum(results)
    total = len(results)
    print(f"  Result: {passed}/{total} providers passed")
    print(f"{'═' * 60}\n")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
