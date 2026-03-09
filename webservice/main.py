"""
TEI Annotator webservice.

Start with:
    python main.py           # reads HOST/PORT from .env
    python main.py --reload  # development mode with auto-reload

Do NOT start with `uvicorn main:app` directly: uvicorn parses its --port from
the CLI before importing this module, so load_dotenv() would run too late to
affect the port binding.

Configuration is read from a .env file in this directory (or from environment
variables).  Copy .env.template to .env and fill in at least one API key.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
_KISSKI_KEY = os.environ.get("KISSKI_API_KEY", "")
_DEFAULT_PROVIDER = os.environ.get("DEFAULT_PROVIDER", "gemini")
_GLINER_MODEL = os.environ.get("GLINER_MODEL", "") or None

_AVAILABLE_PROVIDERS: list[str] = []
if _GEMINI_KEY:
    _AVAILABLE_PROVIDERS.append("gemini")
if _KISSKI_KEY:
    _AVAILABLE_PROVIDERS.append("kisski")

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


# ---------------------------------------------------------------------------
# Provider call_fn factories
# ---------------------------------------------------------------------------


def _make_gemini_call_fn(api_key: str, model: str = "gemini-2.0-flash", timeout: int = 120):
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models"
        f"/{model}:generateContent?key={api_key}"
    )

    def call_fn(prompt: str) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        result = _post_json(url, payload, {"Content-Type": "application/json"}, timeout)
        return result["candidates"][0]["content"]["parts"][0]["text"]

    call_fn.__name__ = f"gemini/{model}"
    return call_fn


def _make_kisski_call_fn(
    api_key: str,
    base_url: str = "https://chat-ai.academiccloud.de/v1",
    model: str = "llama-3.3-70b-instruct",
    timeout: int = 120,
):
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def call_fn(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        result = _post_json(url, payload, headers, timeout)
        return result["choices"][0]["message"]["content"]

    call_fn.__name__ = f"kisski/{model}"
    return call_fn


def _get_call_fn(provider: str):
    if provider == "gemini":
        if not _GEMINI_KEY:
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured")
        return _make_gemini_call_fn(_GEMINI_KEY)
    if provider == "kisski":
        if not _KISSKI_KEY:
            raise HTTPException(status_code=503, detail="KISSKI_API_KEY not configured")
        return _make_kisski_call_fn(_KISSKI_KEY)
    raise HTTPException(status_code=400, detail=f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _schema_from_dict(data: dict):
    """Build a TEISchema from a plain dict (as received from the JSON API)."""
    from tei_annotator.models.schema import TEIAttribute, TEIElement, TEISchema

    elements = []
    for e in data.get("elements", []):
        attrs = [
            TEIAttribute(
                name=a["name"],
                description=a.get("description", ""),
                allowed_values=a.get("allowed_values"),
            )
            for a in e.get("attributes", [])
        ]
        elements.append(
            TEIElement(
                tag=e["tag"],
                description=e.get("description", ""),
                allowed_children=e.get("allowed_children", []),
                attributes=attrs,
            )
        )
    return TEISchema(elements=elements, rules=data.get("rules", []))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TEI Annotator",
    description="Annotate plain text with TEI XML tags using an LLM backend.",
    version="0.1.0",
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# HTML endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "providers": _AVAILABLE_PROVIDERS,
            "default_provider": _DEFAULT_PROVIDER if _DEFAULT_PROVIDER in _AVAILABLE_PROVIDERS else (_AVAILABLE_PROVIDERS[0] if _AVAILABLE_PROVIDERS else ""),
            "result": None,
            "error": None,
            "input_text": "",
        },
    )


@app.post("/annotate", response_class=HTMLResponse)
async def annotate_html(
    request: Request,
    text: str = Form(...),
    provider: str = Form(...),
):
    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.pipeline import annotate
    from tei_annotator.schemas.blbl import build_blbl_schema

    result_xml = None
    error = None
    try:
        call_fn = _get_call_fn(provider)
        endpoint = EndpointConfig(
            capability=EndpointCapability.TEXT_GENERATION,
            call_fn=call_fn,
        )
        result = annotate(
            text=text,
            schema=build_blbl_schema(),
            endpoint=endpoint,
            gliner_model=_GLINER_MODEL,
        )
        result_xml = result.xml
    except HTTPException as exc:
        error = exc.detail
    except Exception as exc:
        error = str(exc)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "providers": _AVAILABLE_PROVIDERS,
            "default_provider": provider,
            "result": result_xml,
            "error": error,
            "input_text": text,
        },
    )


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


class AttributeSchema(BaseModel):
    name: str
    description: str = ""
    allowed_values: list[str] | None = None


class ElementSchema(BaseModel):
    tag: str
    description: str = ""
    allowed_children: list[str] = []
    attributes: list[AttributeSchema] = []


class SchemaInput(BaseModel):
    elements: list[ElementSchema]
    rules: list[str] = []


class AnnotateRequest(BaseModel):
    model_config = {"populate_by_name": True}

    text: str
    provider: str | None = None
    tei_schema: SchemaInput | None = Field(None, alias="schema")


class FuzzySpan(BaseModel):
    element: str
    start: int
    end: int


class AnnotateResponse(BaseModel):
    xml: str
    fuzzy_spans: list[FuzzySpan]


@app.post("/api/annotate", response_model=AnnotateResponse)
async def annotate_api(body: AnnotateRequest):
    """
    Annotate *text* and return the XML result.

    - **text**: plain text to annotate.
    - **provider**: `"gemini"` or `"kisski"` (default: `DEFAULT_PROVIDER` from env).
    - **schema**: TEI schema definition. Omit to use the built-in BLBL bibliographic schema.
    """
    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.pipeline import annotate
    from tei_annotator.schemas.blbl import build_blbl_schema

    provider = body.provider or _DEFAULT_PROVIDER
    call_fn = _get_call_fn(provider)

    if body.tei_schema is not None:
        schema = _schema_from_dict(body.tei_schema.model_dump())
    else:
        schema = build_blbl_schema()

    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=call_fn,
    )

    try:
        result = annotate(
            text=body.text,
            schema=schema,
            endpoint=endpoint,
            gliner_model=_GLINER_MODEL,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnnotateResponse(
        xml=result.xml,
        fuzzy_spans=[
            FuzzySpan(element=s.element, start=s.start, end=s.end)
            for s in result.fuzzy_spans
        ],
    )


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    _parser = argparse.ArgumentParser()
    _parser.add_argument("--reload", action="store_true", default=False,
                         help="Enable auto-reload on code changes (development only).")
    _args = _parser.parse_args()

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=_args.reload)
