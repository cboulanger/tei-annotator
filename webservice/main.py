"""
TEI Annotator webservice.

Start with:
    python main.py           # reads HOST/PORT from .env
    python main.py --reload  # development mode with auto-reload

Do NOT start with `uvicorn main:app` directly: uvicorn parses its --port from
the CLI before importing this module, so load_dotenv() would run too late to
affect the port binding.

Configuration is read from a .env file in this directory (or from environment
variables).  Copy .env.template to .env and fill in your provider API keys.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

_GLINER_MODEL = os.environ.get("GLINER_MODEL", "") or None

# Optional shared-secret token.  Set DEMO_TOKEN in .env to enable enforcement.
# Leave empty for open access (local development).
_DEMO_TOKEN = os.environ.get("DEMO_TOKEN", "") or None


async def _verify_token(authorization: str | None = Header(None)) -> None:
    """Dependency: reject requests that don't carry the configured bearer token."""
    if _DEMO_TOKEN is None:
        return  # token enforcement disabled
    if authorization != f"Bearer {_DEMO_TOKEN}":
        raise HTTPException(status_code=401, detail="Missing or invalid token.")

from connectors import get_available_connectors, get_connector  # noqa: E402

# SELECTED_MODEL=<provider>/<model> — e.g. "hf/meta-llama/Llama-3.3-70B-Instruct:nscale"
_sel = os.environ.get("SELECTED_MODEL", "")
_SELECTED_PROVIDER, _SELECTED_MODEL_ID = (_sel.split("/", 1) if "/" in _sel else (None, None))

# Path to the evaluation fixture (relative to this file's parent directory).
_FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "blbl-examples.tei.xml"

# Static HTML file served at GET /.
_HTML_FILE = Path(__file__).parent / "static" / "index.html"


# ---------------------------------------------------------------------------
# Connector / call_fn helper
# ---------------------------------------------------------------------------


def _resolve_call_fn(provider_id: str | None, model_id: str | None, timeout: int = 300):
    connectors = get_available_connectors()
    if not connectors:
        raise HTTPException(status_code=503, detail="No providers configured. Set at least one API key.")
    if not provider_id and not model_id:
        provider_id = _SELECTED_PROVIDER
        model_id = _SELECTED_MODEL_ID
    connector = get_connector(provider_id) if provider_id else connectors[0]
    if connector is None or not connector.is_available():
        raise HTTPException(status_code=400, detail=f"Provider {provider_id!r} not available")
    model = model_id or connector.default_model
    if model not in connector.models():
        raise HTTPException(status_code=400, detail=f"Model {model!r} not available for provider {connector.id!r}")
    return connector.make_call_fn(model, timeout=timeout)


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
# Evaluation helpers
# ---------------------------------------------------------------------------

_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _load_fixture_bibls():
    """Parse the fixture file and return all <bibl> elements."""
    from lxml import etree

    if not _FIXTURE_PATH.exists():
        raise RuntimeError(
            f"Evaluation fixture not found at {_FIXTURE_PATH}. "
            "Make sure the tests/fixtures/ directory is present in the deployment."
        )
    tree = etree.parse(str(_FIXTURE_PATH))
    return tree.findall(".//tei:bibl", _TEI_NS)


def _run_evaluation(
    provider_id: str | None,
    model_id: str | None,
    n: int = 5,
    seed: int | None = None,
) -> dict:
    """Sample n bibl elements, annotate each, compute metrics vs gold standard.

    Pass *seed* to reproduce the same sample across multiple calls (e.g. when
    comparing several models on identical inputs).
    """
    from lxml import etree

    from tei_annotator.evaluation.extractor import extract_spans
    from tei_annotator.evaluation.metrics import MatchMode, aggregate, compute_metrics
    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.pipeline import annotate
    from tei_annotator.schemas.blbl import build_blbl_schema

    call_fn = _resolve_call_fn(provider_id, model_id)
    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=call_fn,
    )
    schema = build_blbl_schema()
    bibls = _load_fixture_bibls()
    rng = random.Random(seed)
    samples = rng.sample(bibls, min(n, len(bibls)))

    per_result = []
    sample_texts = []
    t0 = time.monotonic()
    for el in samples:
        plain_text, gold_spans = extract_spans(el)
        sample_texts.append(plain_text)
        ann_result = annotate(plain_text, schema, endpoint, gliner_model=_GLINER_MODEL)
        try:
            _parser = etree.XMLParser(recover=True)
            pred_el = etree.fromstring(f"<bibl>{ann_result.xml}</bibl>".encode(), _parser)
        except Exception:
            pred_el = etree.Element("bibl")
        _, pred_spans = extract_spans(pred_el)
        per_result.append(compute_metrics(gold_spans, pred_spans, mode=MatchMode.TEXT))

    elapsed = time.monotonic() - t0
    agg = aggregate(per_result)
    # Include a display label combining provider+model for the UI
    connector = get_connector(provider_id) if provider_id else get_available_connectors()[0]
    resolved_model = model_id or connector.default_model
    return {
        "n_samples": len(samples),
        "provider": connector.id,
        "model": resolved_model,
        "sample_texts": sample_texts,
        "elapsed_seconds": round(elapsed, 1),
        "micro_precision": agg.micro_precision,
        "micro_recall": agg.micro_recall,
        "micro_f1": agg.micro_f1,
        "per_element": {
            tag: {
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "tp": m.true_positives,
                "fp": m.false_positives,
                "fn": m.false_negatives,
            }
            for tag, m in sorted(agg.per_element.items())
        },
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TEI Annotator",
    description="Annotate plain text with TEI XML tags using an LLM backend.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(_HTML_FILE, media_type="text/html")


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


@app.get("/api/config")
async def api_config():
    """Return server configuration needed by the SPA on startup."""
    providers = []
    for c in get_available_connectors():
        models = c.models()
        if _SELECTED_PROVIDER == c.id and _SELECTED_MODEL_ID in models:
            default_model = _SELECTED_MODEL_ID
        else:
            default_model = c.default_model
        providers.append({
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "models": models,
            "default_model": default_model,
        })
    return {"providers": providers, "token": _DEMO_TOKEN}


@app.get("/api/sample")
async def sample_api(n: int = 5, _: None = Depends(_verify_token)):
    """
    Return *n* random plain-text bibliographic entries from the test fixture.
    Useful for populating the evaluation textarea in the UI.
    """
    from tei_annotator.evaluation.extractor import extract_spans

    bibls = _load_fixture_bibls()
    samples = random.sample(bibls, min(n, len(bibls)))
    return [{"text": extract_spans(el)[0]} for el in samples]


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
    model: str | None = None
    tei_schema: SchemaInput | None = Field(None, alias="schema")


class FuzzySpan(BaseModel):
    element: str
    start: int
    end: int


class AnnotateResponse(BaseModel):
    xml: str
    fuzzy_spans: list[FuzzySpan]
    elapsed_seconds: float


@app.post("/api/annotate", response_model=AnnotateResponse)
async def annotate_api(body: AnnotateRequest, _: None = Depends(_verify_token)):
    """
    Annotate *text* and return the XML result.

    - **text**: plain text to annotate.
    - **provider**: connector id (default: first available provider).
    - **model**: model ID for the provider (default: provider's default model).
    - **schema**: TEI schema definition. Omit to use the built-in BLBL bibliographic schema.
    """
    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.pipeline import annotate
    from tei_annotator.schemas.blbl import build_blbl_schema

    call_fn = _resolve_call_fn(body.provider, body.model)

    if body.tei_schema is not None:
        schema = _schema_from_dict(body.tei_schema.model_dump())
    else:
        schema = build_blbl_schema()

    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=call_fn,
    )

    try:
        import asyncio
        t0 = time.monotonic()
        result = await asyncio.to_thread(
            annotate,
            text=body.text,
            schema=schema,
            endpoint=endpoint,
            gliner_model=_GLINER_MODEL,
        )
        elapsed = time.monotonic() - t0
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnnotateResponse(
        xml=result.xml,
        elapsed_seconds=round(elapsed, 1),
        fuzzy_spans=[
            FuzzySpan(element=s.element, start=s.start, end=s.end)
            for s in result.fuzzy_spans
        ],
    )


class EvaluateRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    n: int = 5
    seed: int | None = None


@app.post("/api/evaluate")
async def evaluate_api(body: EvaluateRequest, _: None = Depends(_verify_token)):
    """
    Sample *n* bibliographic entries from the test fixture, annotate each with
    the chosen model, and return precision/recall/F1 against the gold standard.

    - **provider**: connector id (default: first available provider).
    - **model**: model ID for the provider (default: provider's default model).
    - **n**: number of random samples (default: 5).
    """
    try:
        import asyncio
        return await asyncio.to_thread(_run_evaluation, body.provider, body.model, n=body.n, seed=body.seed)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

_PID_FILE = Path(__file__).parent / ".server.pid"


def _kill_previous() -> None:
    """Kill the process recorded in .server.pid, if it still exists."""
    import signal

    if not _PID_FILE.exists():
        return
    try:
        pid = int(_PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        # Give it a moment to release the port
        import time as _time
        _time.sleep(0.5)
    except (ProcessLookupError, ValueError):
        pass  # already gone
    finally:
        _PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse
    import uvicorn

    _parser = argparse.ArgumentParser()
    _parser.add_argument("--reload", action="store_true", default=False,
                         help="Enable auto-reload on code changes (development only).")
    _args = _parser.parse_args()

    _kill_previous()
    _PID_FILE.write_text(str(os.getpid()))
    try:
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "8000"))
        uvicorn.run("main:app", host=host, port=port, reload=_args.reload)
    finally:
        _PID_FILE.unlink(missing_ok=True)
