"""
TEI Annotator webservice.

Start with:
    python main.py           # reads HOST/PORT from .env
    python main.py --reload  # development mode with auto-reload

Do NOT start with `uvicorn main:app` directly: uvicorn parses its --port from
the CLI before importing this module, so load_dotenv() would run too late to
affect the port binding.

Configuration is read from a .env file in this directory (or from environment
variables).  Copy .env.template to .env and fill in HF_TOKEN.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_HF_MODEL_DEFAULT = os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct:nscale")
_GLINER_MODEL = os.environ.get("GLINER_MODEL", "") or None

# Curated list of open models available on the HuggingFace Inference Router.
# All pinned to providers confirmed to work from AWS-hosted HF Spaces.
# Blocked from AWS: groq, cerebras, together-ai, sambanova.
# Safe providers used here: nscale, scaleway.
_HF_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct:nscale",
    "meta-llama/Llama-3.1-70B-Instruct:scaleway",
    "Qwen/Qwen3-32B:nscale",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B:nscale",
    "Qwen/Qwen2.5-Coder-32B-Instruct:nscale",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B:nscale",
    "meta-llama/Llama-3.1-8B-Instruct:nscale",
    "Qwen/Qwen3-14B:nscale",
    "openai/gpt-oss-20b:nscale",
    "Qwen/QwQ-32B:nscale",
]

_HF_BASE_URL = "https://router.huggingface.co/v1"

# Path to the evaluation fixture (relative to this file's parent directory).
_FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "blbl-examples.tei.xml"

# Static HTML file served at GET /.
_HTML_FILE = Path(__file__).parent / "static" / "index.html"

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
# HuggingFace call_fn factory
# ---------------------------------------------------------------------------


def _make_hf_call_fn(model: str, timeout: int = 120):
    url = f"{_HF_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {_HF_TOKEN}"}

    def call_fn(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        result = _post_json(url, payload, headers, timeout)
        return result["choices"][0]["message"]["content"]

    call_fn.__name__ = f"hf/{model}"
    return call_fn


def _get_call_fn(model: str):
    if not _HF_TOKEN:
        raise HTTPException(status_code=503, detail="HF_TOKEN not configured")
    if model not in _HF_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model!r}")
    return _make_hf_call_fn(model)


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


def _run_evaluation(model: str, n: int = 5, seed: int | None = None) -> dict:
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

    call_fn = _get_call_fn(model)
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
    return {
        "n_samples": len(samples),
        "model": model,
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

_DEFAULT_MODEL = _HF_MODEL_DEFAULT if _HF_MODEL_DEFAULT in _HF_MODELS else _HF_MODELS[0]


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
    return {
        "models": _HF_MODELS,
        "default_model": _DEFAULT_MODEL,
        "hf_token_set": bool(_HF_TOKEN),
    }


@app.get("/api/sample")
async def sample_api(n: int = 5):
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
async def annotate_api(body: AnnotateRequest):
    """
    Annotate *text* and return the XML result.

    - **text**: plain text to annotate.
    - **model**: HuggingFace model ID (default: `HF_MODEL` from env).
    - **schema**: TEI schema definition. Omit to use the built-in BLBL bibliographic schema.
    """
    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.pipeline import annotate
    from tei_annotator.schemas.blbl import build_blbl_schema

    model = body.model or _DEFAULT_MODEL
    call_fn = _get_call_fn(model)

    if body.tei_schema is not None:
        schema = _schema_from_dict(body.tei_schema.model_dump())
    else:
        schema = build_blbl_schema()

    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=call_fn,
    )

    try:
        t0 = time.monotonic()
        result = annotate(
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
    model: str | None = None
    n: int = 5
    seed: int | None = None


@app.post("/api/evaluate")
async def evaluate_api(body: EvaluateRequest):
    """
    Sample *n* bibliographic entries from the test fixture, annotate each with
    the chosen model, and return precision/recall/F1 against the gold standard.

    - **model**: HuggingFace model ID (default: `HF_MODEL` from env).
    - **n**: number of random samples (default: 5).
    """
    model = body.model or _DEFAULT_MODEL
    try:
        return _run_evaluation(model, n=body.n, seed=body.seed)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
