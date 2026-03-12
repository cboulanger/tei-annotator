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

# Optional shared-secret bearer token for general API access.
# Set API_KEY in .env to enable enforcement; leave empty for open access.
_API_KEY = os.environ.get("API_KEY", "") or None

# Optional second token that unlocks premium (expensive) models in the UI
# and in the API.  Sent by the browser as X-Premium-Key when the visitor
# arrived via ?key=<secret>.  Leave empty to disable the premium tier.
_PREMIUM_TOKEN = os.environ.get("PREMIUM_TOKEN", "") or None


async def _verify_token(authorization: str | None = Header(None)) -> None:
    """Dependency: reject requests that don't carry the configured API_KEY."""
    if _API_KEY is None:
        return  # enforcement disabled
    if authorization != f"Bearer {_API_KEY}":
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")

from connectors import get_available_connectors, get_connector  # noqa: E402

# Separator used to join/split multiple texts in a single LLM call.
# Must not appear in any real bibliographic text.
_BATCH_SEP = "\n---RECORD|||SEP|||BOUNDARY---\n"

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


def _resolve_call_fn(
    provider_id: str | None,
    model_id: str | None,
    timeout: int = 300,
    premium_key: str | None = None,
):
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
    # Enforce premium gate: if the model is not in standard_models(), require the premium key.
    if _PREMIUM_TOKEN and model not in connector.standard_models():
        if premium_key != _PREMIUM_TOKEN:
            raise HTTPException(status_code=403, detail=f"Model {model!r} requires the premium key.")
    return connector.make_call_fn(model, timeout=timeout)


# ---------------------------------------------------------------------------
# Batch annotation helper
# ---------------------------------------------------------------------------


def _batch_annotate_texts(
    texts: list,
    schema,
    endpoint,
    gliner_model: str | None,
    batch_size: int = 1,
) -> list:
    """Annotate *texts* in batches of *batch_size* using a single LLM call per batch.

    Returns a list of xml strings (one per input text).  On a batch-split
    mismatch the affected texts receive an empty string.
    """
    import warnings

    from tei_annotator.pipeline import annotate

    results = [""] * len(texts)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]

        # Guard: separator must not appear in any text of this batch
        if any(_BATCH_SEP in t for t in chunk):
            warnings.warn(
                "A text in this batch contains the batch separator; "
                "falling back to empty results for this batch.",
                stacklevel=2,
            )
            continue

        combined = _BATCH_SEP.join(chunk)
        ann_result = annotate(
            text=combined,
            schema=schema,
            endpoint=endpoint,
            gliner_model=gliner_model,
        )
        pieces = ann_result.xml.split(_BATCH_SEP)

        if len(pieces) != len(chunk):
            warnings.warn(
                f"Batch split mismatch: expected {len(chunk)} pieces, "
                f"got {len(pieces)}. Returning empty results for this batch.",
                stacklevel=2,
            )
            continue

        for j, piece in enumerate(pieces):
            results[start + j] = piece

    return results


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
    premium_key: str | None = None,
    batch_size: int = 1,
) -> dict:
    """Sample n bibl elements, annotate each, compute metrics vs gold standard.

    Pass *seed* to reproduce the same sample across multiple calls (e.g. when
    comparing several models on identical inputs).

    Pass *batch_size* > 1 to annotate multiple records per LLM call, which
    reduces latency at a potential quality cost.
    """
    from lxml import etree

    from tei_annotator.evaluation.extractor import extract_spans
    from tei_annotator.evaluation.metrics import MatchMode, aggregate, compute_metrics
    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.pipeline import annotate
    from tei_annotator.schemas.blbl import build_blbl_schema

    call_fn = _resolve_call_fn(provider_id, model_id, premium_key=premium_key)
    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=call_fn,
    )
    schema = build_blbl_schema()
    bibls = _load_fixture_bibls()
    rng = random.Random(seed)
    samples = rng.sample(bibls, min(n, len(bibls)))

    _parser = etree.XMLParser(recover=True)

    def _evaluate_one(el):
        plain_text, gold_spans = extract_spans(el)
        ann_result = annotate(plain_text, schema, endpoint, gliner_model=_GLINER_MODEL)
        try:
            pred_el = etree.fromstring(f"<bibl>{ann_result.xml}</bibl>".encode(), _parser)
        except Exception:
            pred_el = etree.Element("bibl")
        _, pred_spans = extract_spans(pred_el)
        return plain_text, compute_metrics(gold_spans, pred_spans, mode=MatchMode.TEXT)

    def _evaluate_batch_group(batch_els):
        """Annotate a batch of elements in a single LLM call."""
        plain_texts = []
        gold_spans_list = []
        for el in batch_els:
            pt, gs = extract_spans(el)
            plain_texts.append(pt)
            gold_spans_list.append(gs)

        if any(_BATCH_SEP in t for t in plain_texts):
            # Fallback to individual calls if separator appears in text
            return [_evaluate_one(el) for el in batch_els]

        combined = _BATCH_SEP.join(plain_texts)
        ann_result = annotate(combined, schema, endpoint, gliner_model=_GLINER_MODEL)
        pieces = ann_result.xml.split(_BATCH_SEP)

        if len(pieces) != len(batch_els):
            # Fallback: return empty predictions for all in this batch
            return [
                (pt, compute_metrics(gs, [], mode=MatchMode.TEXT))
                for pt, gs in zip(plain_texts, gold_spans_list)
            ]

        batch_results = []
        for piece, pt, gs in zip(pieces, plain_texts, gold_spans_list):
            try:
                pred_el = etree.fromstring(f"<bibl>{piece}</bibl>".encode(), _parser)
            except Exception:
                pred_el = etree.Element("bibl")
            _, pred_spans = extract_spans(pred_el)
            batch_results.append((pt, compute_metrics(gs, pred_spans, mode=MatchMode.TEXT)))
        return batch_results

    from concurrent.futures import ThreadPoolExecutor
    t0 = time.monotonic()

    if batch_size <= 1:
        with ThreadPoolExecutor(max_workers=min(len(samples), 8)) as pool:
            results = list(pool.map(_evaluate_one, samples))
    else:
        batches = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]
        with ThreadPoolExecutor(max_workers=min(len(batches), 8)) as pool:
            batch_results = list(pool.map(_evaluate_batch_group, batches))
        results = [item for group in batch_results for item in group]

    sample_texts = [r[0] for r in results]
    per_result = [r[1] for r in results]

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
async def api_config(key: str | None = None):
    """Return server configuration needed by the SPA on startup.

    Pass *key* (the PREMIUM_TOKEN) to unlock premium models in the response.
    """
    premium = bool(_PREMIUM_TOKEN and key == _PREMIUM_TOKEN)
    providers = []
    for c in get_available_connectors():
        all_models = c.models()
        visible_models = all_models if premium else c.standard_models()
        if _SELECTED_PROVIDER == c.id and _SELECTED_MODEL_ID in visible_models:
            default_model = _SELECTED_MODEL_ID
        else:
            default_model = c.default_model if c.default_model in visible_models else visible_models[0]
        providers.append({
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "models": visible_models,
            "default_model": default_model,
        })
    return {"providers": providers, "token": _API_KEY, "premium": premium}


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

    text: str | None = None
    texts: list[str] | None = None
    batch_size: int = 1
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


@app.post("/api/annotate")
async def annotate_api(
    body: AnnotateRequest,
    _: None = Depends(_verify_token),
    x_premium_key: str | None = Header(None),
):
    """
    Annotate text and return the XML result.

    - **text**: single plain text to annotate (returns a single AnnotateResponse).
    - **texts**: list of plain texts to annotate (returns a list of AnnotateResponse).
    - **batch_size**: number of texts to send in a single LLM call when using *texts*
      (default 1 — one call per text).  Values > 1 reduce latency at a potential
      quality cost ("lost in the middle" effect for large batches).
    - **provider**: connector id (default: first available provider).
    - **model**: model ID for the provider (default: provider's default model).
    - **schema**: TEI schema definition. Omit to use the built-in BLBL bibliographic schema.
    """
    import asyncio

    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.pipeline import annotate
    from tei_annotator.schemas.blbl import build_blbl_schema

    if body.text is None and body.texts is None:
        raise HTTPException(status_code=422, detail="Provide either 'text' or 'texts'.")

    call_fn = _resolve_call_fn(body.provider, body.model, premium_key=x_premium_key)

    if body.tei_schema is not None:
        schema = _schema_from_dict(body.tei_schema.model_dump())
    else:
        schema = build_blbl_schema()

    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=call_fn,
    )

    # ── Batch mode (texts list) ──────────────────────────────────────────────
    if body.texts is not None:
        try:
            t0 = time.monotonic()
            xml_list = await asyncio.to_thread(
                _batch_annotate_texts,
                body.texts,
                schema,
                endpoint,
                _GLINER_MODEL,
                body.batch_size,
            )
            elapsed = round(time.monotonic() - t0, 1)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return [
            AnnotateResponse(xml=xml, elapsed_seconds=elapsed, fuzzy_spans=[])
            for xml in xml_list
        ]

    # ── Single-text mode (backward compatible) ───────────────────────────────
    try:
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
    batch_size: int = 1


@app.post("/api/evaluate")
async def evaluate_api(
    body: EvaluateRequest,
    _: None = Depends(_verify_token),
    x_premium_key: str | None = Header(None),
):
    """
    Sample *n* bibliographic entries from the test fixture, annotate each with
    the chosen model, and return precision/recall/F1 against the gold standard.

    - **provider**: connector id (default: first available provider).
    - **model**: model ID for the provider (default: provider's default model).
    - **n**: number of random samples (default: 5).
    """
    try:
        import asyncio
        return await asyncio.to_thread(
            _run_evaluation, body.provider, body.model,
            n=body.n, seed=body.seed, premium_key=x_premium_key,
            batch_size=body.batch_size,
        )
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
