# tei-annotator

A Python library for annotating text with [TEI XML](https://tei-c.org/) tags using a two-stage LLM pipeline.

The pipeline:

1. **(Optional) GLiNER pre-detection** — fast CPU-based span labelling generates candidates for the LLM to verify and extend.
2. **LLM annotation** — a prompted language model identifies entities, returns structured spans (element + verbatim text + surrounding context + attributes).
3. **Deterministic post-processing** — spans are resolved to character offsets, validated against the schema, and injected as XML tags. The source text is **never modified** by any model call.

Works with any inference endpoint through an injected `call_fn: (str) -> str` — Anthropic, OpenAI, Gemini, a local Ollama instance, or a constrained-decoding API.

---

## Pipeline diagram

```text
  Input text (may contain XML markup)
               │
               ▼
  ┌────────────────────────────────────┐
  │  Strip existing XML tags           │  pipeline.py
  └──────────────┬─────────────────────┘
                 │
                 ▼  (optional)
  ┌────────────────────────────────────┐
  │  GLiNER pre-detection              │  detection/
  └──────────────┬─────────────────────┘
                 │
                 ▼
  ┌────────────────────────────────────┐
  │  Chunk text                        │  chunking/
  └──────────────┬─────────────────────┘
                 │
          ╔══════╧══════╗
          ║  per chunk  ║
          ╚══════╤══════╝
                 │
                 ▼
  ┌────────────────────────────────────┐
  │  Build LLM prompt                  │  prompting/
  └──────────────┬─────────────────────┘
                 │
                 ▼
  ┌────────────────────────────────────┐
  │  LLM inference                     │  inference/
  └──────────────┬─────────────────────┘
                 │
                 ▼
  ┌────────────────────────────────────┐
  │  Parse JSON response               │  postprocessing/
  └──────────────┬─────────────────────┘
                 │
        ╔════════╧════════╗
        ║ merge all chunks ║
        ╚════════╤════════╝
                 │
                 ▼
  ┌────────────────────────────────────┐
  │  Resolve spans → char offsets      │  postprocessing/
  ├────────────────────────────────────┤
  │  Validate against schema           │  postprocessing/
  ├────────────────────────────────────┤
  │  Inject XML tags                   │  postprocessing/
  └──────────────┬─────────────────────┘
                 │
                 ▼
  Annotated XML output
```

Detailed documentation for each stage:
[Data models](tei_annotator/models/README.md) ·
[GLiNER detection](tei_annotator/detection/README.md) ·
[Chunking](tei_annotator/chunking/README.md) ·
[Prompt building](tei_annotator/prompting/README.md) ·
[Inference configuration](tei_annotator/inference/README.md) ·
[Post-processing](tei_annotator/postprocessing/README.md) ·
[Evaluation](tei_annotator/evaluation/README.md)

---

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd tei-annotator
uv sync                        # installs runtime deps: jinja2, lxml, rapidfuzz
uv sync --extra gliner         # also installs gliner for the optional pre-detection pass
```

API keys for real LLM endpoints go in `.env` (see `.env` for the expected variable names).

---

## Quick example

```python
from tei_annotator import (
    annotate,
    TEISchema, TEIElement, TEIAttribute,
    EndpointConfig, EndpointCapability,
)

# 1. Describe the elements you want to annotate
schema = TEISchema(elements=[
    TEIElement(
        tag="persName",
        description="a person's name",
        attributes=[TEIAttribute(name="ref", description="authority URI")],
    ),
    TEIElement(
        tag="placeName",
        description="a geographical place name",
        attributes=[],
    ),
])

# 2. Wrap your inference endpoint
def my_call_fn(prompt: str) -> str:
    # replace with any LLM call — Anthropic, OpenAI, Gemini, Ollama, …
    ...

endpoint = EndpointConfig(
    capability=EndpointCapability.TEXT_GENERATION,
    call_fn=my_call_fn,
)

# 3. Annotate
result = annotate(
    text="Marie Curie was born in Warsaw and later worked in Paris.",
    schema=schema,
    endpoint=endpoint,
    gliner_model=None,   # set to e.g. "numind/NuNER_Zero" to enable pre-detection
)

print(result.xml)
# <persName>Marie Curie</persName> was born in <placeName>Warsaw</placeName>
# and later worked in <placeName>Paris</placeName>.

if result.fuzzy_spans:
    print("Review these spans — context was matched approximately:")
    for span in result.fuzzy_spans:
        print(f"  <{span.element}>{span.text}</{span.element}>")
```

The input text may already contain XML markup; existing tags are stripped before the LLM sees the text and restored in the final output.

### Real-endpoint smoke test

`scripts/smoke_test_llm.py` runs the full pipeline against **Gemini 2.0 Flash** and **KISSKI `llama-3.3-70b-instruct`** using API keys from `.env`:

```bash
uv run scripts/smoke_test_llm.py
```

---

## `annotate()` parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `text` | — | Input text; may contain existing XML tags |
| `schema` | — | `TEISchema` describing elements and attributes in scope |
| `endpoint` | — | `EndpointConfig` wrapping any `call_fn: (str) -> str` |
| `gliner_model` | `"numind/NuNER_Zero"` | HuggingFace model for optional pre-detection; `None` to disable |
| `chunk_size` | `1500` | Maximum characters per LLM prompt chunk |
| `chunk_overlap` | `200` | Character overlap between consecutive chunks |

### `EndpointCapability` values

| Value | When to use |
| --- | --- |
| `TEXT_GENERATION` | Plain LLM — JSON requested via prompt, with one automatic retry on parse failure |
| `JSON_ENFORCED` | Constrained-decoding endpoint that guarantees valid JSON output |
| `EXTRACTION` | Native extraction model (GLiNER2 / NuExtract-style); raw text is passed directly |

---

## Evaluation

The `tei_annotator.evaluation` module compares annotator output against a manually annotated gold-standard TEI XML file to compute **precision**, **recall**, and **F1 score**.

### How it works

For each gold-standard element (e.g. `<bibl>`):

1. Extract gold spans — walk the element tree and record `(tag, start, end, text)` for every descendant element, using character offsets in the element's plain text.
2. Strip all tags → plain text (identical to what the annotator will see).
3. Run `annotate()` on the plain text.
4. Extract predicted spans from the annotated XML output.
5. Greedily match predicted spans against gold spans; compute TP / FP / FN.

Because the annotator receives *exactly the same plain text* that the gold spans are anchored to, character offsets align without any additional normalisation.

### Match modes

| Mode | A match if… |
| --- | --- |
| `TEXT` (default) | same element tag + normalised text content |
| `EXACT` | same element tag + identical `(start, end)` offsets |
| `OVERLAP` | same element tag + intersection-over-union ≥ threshold (default 0.5) |

### Quick example

```python
from tei_annotator import create_schema, EndpointConfig, EndpointCapability
from tei_annotator.evaluation import evaluate_file, MatchMode

schema = create_schema("schema/tei-bib.rng", element="biblStruct", depth=1)

endpoint = EndpointConfig(
    capability=EndpointCapability.TEXT_GENERATION,
    call_fn=my_call_fn,
)

per_record, overall = evaluate_file(
    gold_xml_path="tests/fixtures/blbl-examples.tei.xml",
    schema=schema,
    endpoint=endpoint,
    match_mode=MatchMode.TEXT,
    max_items=10,   # optional: evaluate only the first N records
)

print(overall.report())
# === Evaluation Results ===
# Micro  P=0.821  R=0.754  F1=0.786  (TP=83  FP=18  FN=27)
# Macro  P=0.834  R=0.762  F1=0.791
#
# Per-element breakdown:
#   author               P=0.923  R=0.960  F1=0.941  (TP=24  FP=2  FN=1)
#   biblScope            P=0.750  R=0.600  F1=0.667  (TP=12  FP=4  FN=8)
#   date                 P=0.867  R=0.867  F1=0.867  (TP=13  FP=2  FN=2)
#   ...
```

`EvaluationResult` exposes both **micro-averaged** metrics (aggregate TP/FP/FN counts, then compute rates — correct for imbalanced element-type distributions) and **macro-averaged** metrics (average per-element rates — weights all types equally). The full span lists (`matched`, `unmatched_gold`, `unmatched_pred`) are available for detailed inspection.

---

## Testing

```bash
# Unit tests (fully mocked, < 0.1 s)
uv run pytest

# Integration tests — complex pipeline scenarios, no model download needed
uv run pytest --override-ini="addopts=" -m integration \
    tests/integration/test_pipeline_e2e.py -k "not real_gliner"

# Integration tests — real GLiNER model (downloads ~400 MB on first run)
uv run pytest --override-ini="addopts=" -m integration \
    tests/integration/test_gliner_detector.py \
    tests/integration/test_pipeline_e2e.py::test_pipeline_with_real_gliner
```

Integration tests are excluded from the default `pytest` run via `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-m 'not integration'"
```
