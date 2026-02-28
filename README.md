# tei-annotator

A Python library for annotating text with [TEI XML](https://tei-c.org/) tags using a two-stage LLM pipeline.

The pipeline:

1. **(Optional) GLiNER pre-detection** — fast CPU-based span labelling generates candidates for the LLM to verify and extend.
2. **LLM annotation** — a prompted language model identifies entities, returns structured spans (element + verbatim text + surrounding context + attributes).
3. **Deterministic post-processing** — spans are resolved to character offsets, validated against the schema, and injected as XML tags. The source text is **never modified** by any model call.

Works with any inference endpoint through an injected `call_fn: (str) -> str` — Anthropic, OpenAI, Gemini, a local Ollama instance, or a constrained-decoding API.

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
