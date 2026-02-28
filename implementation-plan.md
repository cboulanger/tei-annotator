# Implementation Plan: `tei-annotator`

Prompt:

> Design a Python library called `tei-annotator` for annotating text with TEI XML tags using a two-stage LLM pipeline. The library should:
>
> **Inputs:**
>
> - A text string to annotate (may already contain partial XML)
> - An injected `call_fn: (str) -> str` for calling an arbitrary inference endpoint
> - An `EndpointCapability` enum indicating whether the endpoint is plain text generation, JSON-constrained, or a native extraction model like GLiNER2
> - A `TEISchema` data structure describing a subset of TEI elements with descriptions, allowed attributes, and legal child elements

> The source text must never be modified by any model. Provide a package structure, all key data structures, and a step-by-step execution flow.

## Package Structure

```
tei_annotator/
├── __init__.py
├── models/
│   ├── schema.py           # TEI element/attribute data structures
│   └── spans.py            # Span manifest data structures
├── detection/
│   └── gliner_detector.py  # Optional local GLiNER first-pass span detection
├── chunking/
│   └── chunker.py          # Overlap-aware text chunker, XML-safe boundaries
├── prompting/
│   ├── builder.py          # Prompt assembly
│   └── templates/
│       ├── text_gen.jinja2         # For plain text-generation endpoints
│       └── json_enforced.jinja2    # For JSON-mode / constrained endpoints
├── inference/
│   └── endpoint.py         # Endpoint wrapper + capability enum
├── postprocessing/
│   ├── resolver.py         # Context-anchor → char offset resolution
│   ├── validator.py        # Span verification + schema validation
│   └── injector.py         # Deterministic XML construction
└── pipeline.py             # Top-level orchestration
```

---

## Data Structures (`models/`)

```python
# schema.py
@dataclass
class TEIAttribute:
    name: str                        # e.g. "ref", "type", "cert"
    description: str
    required: bool = False
    allowed_values: list[str] | None = None   # None = free string

@dataclass
class TEIElement:
    tag: str                         # e.g. "persName"
    description: str                 # from TEI Guidelines
    allowed_children: list[str]      # tags of legal child elements
    attributes: list[TEIAttribute]

@dataclass
class TEISchema:
    elements: list[TEIElement]
    # convenience lookup
    def get(self, tag: str) -> TEIElement | None: ...

# spans.py
@dataclass
class SpanDescriptor:
    element: str
    text: str
    context: str                     # must contain text as substring
    attrs: dict[str, str]
    # always flat — nesting is inferred from offset containment in resolver/injector,
    # not emitted by the model (models produce unreliable nested trees)
    confidence: float | None = None  # passed through from GLiNER

@dataclass
class ResolvedSpan:
    element: str
    start: int
    end: int
    attrs: dict[str, str]
    children: list["ResolvedSpan"]
    fuzzy_match: bool = False        # flagged for human review
```

---

## GLiNER Dependency (`detection/gliner_detector.py`)

`gliner` is a regular package dependency declared in `pyproject.toml` and installed via `uv add gliner`. No manual setup step is needed.

Model weights are fetched from HuggingFace Hub automatically on first use of `GLiNER.from_pretrained(model_id)` and cached in `~/.cache/huggingface/`. If the import fails at runtime (e.g. the optional extra was not installed), the module raises a standard `ImportError` with a clear message — no wrapper needed.

Recommended models (specified as `gliner_model` parameter):

- `urchade/gliner_medium-v2.1` — balanced, Apache 2.0
- `numind/NuNER_Zero` — stronger multi-word entities, MIT (default)
- `knowledgator/gliner-multitask-large-v0.5` — adds relation extraction

All models run on CPU; no GPU required.

---

## Endpoint Abstraction (`inference/endpoint.py`)

```python
class EndpointCapability(Enum):
    TEXT_GENERATION = "text_generation"   # plain LLM, JSON via prompt only
    JSON_ENFORCED   = "json_enforced"     # constrained decoding guaranteed
    EXTRACTION      = "extraction"        # GLiNER2/NuExtract-style native

@dataclass
class EndpointConfig:
    capability: EndpointCapability
    call_fn: Callable[[str], str]
    # call_fn signature: takes a prompt string, returns a response string
    # caller is responsible for auth, model selection, retries
```

The `call_fn` injection means the library is agnostic about whether the caller is hitting Anthropic, OpenAI, a local Ollama instance, or Fastino's GLiNER2 API. The library just hands it a string and gets a string back.

---

## Pipeline (`pipeline.py`)

```python
@dataclass
class AnnotationResult:
    xml: str                          # annotated XML string
    fuzzy_spans: list[ResolvedSpan]   # spans flagged for human review

def annotate(
    text: str,                        # may contain existing XML tags
    schema: TEISchema,                # subset of TEI elements in scope
    endpoint: EndpointConfig,         # injected inference dependency
    gliner_model: str | None = "numind/NuNER_Zero",  # None disables GLiNER pass
    chunk_size: int = 1500,           # chars
    chunk_overlap: int = 200,
) -> AnnotationResult:
```

### Execution Flow

```
1. SETUP
   strip existing XML tags from text for processing,
   preserve them as a restoration map for final merge

2. GLINER PASS  (skipped if gliner_model=None, endpoint is EXTRACTION, or text is short)
   map TEISchema elements → flat label list for GLiNER
     e.g. [("persName", "a person's name"), ("placeName", "a place name"), ...]
   chunk text if len(text) > chunk_size (with overlap)
   run gliner.predict_entities() on each chunk
   merge cross-chunk duplicates by span text + context overlap
   output: list[SpanDescriptor] with text + context + element + confidence
   (GLiNER is a pre-filter only; the LLM may reject, correct, or extend its candidates)

3. PROMPT ASSEMBLY  
   select template based on EndpointCapability:
     TEXT_GENERATION:   include JSON structure example + "output only JSON" instruction
     JSON_ENFORCED:     minimal prompt, schema enforced externally
     EXTRACTION:        pass schema directly in endpoint's native format, skip LLM prompt
   inject into prompt:
     - TEIElement descriptions + allowed attributes for in-scope elements
     - GLiNER pre-detected spans as candidates for the model to enrich/correct
     - source text chunk
     - instruction to emit one SpanDescriptor per occurrence, not per unique entity

4. INFERENCE
   call endpoint.call_fn(prompt) → raw response string

5. POSTPROCESSING  (per chunk, then merged)

   a. Parse
      JSON_ENFORCED/EXTRACTION: parse directly
      TEXT_GENERATION: strip markdown fences, parse JSON,
                       on failure: retry once with a correction prompt that includes
                       the original (bad) response and the parse error message,
                       so the model can self-correct rather than starting from scratch

   b. Resolve  (resolver.py)
      for each SpanDescriptor:
        find context string in source → exact match preferred
        find text within context window
        assert source[start:end] == span.text → reject on mismatch
        fuzzy fallback (threshold 0.92) → flag for review

   c. Validate  (validator.py)
      reject spans where text not in source
      check attributes against TEISchema allowed values
      check element is in schema scope

   d. Inject  (injector.py)
      infer nesting from offset containment (child ⊂ parent by [start, end] bounds)
      check inferred nesting: children must be within parent bounds
      sort ResolvedSpans by start offset, handle nesting depth-first
      insert tags into a copy of the original source string
      restore previously existing XML tags from step 1

   e. Final validation
      parse output as XML → reject malformed documents
      optionally validate against full TEI RelaxNG schema via lxml

6. RETURN
   AnnotationResult(
     xml=annotated_xml_string,
     fuzzy_spans=list_of_flagged_resolved_spans,
   )
```

---

## Key Design Constraints

- The source text is **never modified by any model call**. All text in the output comes from the original input; models only contribute tag positions and attributes.
- The **GLiNER pass is optional** (`gliner_model=None` disables it). It is most useful for long texts with `TEXT_GENERATION` endpoints; it is skipped automatically for `EXTRACTION` endpoints or short inputs. When enabled, GLiNER is a pre-filter only — the LLM may reject, correct, or extend its candidates.
- **Span nesting is inferred from offsets**, never emitted by the model. `SpanDescriptor` is always flat; `ResolvedSpan.children` is populated by the injector from containment relationships.
- `call_fn` has **no required signature beyond `(str) -> str`**, making it trivial to swap endpoints, add logging, or inject mock functions for testing.
- Fuzzy-matched spans are **surfaced, not silently accepted** — `AnnotationResult.fuzzy_spans` provides a reviewable list alongside the XML.

---

## Testing Strategy

### Mocking philosophy

**Always mock `call_fn` and the GLiNER detector in unit tests.** Do not use a real GLiNER model as a substitute for a remote LLM endpoint — GLiNER is a span-labelling model that cannot produce JSON responses; it cannot exercise the parse/resolve/inject pipeline. Using a real model also makes tests slow (~seconds per inference on CPU), non-deterministic across versions and hardware, and dependent on a 400MB+ download.

The `call_fn: (str) -> str` design makes mocking trivial — a lambda returning a hardcoded JSON string is sufficient. No mock framework is needed.

### Test layers

**Layer 1 — Unit tests** (always run, <1s total, fully mocked):

```
tests/
├── test_chunker.py        # chunker unit tests
├── test_resolver.py       # resolver unit tests
├── test_validator.py      # validator unit tests
├── test_injector.py       # injector unit tests
├── test_builder.py        # prompt builder unit tests
├── test_parser.py         # JSON parse + retry unit tests
└── test_pipeline.py       # full pipeline smoke test (mocked call_fn + GLiNER)
```

**Layer 2 — Integration tests** (opt-in, gated by `pytest -m integration`):

```
tests/integration/
├── test_gliner_detector.py   # real GLiNER model, real HuggingFace download
└── test_pipeline_e2e.py      # full annotate() with real GLiNER + mocked call_fn
```

Integration tests are excluded from CI by default via `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-m 'not integration'"
```

### TDD cycle (red → green → refactor)

Each module is written test-first. Write a failing test, implement the minimum code to pass, refactor.

### Key test cases per module

**`chunker.py`**

- Short text below `chunk_size` → single chunk, offset 0
- Long text → multiple chunks with correct `start_offset` per chunk
- Span exactly at a chunk boundary → appears in both chunks with correct global offset
- Input with existing XML tags → chunk boundaries never split a tag

**`resolver.py`**

- Exact context match → `ResolvedSpan` with `fuzzy_match=False`
- Context not found in source → span rejected
- `source[start:end] != span.text` → span rejected
- Context found but span text not within it → span rejected
- Context found, span text found, score < 0.92 → span rejected
- Context found, span text found, 0.92 ≤ score < 1.0 → `fuzzy_match=True`
- Multiple occurrences of context → first match used, or rejection if ambiguous

**`validator.py`**

- Element not in schema → span rejected
- Attribute not in schema → span rejected
- Attribute value not in `allowed_values` → span rejected
- Valid span → passes through unchanged

**`injector.py`**

- Two non-overlapping spans → both tags inserted correctly
- Span B offset-contained in span A → B is child of A in output
- Overlapping (non-nesting) spans → reject or flatten with warning
- Restored XML tags from step 1 do not conflict with injected tags

**`builder.py`**

- `TEXT_GENERATION` capability → prompt contains JSON example and "output only JSON" instruction
- `JSON_ENFORCED` capability → prompt is minimal (no JSON scaffolding)
- GLiNER candidates present → candidates appear in prompt
- GLiNER candidates absent (pass skipped) → prompt has no candidate section

**`parser.py` (parse + retry logic)**

- Valid JSON response → parsed to `list[SpanDescriptor]` without retry
- Markdown-fenced JSON → fences stripped, parsed correctly
- Invalid JSON on first attempt → retry triggered with correction prompt that includes original bad response + parse error message
- Invalid JSON on second attempt → exception raised, chunk skipped

**`pipeline.py` (smoke test)**

```python
def mock_call_fn(prompt: str) -> str:
    return json.dumps([
        {"element": "persName", "text": "John Smith",
         "context": "...said John Smith yesterday...", "attrs": {}}
    ])

def test_annotate_smoke():
    schema = TEISchema(elements=[
        TEIElement(tag="persName", description="a person's name",
                   allowed_children=[], attributes=[])
    ])
    endpoint = EndpointConfig(
        capability=EndpointCapability.JSON_ENFORCED,
        call_fn=mock_call_fn,
    )
    result = annotate(
        text="He said John Smith yesterday.",
        schema=schema,
        endpoint=endpoint,
        gliner_model=None,   # disable GLiNER in unit tests
    )
    assert "persName" in result.xml
    assert "John Smith" in result.xml
    assert result.xml.count("John Smith") == 1   # text not duplicated
```

---

## Implementation Status

**Completed 2026-02-28** — full implementation per the plan above.

### What was built

All modules in the package structure were implemented:

| File | Notes |
| --- | --- |
| `tei_annotator/models/schema.py` | `TEIAttribute`, `TEIElement`, `TEISchema` dataclasses |
| `tei_annotator/models/spans.py` | `SpanDescriptor`, `ResolvedSpan` dataclasses |
| `tei_annotator/inference/endpoint.py` | `EndpointCapability` enum, `EndpointConfig` dataclass |
| `tei_annotator/chunking/chunker.py` | `chunk_text()` — overlap chunker, XML-safe boundaries |
| `tei_annotator/detection/gliner_detector.py` | `detect_spans()` — optional, raises `ImportError` if `[gliner]` extra not installed |
| `tei_annotator/prompting/builder.py` | `build_prompt()` + `make_correction_prompt()` |
| `tei_annotator/prompting/templates/text_gen.jinja2` | Verbose prompt with JSON example, "output only JSON" instruction |
| `tei_annotator/prompting/templates/json_enforced.jinja2` | Minimal prompt for constrained-decoding endpoints |
| `tei_annotator/postprocessing/parser.py` | `parse_response()` — fence stripping, one-shot self-correction retry |
| `tei_annotator/postprocessing/resolver.py` | `resolve_spans()` — context-anchor → char offset, rapidfuzz fuzzy fallback at threshold 0.92 |
| `tei_annotator/postprocessing/validator.py` | `validate_spans()` — element, attribute name, allowed-value checks |
| `tei_annotator/postprocessing/injector.py` | `inject_xml()` — stack-based nesting tree, recursive tag insertion |
| `tei_annotator/pipeline.py` | `annotate()` — full orchestration, tag strip/restore, deduplication across chunks, lxml final validation |

### Dependencies added

Runtime: `jinja2`, `lxml`, `rapidfuzz`. Optional extra `[gliner]` for GLiNER support. Dev: `pytest`, `pytest-cov`.

### Tests

- **63 unit tests** (Layer 1) — fully mocked, run in < 0.1 s via `uv run pytest`
- **9 integration tests** (Layer 2, no GLiNER) — complex resolver/injector/pipeline scenarios, run via `uv run pytest --override-ini="addopts=" -m integration tests/integration/test_pipeline_e2e.py -k "not real_gliner"`
- **1 GLiNER integration test** — requires `[gliner]` extra and HuggingFace model download

### Smoke script

`scripts/smoke_test_llm.py` — end-to-end test with real LLM calls (no GLiNER). Verified against:

- **Google Gemini 2.0 Flash** (`GEMINI_API_KEY` from `.env`)
- **KISSKI `llama-3.3-70b-instruct`** (`KISSKI_API_KEY` from `.env`, OpenAI-compatible API at `https://chat-ai.academiccloud.de/v1`)

Run with `uv run scripts/smoke_test_llm.py`.

### Key implementation notes

- The `_strip_existing_tags` / `_restore_existing_tags` pair in `pipeline.py` preserves original markup by tracking plain-text offsets of each stripped tag and re-inserting them after annotation.
- `_build_nesting_tree` in `injector.py` uses a sort-by-(start-asc, length-desc) + stack algorithm; partial overlaps are dropped with a `warnings.warn`.
- The resolver does an exact `str.find` first; fuzzy search (sliding-window rapidfuzz) is only attempted if exact fails and rapidfuzz is installed.
- `parse_response` passes `call_fn` and `make_correction_prompt` only for `TEXT_GENERATION` endpoints; `JSON_ENFORCED` and `EXTRACTION` never retry.
