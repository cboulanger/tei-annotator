# Implementation Plan: `tei-annotator`

Prompt:

> Design a Python library called `tei-annotator` for annotating text with TEI XML tags using a two-stage LLM pipeline. The library should:
>
> **Inputs:**
> - A text string to annotate (may already contain partial XML)
> - An injected `call_fn: (str) -> str` for calling an arbitrary inference endpoint
> - An `EndpointCapability` enum indicating whether the endpoint is plain text generation, JSON-constrained, or a native extraction model like GLiNER2
> - A `TEISchema` data structure describing a subset of TEI elements with descriptions, allowed attributes, and legal child elements
>
> **Pipeline:**
> 1. Check for local GLiNER installation and print setup instructions if missing
> 2. Run a local GLiNER model as a first-pass span detector, mapping TEI element descriptions to GLiNER labels
> 3. Chunk long texts with overlap, tracking global character offsets
> 4. Assemble a prompt using the GLiNER candidates, TEI schema context, and JSON output instructions tailored to the endpoint capability
> 5. Parse and validate the returned span manifest — each span has `element`, `text`, `context` (surrounding text for position resolution), and `attrs`
> 6. Resolve spans to character offsets by searching for the context string in the source, then locating the span text within it — reject any span where `source[start:end] != span.text`
> 7. Inject tags deterministically into the original source text, handling nesting
> 8. Return the annotated XML plus a list of fuzzy-matched spans flagged for human review
>
> The source text must never be modified by any model. Provide a package structure, all key data structures, and a step-by-step execution flow.

## Package Structure

```
tei_annotator/
├── __init__.py
├── models/
│   ├── schema.py           # TEI element/attribute data structures
│   └── spans.py            # Span manifest data structures
├── detection/
│   ├── gliner_detector.py  # Local GLiNER first-pass span detection
│   └── setup.py            # GLiNER availability check + install instructions
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
    children: list["SpanDescriptor"] # for nested annotations
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

## GLiNER Setup (`detection/setup.py`)

```python
def check_gliner() -> bool:
    try:
        import gliner
        return True
    except ImportError:
        print("""
GLiNER is not installed. To install:

    pip install gliner

Recommended models (downloaded automatically on first use):
    urchade/gliner_medium-v2.1       # balanced, Apache 2.0
    numind/NuNER_Zero                # stronger multi-word entities, MIT
    knowledgator/gliner-multitask-large-v0.5  # adds relation extraction

Models run on CPU; no GPU required.
        """)
        return False
```

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
def annotate(
    text: str,                        # may contain existing XML tags
    schema: TEISchema,                # subset of TEI elements in scope
    endpoint: EndpointConfig,         # injected inference dependency
    gliner_model: str = "numind/NuNER_Zero",
    chunk_size: int = 1500,           # chars
    chunk_overlap: int = 200,
) -> str:                             # returns annotated XML string
```

### Execution Flow

```
1. SETUP
   check_gliner() → prompt user to install if missing
   strip existing XML tags from text for processing,
   preserve them as a restoration map for final merge

2. GLINER PASS
   map TEISchema elements → flat label list for GLiNER
     e.g. [("persName", "a person's name"), ("placeName", "a place name"), ...]
   chunk text if len(text) > chunk_size (with overlap)
   run gliner.predict_entities() on each chunk
   merge cross-chunk duplicates by span text + context overlap
   output: list[SpanDescriptor] with text + context + element + confidence

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
                       retry once with correction prompt on failure

   b. Resolve  (resolver.py)
      for each SpanDescriptor:
        find context string in source → exact match preferred
        find text within context window
        assert source[start:end] == span.text → reject on mismatch
        fuzzy fallback (threshold 0.92) → flag for review

   c. Validate  (validator.py)
      reject spans where text not in source
      check nesting: children must be within parent bounds
      check attributes against TEISchema allowed values
      check element is in schema scope

   d. Inject  (injector.py)
      sort ResolvedSpans by start offset, handle nesting depth-first
      insert tags into a copy of the original source string
      restore previously existing XML tags from step 1

   e. Final validation
      parse output as XML → reject malformed documents
      optionally validate against full TEI RelaxNG schema via lxml

6. RETURN
   annotated XML string
   + list of fuzzy-matched spans flagged for human review
```

---

## Key Design Constraints

- The source text is **never modified by any model call**. All text in the output comes from the original input; models only contribute tag positions and attributes.
- GLiNER is a **pre-filter**, not the authority. The LLM can reject, correct, or add to its candidates. GLiNER's value is positional reliability; the LLM's value is schema reasoning.
- `call_fn` has **no required signature beyond `(str) -> str`**, making it trivial to swap endpoints, add logging, or inject mock functions for testing.
- Fuzzy-matched spans are **surfaced, not silently accepted** — the return value includes a reviewable list alongside the XML.