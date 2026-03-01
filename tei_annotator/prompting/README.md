# Prompt building

The prompt builder (`builder.py`) constructs the full instruction string sent to the LLM for each text chunk. It uses [Jinja2](https://jinja.palletsprojects.com/) templates so that the prompt structure can be read and modified independently of the Python code.

---

## Templates

Two templates live in `templates/`, one per `EndpointCapability`:

### `text_gen.jinja2` — for `TEXT_GENERATION` endpoints

Used with standard chat/completion LLMs. The template includes:

1. **Role preamble** — declares the model's role as a TEI XML annotation assistant.
2. **Schema description** — for each `TEIElement`: tag name, description, allowed attributes (with descriptions and allowed values if constrained), and allowed child elements.
3. **Pre-detected candidates** (if any) — rendered as a JSON array so the LLM sees what GLiNER found and can decide whether to confirm, correct, or discard each candidate.
4. **Source text** — the chunk to annotate, presented verbatim.
5. **Output instructions** — instructs the model to return a JSON array of objects with no prose, markdown, or explanation:

```json
[
  {
    "element": "persName",
    "text": "Marie Curie",
    "context": "scientist Marie Curie was born in",
    "attrs": {"ref": "https://viaf.org/viaf/36924049/"}
  }
]
```

### `json_enforced.jinja2` — for `JSON_ENFORCED` endpoints

A compact variant for constrained-decoding endpoints (e.g. vLLM with guided JSON). Verbose explanations are omitted because the endpoint guarantees syntactically valid JSON output; less hand-holding is needed. The expected JSON structure is identical.

---

## The `context` field: why it exists

LLMs cannot reliably count characters in long strings, so asking them to return numeric offsets produces errors. Instead, each span is described by its surrounding text (`context`). The resolver later anchors each span to the source text by searching for `context` as a substring (exact match first, then fuzzy). This makes the pipeline robust to small model errors in character arithmetic.

The prompt instructs the model to copy a ~20-40 character window around the entity verbatim from the source text — enough to uniquely identify the span in most real-world texts.

---

## Self-correction / retry

When a `TEXT_GENERATION` response cannot be parsed as JSON, `make_correction_prompt()` constructs a follow-up prompt that:

1. Quotes the malformed response.
2. States the parse error.
3. Asks the model to return only a corrected JSON array, nothing else.

The pipeline sends this correction prompt to the same `call_fn`. If the corrected response also fails to parse, the chunk is skipped with a warning — one retry is the limit.

---

## Jinja2 environment

`_get_env()` initialises a `jinja2.Environment` with a custom `tojson` filter (a thin wrapper around `json.dumps`). This is necessary because `tojson` is provided automatically only in Flask applications; without it, serialising Python objects to JSON inside templates would fail.

---

## Prompt size considerations

Each prompt is built per chunk. With `chunk_size=1500` characters the schema preamble typically adds 200–500 characters depending on the number of elements and attributes. For very large schemas (many elements, many attributes with long descriptions), consider:

- Reducing `chunk_size` to keep total prompt length within model limits.
- Splitting the schema into focused subsets and running multiple `annotate()` passes, one per element group.
