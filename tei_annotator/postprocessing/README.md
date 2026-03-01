# Post-processing

Post-processing converts the LLM's raw text output into XML tags inserted at the correct positions in the source text. It is split into four focused modules that execute in sequence:

```
LLM response (string)
       │
       ▼
 1. parse      →  list[SpanDescriptor]         (parser.py)
       │
       ▼
 2. resolve    →  list[ResolvedSpan]            (resolver.py)
       │
       ▼
 3. validate   →  list[ResolvedSpan] (filtered) (validator.py)
       │
       ▼
 4. inject     →  annotated XML string          (injector.py)
```

---

## 1. Parse (`parser.py`)

**Input:** raw string returned by `call_fn`
**Output:** `list[SpanDescriptor]`

Handles the messiness of real LLM output:

- **Fence stripping** — models often wrap JSON in markdown code fences (`` ```json … ``` ``). These are detected and removed before parsing, regardless of the language tag used.
- **JSON extraction** — the cleaned string is parsed with `json.loads`. Only a JSON array is accepted; any other top-level type raises `ValueError`.
- **Dict-to-span conversion** — each dict is validated for required keys (`element`, `text`, `context`) and coerced to a `SpanDescriptor`. Dicts with missing or non-string values are silently dropped.
- **Retry on failure** — for `TEXT_GENERATION` endpoints, if the initial parse fails the pipeline calls `make_correction_prompt()` and retries exactly once with the same `call_fn`. If the second parse also fails, the chunk is skipped with a warning.

---

## 2. Resolve (`resolver.py`)

**Input:** `list[SpanDescriptor]`, plain source text
**Output:** `list[ResolvedSpan]`

Converts context-anchored descriptors to absolute character offsets.

### Why context anchoring?

LLMs cannot reliably count characters in long strings, so asking them to return numeric offsets produces frequent errors. Instead, each span is described with its surrounding text (`SpanDescriptor.context`). The resolver searches the source text for that context string to determine where the entity lives. This approach trades a small risk of positional ambiguity (two identical context strings in different positions) for a large gain in robustness. In practice, entity text is distinctive enough that collisions are rare.

### Resolution algorithm

For each `SpanDescriptor`:

1. **Locate context** — search for `context` in the source text:
   - First, try exact substring search (`str.find`).
   - If not found, fall back to **fuzzy matching** via [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) with a default similarity threshold of 0.92. Fuzzy-matched spans receive `ResolvedSpan.fuzzy_match = True` and appear in `AnnotationResult.fuzzy_spans` for human review.
   - If neither match succeeds, the span is silently discarded.

2. **Locate text within context** — once the context window is pinned to a position in the source text, do a substring search for `SpanDescriptor.text` within that window.
   - If not found, the span is discarded.

3. **Compute offsets** — the context window's start offset plus the text's offset within the window gives `(start, end)` in the original source text.

### Fuzzy matching details

rapidfuzz's `extractOne` function scores candidate substrings using the `partial_ratio` scorer, which finds the best-matching substring of the target for the query string. The 0.92 threshold was chosen empirically to accept minor OCR errors, whitespace normalisation differences, and Unicode equivalences while rejecting clearly wrong matches.

---

## 3. Validate (`validator.py`)

**Input:** `list[ResolvedSpan]`, `TEISchema`
**Output:** `list[ResolvedSpan]` (filtered)

Three checks are applied. Spans failing any check are silently dropped and logged at `WARNING` level:

1. **Element exists** — the span's `element` must appear in the schema's element list.
2. **Attribute names** — every key in `ResolvedSpan.attributes` must be declared in the element's `TEIAttribute` list.
3. **Attribute values** — if `TEIAttribute.allowed_values` is set, the value must be one of those strings.

Additionally, spans with invalid bounds (`start < 0`, `end > len(text)`, or `start >= end`) are rejected.

Validation is a safety net against hallucinations: the LLM occasionally invents element tags or attribute names not in the schema. Dropping those silently keeps the output valid.

---

## 4. Inject (`injector.py`)

**Input:** plain source text, `list[ResolvedSpan]`
**Output:** annotated XML string

### Building the nesting tree

Spans are flat at this point (no children). The injector infers nesting geometrically: span A is a child of span B if `B.start <= A.start` and `A.end <= B.end`.

`_build_nesting_tree` implements a greedy algorithm:

1. Sort spans by start offset, then by length descending (so parents — which are longer — sort before their children).
2. For each span, find its tightest enclosing parent by scanning already-placed spans.
3. Attach it as a child of that parent, or as a root-level span if no enclosing parent exists.

**Overlapping spans** — where two spans partially overlap without one containing the other — are detected and logged as warnings. The offending span (the one with the later start position) is skipped, since overlapping XML tags are not well-formed.

### Recursive tag injection

`_inject_recursive` walks the nesting tree depth-first. At each node it emits:

1. Any source text between the previous sibling's end and this span's start.
2. The opening tag with any attributes, e.g. `<persName ref="...">`.
3. Recursively, the content of all child spans.
4. Any source text after the last child and before this span's end.
5. The closing tag, e.g. `</persName>`.

Because the injector works from pre-computed offsets on the **original source text**, the text content is never altered — only tags are inserted. This preserves the exact wording, whitespace, and punctuation of the input.
