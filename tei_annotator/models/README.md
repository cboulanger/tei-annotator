# Data models

Two groups of data classes power the pipeline: **schema models** (what the annotator is allowed to produce) and **span models** (the annotations as they flow through each stage).

---

## Schema models (`schema.py`)

### `TEIAttribute`

Describes a single XML attribute that may appear on a `TEIElement`.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Attribute name, e.g. `"ref"` |
| `description` | `str` | Human-readable explanation included in LLM prompts |
| `required` | `bool` | Whether the attribute must be present (informational only — not enforced by the pipeline) |
| `allowed_values` | `list[str] \| None` | If set, the validator rejects any value not in this list |

### `TEIElement`

Describes one XML element the annotator may produce.

| Field | Type | Description |
|-------|------|-------------|
| `tag` | `str` | XML tag name, e.g. `"persName"` |
| `description` | `str` | Human-readable explanation included in LLM prompts |
| `children` | `list[TEIElement]` | Allowed child elements (enables nested annotation) |
| `attributes` | `list[TEIAttribute]` | Allowed attributes |

Elements may nest: a `biblStruct` can contain `author`, which can contain `persName`. The injector uses the children hierarchy to build valid nesting trees.

### `TEISchema`

A flat container of `TEIElement` objects with a `get(tag) -> TEIElement | None` method for O(1) lookup by tag name.

**Building a schema programmatically:**

```python
from tei_annotator import TEISchema, TEIElement, TEIAttribute

schema = TEISchema(elements=[
    TEIElement(
        tag="persName",
        description="a person's name",
        attributes=[TEIAttribute(name="ref", description="authority URI")],
    ),
    TEIElement(tag="placeName", description="a geographical place name"),
])
```

**Building from a RELAX NG file** (see [`tei.py`](../tei.py)):

```python
from tei_annotator import create_schema

schema = create_schema("schema/tei-bib.rng", element="biblStruct", depth=1)
```

`create_schema` walks the RNG content model breadth-first to `depth` levels, collecting allowed child elements and their attribute definitions automatically.

---

## Span models (`spans.py`)

### `SpanDescriptor`

Produced by the LLM (via the parser) or by the GLiNER detector. A `SpanDescriptor` is **context-anchored**: instead of character offsets (which LLMs count unreliably), it stores the surrounding text around the entity. The resolver later searches the source text for `context` to determine where `text` lives.

| Field | Type | Description |
|-------|------|-------------|
| `element` | `str` | TEI tag to apply, e.g. `"persName"` |
| `text` | `str` | Verbatim text of the entity |
| `context` | `str` | Surrounding text used to locate `text` in the source |
| `attributes` | `dict[str, str]` | Attribute key/value pairs |
| `score` | `float \| None` | Confidence score (populated by GLiNER; LLM-produced spans use `None`) |

All `SpanDescriptor` objects are **flat** — they carry no children. Nesting is inferred geometrically by the injector once offsets are known.

### `ResolvedSpan`

The output of the resolver and the input to the injector. Offsets are absolute positions in the plain (tag-stripped) source text.

| Field | Type | Description |
|-------|------|-------------|
| `element` | `str` | TEI tag |
| `start` | `int` | Start character offset (inclusive) |
| `end` | `int` | End character offset (exclusive) |
| `attributes` | `dict[str, str]` | Attribute key/value pairs |
| `children` | `list[ResolvedSpan]` | Nested child spans (populated by the injector via offset containment) |
| `fuzzy_match` | `bool` | `True` if the context was located by fuzzy rather than exact matching |

Span A is a child of span B if `B.start <= A.start` and `A.end <= B.end`. The injector constructs this nesting tree from the flat list of `ResolvedSpan` objects.
