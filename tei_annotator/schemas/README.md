# Schemas

Annotation schema definitions and the registry that maps schema names to their build functions and evaluation parameters.

---

## Built-in schemas

| Key | Module | Task | `root_element` | `child_element` |
| --- | --- | --- | --- | --- |
| `bibl` | `bibl.py` | Tag internal fields of a bibliographic reference (author, title, date, publisher, …) | `listBibl` | `bibl` |
| `bibl-reference-segmenter` | `bibl_reference_segmenter.py` | Segment a reference list into `<bibl>` spans with optional `<label>` | `text` | `listBibl` |

Each schema ships with at least one gold-standard corpus file at `data/corpus/<schema>.default.tei.xml` used by `scripts/evaluate_llm.py` and the webservice.

---

## Registry (`registry.py`)

```python
from tei_annotator.schemas.registry import get_schema_names, get_schema_config, build_schema

print(get_schema_names())               # ['bibl', 'bibl-reference-segmenter']
cfg = get_schema_config("bibl")         # {'build': ..., 'root_element': 'listBibl', 'child_element': 'bibl'}
schema = build_schema("bibl")           # TEISchema instance
```

`get_schema_config` returns the raw registry entry (useful to get `root_element` / `child_element` for `evaluate_file`). `build_schema` calls `entry["build"]()` and returns the `TEISchema`.

---

## Adding a new schema

1. Create `tei_annotator/schemas/myschema.py` with a `build_myschema()` function that returns a `TEISchema`.
2. Register it in `SCHEMA_REGISTRY` in `registry.py`:

```python
from tei_annotator.schemas.registry import SCHEMA_REGISTRY

SCHEMA_REGISTRY["myschema"] = {
    "build": lambda: __import__(
        "tei_annotator.schemas.myschema", fromlist=["build_myschema"]
    ).build_myschema(),
    "root_element": "body",      # container element in your gold XML
    "child_element": "div",      # record element to annotate
}
```

Or more cleanly, add it directly in `registry.py` alongside the existing entries.

3. Create a gold-standard fixture at `tests/fixtures/myschema-examples.tei.xml` to enable auto-detected evaluation:

```bash
uv run scripts/evaluate_llm.py --schema myschema --provider gemini --max-items 5 --verbose
```

---

## Writing effective element descriptions

The `TEIElement.description` string is the primary signal the LLM uses to decide what to annotate and where boundaries lie. See [docs/tei-element-descriptions.md](../../docs/tei-element-descriptions.md) for evidence-based guidelines.
