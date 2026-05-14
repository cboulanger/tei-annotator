# TEI Annotator — Webservice

FastAPI JSON API with a browser UI. Supports multiple LLM providers via a connector system — providers are enabled automatically based on which API keys are present in `.env`.

---

## Running

```bash
uv sync --extra webservice
cp .env.template .env
# edit .env — set at least one provider API key (see Providers below)
uv run task webservice          # reads HOST / PORT from .env
uv run python webservice/main.py --reload  # development mode with auto-reload
```

> **Note:** Do not start with `uvicorn main:app` directly — uvicorn binds the port from its CLI args *before* the module is imported, so `load_dotenv()` would run too late to affect the port. Running via `python main.py` also handles automatic cleanup of any previously running instance on the same port.

The default port is `8099` (configurable via `PORT` in `.env`).

For deploying behind nginx with HTTPS and as a systemd service, see [`nginx.conf`](nginx.conf) and [`tei-annotator.service`](tei-annotator.service).

---

## Providers

Providers are enabled based on which environment variables are set. Configure them in `.env` (copy from `.env.template`):

| Provider | Env var | Notes |
| --- | --- | --- |
| **HuggingFace Inference Router** | `HF_TOKEN` | Open models via router.huggingface.co |
| **Google Gemini** | `GEMINI_API_KEY` | Gemini models via the generateContent REST API |
| **OpenAI** | `OPENAI_API_KEY` | GPT models via the chat completions API |
| **Anthropic Claude** | `ANTHROPIC_API_KEY` | Claude models via the Messages API |
| **KISSKI** | `KISSKI_API_KEY` | Academic cloud at chat-ai.academiccloud.de; model list fetched dynamically; override base URL with `KISSKI_BASE_URL` |

Set `SELECTED_MODEL=<provider>/<model>` to pre-select a specific model (e.g. `SELECTED_MODEL=gemini/gemini-2.5-flash`). Omit to use each provider's first standard model.

At least one key must be set. The browser UI shows all available providers grouped in the model dropdown; if no key is configured a setup notice is shown instead.

### Standard vs premium models

Each connector declares some models as *premium* by prefixing the model name with `*` in its `_MODELS` list (see `tei_annotator/providers/`). Standard models are shown to all users; premium models are hidden unless the visitor holds the `PREMIUM_TOKEN` (see [Security](#security) below). The `*` is stripped before the model ID is passed to any API.

### Adding a new provider

Connectors live in `tei_annotator/providers/` (one file per provider). To add one:

1. Create `tei_annotator/providers/myprovider.py` — subclass `Connector` from `.base`.
2. Append an instance to `_ALL_CONNECTORS` in `tei_annotator/providers/__init__.py`.

Required interface: `id`, `name`, `description`, `is_available()`, `_MODELS`, `make_call_fn(model_id, timeout)`. Prefix a model name with `*` in `_MODELS` to mark it as premium. Override `models()` / `standard_models()` only for dynamic model lists (see `KISSKIConnector`).

See [tei_annotator/providers/README.md](../tei_annotator/providers/README.md) for a full walkthrough.

---

## Evaluation corpora

Gold-standard corpus files live in `data/corpus/` at the repository root and are tracked by git. The `data/raw/` subdirectory (gitignored) holds large raw source batches used as input to `scripts/collect_hard_examples.py`.

### File naming

```
data/corpus/<schema-id>.<label>.tei.xml
```

| File | Schema | Label |
| --- | --- | --- |
| `bibl.default.tei.xml` | `bibl` | `default` |
| `bibl-reference-segmenter.default.tei.xml` | `bibl-reference-segmenter` | `default` |
| `bibl-reference-segmenter.hard-cases.tei.xml` | `bibl-reference-segmenter` | `hard-cases` |

The webservice discovers available corpora automatically by globbing `<schema-id>.*.tei.xml` inside the corpus directory. Add new files matching the pattern and they appear in the UI immediately.

Override the corpus directory for custom deployments:

```
CORPUS_DIR=/path/to/my/corpora   # in .env
```

---

## Security

### API key (`API_KEY`)

Set `API_KEY` in `.env` to require all callers to present `Authorization: Bearer <key>`. The key is returned by `/api/config` and injected into the browser automatically, so regular UI users are unaffected. Leave empty for open access (local development).

### Premium key (`PREMIUM_TOKEN`)

Set `PREMIUM_TOKEN` in `.env` to gate expensive models behind a second secret. Share the URL `https://your-domain/?key=<secret>` with trusted users. The key is:

- stored in `sessionStorage` (persists within the tab, not across new tabs)
- sent as `X-Premium-Key` on every API call
- enforced server-side: requests for premium models without the correct key receive HTTP 403

Knowing only the `API_KEY` is not sufficient to call premium models.

Generate both keys with:

```bash
python -c "import secrets; print(secrets.token_hex(24))"
```

---

## API endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | `GET` | Single-page browser UI |
| `/api/config` | `GET` | Available providers, schemas, corpora, and tokens |
| `/api/annotate` | `POST` (JSON) | Annotate text, return XML |
| `/api/evaluate` | `POST` (JSON) | Run evaluation against a gold-standard corpus |
| `/api/sample` | `GET` | Sample plain-text entries from a corpus file |
| `/docs` | `GET` | Interactive OpenAPI documentation (Swagger UI) |

### `GET /api/config?key=<premium_token>`

```json
{
  "providers": [
    {
      "id": "gemini",
      "name": "Google Gemini",
      "description": "...",
      "models": ["gemini-2.0-flash-lite", "gemini-2.5-flash"],
      "default_model": "gemini-2.0-flash-lite"
    }
  ],
  "schemas": [
    {
      "id": "bibl",
      "default_corpus": "default",
      "corpora": ["default"]
    },
    {
      "id": "bibl-reference-segmenter",
      "default_corpus": "default",
      "corpora": ["default", "hard-cases"]
    }
  ],
  "token": "<api_key_or_null>",
  "premium": false
}
```

Pass `?key=<PREMIUM_TOKEN>` to receive the full model list and `"premium": true`.

### `POST /api/annotate`

#### Request body

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | `string` | — | Single plain text to annotate. Returns a single response object. |
| `texts` | `string[]` | — | List of plain texts to annotate (batch mode). Returns an array of response objects. |
| `batch_size` | `integer` | `1` | Number of texts to send in one LLM call when using `texts`. |
| `provider` | `string` | first available | Connector id (e.g. `"gemini"`, `"openai"`). |
| `model` | `string` | provider default | Model ID for the chosen provider. |
| `schema_id` | `string` | `"bibl"` | Registered schema name. Takes precedence over inline `schema`. |
| `schema` | `object` | — | Custom TEI schema (see below). Used only when `schema_id` is absent. |

Exactly one of `text` or `texts` must be provided.

**Custom `schema` object:**

```json
{
  "elements": [
    {
      "tag": "author",
      "description": "Author of the work",
      "allowed_children": [],
      "attributes": [
        {
          "name": "role",
          "description": "Author role",
          "allowed_values": ["editor", "translator"]
        }
      ]
    }
  ],
  "rules": ["author must precede title"]
}
```

**Single-text example:**

```json
{
  "text": "Doe, J. (2024). A paper. Journal of Foo, 12(3), 1–10.",
  "provider": "gemini",
  "model": "gemini-2.5-flash",
  "schema_id": "bibl"
}
```

#### Response

When `text` is used, a single object is returned. When `texts` is used, an array of objects is returned (one per input text, in the same order).

| Field | Type | Description |
| --- | --- | --- |
| `xml` | `string` | Annotated XML fragment for the input text. |
| `fuzzy_spans` | `object[]` | Spans resolved via fuzzy matching (see below). |
| `elapsed_seconds` | `float` | Wall-clock time for the annotation call. |

Each `fuzzy_spans` entry:

| Field | Type | Description |
| --- | --- | --- |
| `element` | `string` | TEI element tag name. |
| `start` | `integer` | Start character offset in the original plain text. |
| `end` | `integer` | End character offset in the original plain text. |

**Single-text response example:**

```json
{
  "xml": "<author>Doe, J.</author> (<date>2024</date>). <title>A paper</title>. ...",
  "fuzzy_spans": [
    { "element": "author", "start": 0, "end": 7 },
    { "element": "date",   "start": 9, "end": 13 }
  ],
  "elapsed_seconds": 1.4
}
```

### `POST /api/evaluate`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | `string` | first available | Connector id. |
| `model` | `string` | provider default | Model ID. |
| `schema` | `string` | `"bibl"` | Registered schema name. |
| `corpus` | `string` | `"default"` | Corpus label (e.g. `"default"`, `"hard-cases"`). |
| `n` | `integer` | `5` | Number of random samples. |
| `seed` | `integer` | — | Random seed for reproducible sampling across providers. |
| `batch_size` | `integer` | `1` | Records per LLM call. |

Samples `n` records from the selected corpus, annotates each, and returns micro precision/recall/F1 plus a per-element breakdown.

**Example:**

```json
{
  "provider": "gemini",
  "model": "gemini-2.0-flash",
  "schema": "bibl-reference-segmenter",
  "corpus": "hard-cases",
  "n": 5,
  "seed": 42
}
```

### `GET /api/sample`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `n` | `integer` | `5` | Number of samples to return. |
| `schema` | `string` | `"bibl"` | Registered schema name. |
| `corpus` | `string` | `"default"` | Corpus label. |

Returns an array of `{"text": "..."}` objects.

---

## GLiNER pre-detection (optional)

Set `GLINER_MODEL` in `.env` to enable an optional CPU-based pre-detection pass before the LLM step. This requires the `[gliner]` extra:

```bash
uv sync --extra gliner --extra webservice
```

Leave `GLINER_MODEL` empty (the default) to skip this step.
