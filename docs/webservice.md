# TEI Annotator — Local Webservice

`webservice/` is a FastAPI JSON API with a browser UI for local use and development. It supports multiple LLM providers via a connector system — providers are enabled automatically based on which API keys are present in the environment.

---

## Running

```bash
uv sync --extra webservice
cp webservice/.env.template webservice/.env
# edit webservice/.env — set at least one API key (see Providers below)
uv run python webservice/main.py           # reads HOST / PORT from .env
uv run python webservice/main.py --reload  # development mode with auto-reload
```

> **Note:** Do not start with `uvicorn main:app` directly — uvicorn binds the port from its CLI args *before* the module is imported, so `load_dotenv()` would run too late to affect the port. Running via `python main.py` also handles automatic cleanup of any previously running instance on the same port.

The default port is `8099` (configurable via `PORT` in `.env`).

---

## Providers

The webservice loads providers dynamically based on which environment variables are set. Configure them in `webservice/.env` (copy from `webservice/.env.template`):

| Provider | Env var | Notes |
| --- | --- | --- |
| **HuggingFace Inference Router** | `HF_TOKEN` | Open models via router.huggingface.co |
| **Google Gemini** | `GEMINI_API_KEY` | Gemini models via the generateContent REST API |
| **KISSKI** | `KISSKI_API_KEY` | Academic cloud at chat-ai.academiccloud.de; override base URL with `KISSKI_BASE_URL` |

Set `SELECTED_MODEL=<provider>/<model>` to pre-select a specific model across providers (e.g. `SELECTED_MODEL=hf/meta-llama/Llama-3.3-70B-Instruct:nscale`). Omit it to use each provider's first model as the default.

At least one key must be set. The browser UI shows all available providers grouped in the model dropdown; if no key is configured a setup notice is shown instead.

### Adding a new provider

Subclass `Connector` in `webservice/connectors.py` and append an instance to `_ALL_CONNECTORS`. The connector must implement:

- `id` / `name` / `description` — machine id, UI label, one-sentence description
- `is_available()` — return `True` iff the required env var is set
- `models()` — list of model IDs offered
- `make_call_fn(model_id, timeout)` — return a `(prompt: str) -> str` callable

---

## API endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | `GET` | Single-page browser UI |
| `/api/config` | `GET` | Available providers and models |
| `/api/annotate` | `POST` (JSON) | Annotate text, return XML |
| `/api/evaluate` | `POST` (JSON) | Run evaluation against the gold standard |
| `/api/sample` | `GET` | Sample plain-text entries from the test fixture |
| `/docs` | `GET` | Interactive OpenAPI documentation (Swagger UI) |

### `GET /api/config`

```json
{
  "providers": [
    {
      "id": "hf",
      "name": "HuggingFace Inference Router",
      "description": "...",
      "models": ["meta-llama/Llama-3.3-70B-Instruct:nscale", "..."],
      "default_model": "meta-llama/Llama-3.3-70B-Instruct:nscale"
    }
  ]
}
```

### `POST /api/annotate`

```json
{
  "text": "Doe, J. (2024). A paper. Journal of Foo, 12(3), 1–10.",
  "provider": "hf",
  "model": "meta-llama/Llama-3.3-70B-Instruct:nscale"
}
```

`provider` and `model` are optional; both default to the first available provider and its default model.

### `POST /api/evaluate`

```json
{
  "provider": "gemini",
  "model": "gemini-2.0-flash",
  "n": 5,
  "seed": 42
}
```

Samples `n` records from the gold-standard fixture, annotates each, and returns micro precision/recall/F1 plus a per-element breakdown. Pass `seed` to reproduce the same sample when comparing providers.

---

## GLiNER pre-detection (optional)

Set `GLINER_MODEL` in `.env` to enable an optional CPU-based pre-detection pass before the LLM step. This requires the `[gliner]` extra:

```bash
uv sync --extra gliner --extra webservice
```

Leave `GLINER_MODEL` empty (the default) to skip this step.
