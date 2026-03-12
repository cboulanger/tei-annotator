# TEI Annotator — Webservice

FastAPI JSON API with a browser UI. Supports multiple LLM providers via a connector system — providers are enabled automatically based on which API keys are present in `.env`.

---

## Running

```bash
uv sync --extra webservice
cp .env.template .env
# edit .env — set at least one provider API key (see Providers below)
uv run task webservice          # reads HOST / PORT from .env
uv run python main.py --reload  # development mode with auto-reload
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

Each connector declares some models as *premium* by prefixing the model name with `*` in its `_MODELS` list (see `connectors.py`). Standard models are shown to all users; premium models are hidden unless the visitor holds the `PREMIUM_TOKEN` (see [Security](#security) below). The `*` is stripped before the model ID is passed to any API.

### Adding a new provider

Subclass `Connector` in `connectors.py` and append an instance to `_ALL_CONNECTORS`. Required interface:

- `id` / `name` / `description` — machine id, UI label, one-sentence description
- `is_available()` — return `True` iff the required env var is set
- `_MODELS` — list of model IDs; prefix with `*` to mark as premium
- `make_call_fn(model_id, timeout)` — return a `(prompt: str) -> str` callable

Override `models()` and `standard_models()` only if the model list is dynamic (see `KISSKIConnector`).

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
| `/api/config` | `GET` | Available providers, models, and tokens |
| `/api/annotate` | `POST` (JSON) | Annotate text, return XML |
| `/api/evaluate` | `POST` (JSON) | Run evaluation against the gold standard |
| `/api/sample` | `GET` | Sample plain-text entries from the test fixture |
| `/docs` | `GET` | Interactive OpenAPI documentation (Swagger UI) |

### `GET /api/config?key=<premium_token>`

```json
{
  "providers": [
    {
      "id": "gemini",
      "name": "Google Gemini",
      "description": "...",
      "models": ["gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"],
      "default_model": "gemini-2.0-flash-lite"
    }
  ],
  "token": "<api_key_or_null>",
  "premium": false
}
```

Pass `?key=<PREMIUM_TOKEN>` to receive the full model list and `"premium": true`.

### `POST /api/annotate`

```json
{
  "text": "Doe, J. (2024). A paper. Journal of Foo, 12(3), 1–10.",
  "provider": "gemini",
  "model": "gemini-2.5-flash"
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
