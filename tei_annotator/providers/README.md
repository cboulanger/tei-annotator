# Providers

LLM provider connectors for tei-annotator. Each connector wraps one provider behind a uniform interface and self-reports availability based on whether its required env var is set.

---

## Built-in connectors

| File | ID | Env var | Notes |
| --- | --- | --- | --- |
| `hf.py` | `hf` | `HF_TOKEN` | HuggingFace Inference Router (OpenAI-compatible) |
| `gemini.py` | `gemini` | `GEMINI_API_KEY` | Google Gemini via the generateContent REST API |
| `kisski.py` | `kisski` | `KISSKI_API_KEY` | KISSKI academic cloud; model list fetched live |
| `openai.py` | `openai` | `OPENAI_API_KEY` | OpenAI chat completions |
| `claude.py` | `claude` | `ANTHROPIC_API_KEY` | Anthropic Claude Messages API |

`_ALL_CONNECTORS` in `__init__.py` is the authoritative list. `get_available_connectors()` filters it to connectors whose env var is present at call time. `get_connector(id)` looks up by the short id string.

---

## Adding a new provider

1. Create `tei_annotator/providers/myprovider.py`.
2. Subclass `Connector` from `.base` and implement the required interface:

```python
from .base import Connector, _post_json
import os
from typing import Callable

class MyConnector(Connector):
    _MODELS = ["model-a", "*model-b-premium"]   # prefix '*' for premium-only

    @property
    def id(self) -> str: return "myprovider"

    @property
    def name(self) -> str: return "My Provider"

    @property
    def description(self) -> str: return "One sentence. (requires MY_API_KEY)"

    def is_available(self) -> bool:
        return bool(os.environ.get("MY_API_KEY"))

    def make_call_fn(self, model_id: str, timeout: int = 300) -> Callable[[str], str]:
        api_key = os.environ.get("MY_API_KEY", "")
        # ... build headers, url, etc.
        def call_fn(prompt: str) -> str:
            payload = {"model": model_id, "prompt": prompt}
            result = _post_json(url, payload, headers, timeout)
            return result["output"]
        call_fn.__name__ = f"myprovider/{model_id}"
        return call_fn
```

3. Import it and append an instance in `__init__.py`:

```python
from .myprovider import MyConnector

_ALL_CONNECTORS: list[Connector] = [
    ...,
    MyConnector(),
]
```

That's all. The evaluate script, webservice, and Gradio app pick it up automatically.

---

## Shared utilities (`base.py`)

- `_RateLimiter(rate_per_minute)` — thread-safe token-bucket rate limiter. Used by `KISSKIConnector`.
- `_post_json(url, payload, headers, timeout)` — thin stdlib `urllib` wrapper; raises `RuntimeError` on HTTP errors with the response body.
- `Connector` — abstract base class with `id`, `name`, `description`, `is_available()`, `models()`, `standard_models()`, `default_model`, `make_call_fn()`.

### Standard vs premium models

Prefix a model name in `_MODELS` with `*` to mark it as premium-only. `models()` returns all IDs with `*` stripped; `standard_models()` returns only the unprefixed ones. The webservice uses this to gate expensive models behind `PREMIUM_TOKEN`. Override `models()` and `standard_models()` if the model list is dynamic (see `KISSKIConnector`).
