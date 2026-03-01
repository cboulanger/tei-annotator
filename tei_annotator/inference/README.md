# Inference configuration

The annotator is endpoint-agnostic: it talks to any language model (or extraction model) through a single `call_fn: (str) -> str` callable. `EndpointConfig` wires together a capability declaration and that callable.

---

## `EndpointCapability`

```python
from tei_annotator import EndpointCapability
```

| Value | When to use |
|-------|-------------|
| `TEXT_GENERATION` | Standard chat/completion LLM. JSON is requested via the prompt. If the response cannot be parsed, the pipeline sends a self-correction follow-up and retries once. |
| `JSON_ENFORCED` | Constrained-decoding endpoint that guarantees syntactically valid JSON output (e.g. a vLLM server with `--guided-decoding-backend`). The correction retry is skipped because output is always parseable. |
| `EXTRACTION` | Native extraction model (GLiNER2 / NuExtract-style). The raw source text is passed directly; no Jinja2 prompt is built. Used internally when `gliner_model=` is set on `annotate()`; do not wrap these models in `EndpointConfig`. |

---

## `EndpointConfig`

```python
from tei_annotator import EndpointConfig, EndpointCapability

endpoint = EndpointConfig(
    capability=EndpointCapability.TEXT_GENERATION,
    call_fn=my_call_fn,
)
```

`call_fn` receives the complete prompt string and must return the model's raw response string. Any implementation is valid — an `openai.Client`, an `anthropic.Anthropic` client, a local `requests.post` to Ollama, or a function that reads from a file for testing.

---

## Examples

### Anthropic (Claude)

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

def call_fn(prompt: str) -> str:
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

endpoint = EndpointConfig(
    capability=EndpointCapability.TEXT_GENERATION,
    call_fn=call_fn,
)
```

### OpenAI

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment

def call_fn(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

endpoint = EndpointConfig(
    capability=EndpointCapability.TEXT_GENERATION,
    call_fn=call_fn,
)
```

### Google Gemini

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

def call_fn(prompt: str) -> str:
    return model.generate_content(prompt).text

endpoint = EndpointConfig(
    capability=EndpointCapability.TEXT_GENERATION,
    call_fn=call_fn,
)
```

### Ollama (local)

```python
import requests

def call_fn(prompt: str) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1", "prompt": prompt, "stream": False},
    )
    return resp.json()["response"]

endpoint = EndpointConfig(
    capability=EndpointCapability.TEXT_GENERATION,
    call_fn=call_fn,
)
```

### vLLM with constrained JSON decoding

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

def call_fn(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"guided_json": True},
    )
    return resp.choices[0].message.content

endpoint = EndpointConfig(
    capability=EndpointCapability.JSON_ENFORCED,  # skip correction retry
    call_fn=call_fn,
)
```

---

## How the capability affects pipeline behaviour

| Capability | Prompt template | Retry on parse failure |
|------------|-----------------|------------------------|
| `TEXT_GENERATION` | `text_gen.jinja2` (verbose, with instructions) | Yes — one self-correction attempt |
| `JSON_ENFORCED` | `json_enforced.jinja2` (compact) | No |
| `EXTRACTION` | None — raw text passed directly | N/A |
