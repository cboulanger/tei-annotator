# GLiNER pre-detection (optional)

[GLiNER](https://github.com/urchade/GLiNER) is a zero-shot named-entity recognition model that runs locally on CPU. The pre-detection pass uses GLiNER to quickly surface entity candidates before the (slower, more expensive) LLM annotation pass.

---

## Why a pre-detection pass?

Pre-detection serves two purposes:

1. **Improved recall** — GLiNER is fast and broad. It catches many candidate spans that the LLM can then verify, correct, and extend. This is especially useful in longer texts where lower-salience entities might otherwise be missed.

2. **Cost reduction** — By surfacing strong candidates in the prompt, the LLM can spend less effort on exhaustive detection and more on attribute filling and disambiguation.

The step is entirely optional: passing `gliner_model=None` to `annotate()` skips it and the LLM works from the raw text alone.

---

## How it works

`detect_spans(text, schema, model_name)` in `gliner_detector.py`:

1. **Label mapping** — Each `TEIElement.description` becomes a GLiNER label string. For example, `TEIElement(tag="persName", description="a person's name")` produces the label `"a person's name"`. GLiNER's zero-shot design means no fine-tuning is needed — labels are matched by semantic similarity at inference time.

2. **Inference** — The GLiNER model runs span prediction on the full source text using those labels. It returns a list of `(text, label, score)` triples.

3. **Context windowing** — For each detected span, a context window of ±60 characters is extracted from the source text to populate `SpanDescriptor.context`. This context string is what the resolver later uses to locate the span in the source text when converting it to a character offset.

4. **Output** — Each detection becomes a `SpanDescriptor(element=<tag>, text=<matched text>, context=<window>, score=<confidence>)`.

---

## Candidates in the LLM prompt

The `SpanDescriptor` list is passed to the prompt builder, which renders the candidates as a JSON block inside the LLM prompt. The LLM is instructed to treat them as suggestions, not ground truth: it may confirm, correct, split, merge, or discard them based on its own reading of the text.

---

## Installation

The GLiNER integration is an optional dependency group:

```bash
uv sync --extra gliner
```

This installs `gliner`, PyTorch, and Hugging Face Transformers. Without this extra, importing `detect_spans` raises `ImportError` with a clear message. The `annotate()` function imports GLiNER lazily, so the base package always loads without it.

---

## Choosing a model

Any GLiNER-compatible Hugging Face model can be used. Recommended starting points:

| Model | Download size | Notes |
|-------|--------------|-------|
| `numind/NuNER_Zero` | ~200 MB | Good general-purpose zero-shot NER; default in the README examples |
| `urchade/gliner_medium-v2.1` | ~400 MB | Broader entity type coverage |
| `urchade/gliner_large-v2.1` | ~800 MB | Higher accuracy on ambiguous entities |

Models are downloaded automatically from Hugging Face Hub on first use and cached in the default HF cache directory (`~/.cache/huggingface/`).

---

## Performance characteristics

GLiNER inference is CPU-based and typically completes in milliseconds to low seconds per text chunk, depending on text length and hardware. It is orders of magnitude faster than a remote LLM call. For batch evaluation workloads the GLiNER pass adds negligible overhead.
