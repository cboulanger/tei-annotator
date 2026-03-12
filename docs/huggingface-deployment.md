# Deploying to HuggingFace Spaces

`app.py` at the repository root is a Gradio app ready for deployment on HuggingFace Spaces.

**How billing works:** The Space owner sets their `HF_TOKEN` as a Space secret. All inference calls use that token; visitors use the app without any login or token input. HF PRO accounts include a generous free inference quota on `router.huggingface.co`.

---

## Step 1 — Create the Space

On [huggingface.co/new-space](https://huggingface.co/new-space), choose **Gradio** as the SDK.

HF generates a `README.md` with YAML frontmatter. Make sure it contains at minimum:

```yaml
---
sdk: gradio
sdk_version: "6.9.0"
python_version: "3.12"
app_file: app.py
hardware: cpu-basic
---
```

> **Why `cpu-basic`?** The app makes HTTP calls to external LLM APIs — it does not run any local GPU workloads. Using `cpu-basic` avoids the GPU-slot allocation overhead (5–15 s per request) and GPU-task timeout issues that come with ZeroGPU (`zero-a10g`) hardware.

## Step 2 — Push the repository

```bash
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

HF Spaces reads `requirements.txt` at the repo root and installs dependencies automatically.

## Step 3 — Set the HF_TOKEN secret

In your Space's **Settings → Variables and Secrets**, add a **Secret**:

| Secret name | Value |
| --- | --- |
| `HF_TOKEN` | Your HuggingFace API token ([create one here](https://huggingface.co/settings/tokens)) |

> **Token permissions required:** The token must have the **"Make calls to Inference Providers"** scope enabled (under "Inference" when creating/editing the token at https://huggingface.co/settings/tokens). Without this scope, all annotation and evaluation calls will return HTTP 403.

The app shows a setup warning if this secret is missing.

## Step 4 — Verify

Once the Space has built, open its URL and annotate a sample text.

---

## Model list

Models are defined in `app.py` (`_HF_MODELS`), mirrored in `webservice/main.py`. All are pinned to inference providers that work from AWS-hosted Spaces (nscale, scaleway). Providers blocked from AWS — groq, cerebras, together-ai, sambanova — are avoided.

---

## Local development

```bash
uv sync --extra gradio
HF_TOKEN=hf_... uv run task gradio
# opens at http://localhost:7860
```

Set `HF_TOKEN` to a token with the "Make calls to Inference Providers" scope. You can also put it in a `.env` file at the repo root:

```bash
echo "HF_TOKEN=hf_..." > .env
uv run task gradio
```
