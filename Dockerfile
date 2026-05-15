# TEI Annotator — FastAPI webservice
#
# The image is provider-agnostic: which LLM backend is used depends entirely
# on environment variables set at runtime.
#
# ── Local deployment (Ollama) ────────────────────────────────────────────────
#
#   docker compose up --build
#
#   docker-compose.yml starts an Ollama sidecar, sets OLLAMA_BASE_URL and
#   OLLAMA_MODEL=qwen2.5:7b, and maps port 8099 → 7860.
#   The model is pulled automatically by Ollama on the first request (~4.7 GB).
#
#   To use a different model:
#     OLLAMA_MODEL=llama3.2 docker compose up --build
#
#   To connect to an Ollama instance running on the host instead of a sidecar:
#     docker run -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#                -e OLLAMA_MODEL=qwen2.5:7b \
#                -p 8099:7860 tei-annotator
#
# ── HuggingFace Spaces (Docker SDK) ─────────────────────────────────────────
#
#   1. Set sdk: docker and app_port: 7860 in the Space README.md frontmatter.
#   2. Add HF_TOKEN as a Space secret (needs "Make calls to Inference Providers"
#      scope — https://huggingface.co/settings/tokens).
#   3. Push this repo; HF Spaces builds and runs the image automatically.
#
#   The HF Inference Router (Qwen/Qwen3-14B by default) is used at no cost.
#   Add GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY as additional
#   Space secrets to unlock more providers.
#
# ── Other providers ──────────────────────────────────────────────────────────
#
#   Any combination of the following env vars enables the corresponding provider:
#     HF_TOKEN, GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, KISSKI_API_KEY
#
#   See webservice/.env.template for all configuration options.
#
FROM python:3.12-slim

# HuggingFace Spaces requires UID 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy dependency spec first for layer caching
COPY pyproject.toml ./

# Copy only what the webservice needs at runtime
COPY tei_annotator/ ./tei_annotator/
COPY webservice/ ./webservice/
COPY data/corpus/ ./data/corpus/

RUN uv pip install --system --no-cache -e ".[webservice]"

RUN chown -R appuser:appuser /app
USER appuser

# HuggingFace Spaces default port; remap to taste locally via docker-compose
EXPOSE 7860
ENV PORT=7860 HOST=0.0.0.0

CMD ["python", "webservice/main.py"]
