# ============================================================
# TEI Annotator webservice
# ============================================================
# Build:
#   docker build -t tei-annotator-webservice .
#
# Run (with an .env file in the webservice/ directory):
#   docker run --env-file webservice/.env -p 8000:8000 tei-annotator-webservice
#
# Run (passing individual env vars):
#   docker run -e GEMINI_API_KEY=... -p 8000:8000 tei-annotator-webservice
# ============================================================

FROM python:3.12-slim

# Install the package with webservice extras
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[webservice]"

# Working directory for the server so that relative paths
# (templates/, .env) resolve correctly
WORKDIR /app/webservice

EXPOSE 8000

CMD ["python", "main.py"]
