#!/usr/bin/env python
"""
Smoke test for the TEI Annotator webservice.

Assumes a locally running instance.  Start it first:
    cd webservice && uvicorn main:app --reload

The base URL is derived from HOST and PORT in webservice/.env.
If that file does not exist, the script exits with instructions.

Usage:
    python scripts/smoke_test_webservice.py [--base-url URL]
    uv run scripts/smoke_test_webservice.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve base URL from webservice/.env
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent
_ENV_FILE = _REPO / "webservice" / ".env"
_ENV_TEMPLATE = _REPO / "webservice" / ".env.template"


def _load_webservice_env() -> dict[str, str]:
    """Parse webservice/.env and return key→value pairs (no shell expansion)."""
    if not _ENV_FILE.exists():
        print(
            f"ERROR: {_ENV_FILE} not found.\n"
            f"       Create it from the template first:\n"
            f"         cp {_ENV_TEMPLATE} {_ENV_FILE}\n"
            f"       Then fill in at least one API key.",
            file=sys.stderr,
        )
        sys.exit(1)
    env: dict[str, str] = {}
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _default_base_url() -> str:
    env = _load_webservice_env()
    host = env.get("HOST", "localhost") or "localhost"
    # When HOST is 0.0.0.0 (listen on all interfaces), connect via localhost
    if host in ("0.0.0.0", ""):
        host = "localhost"
    port = env.get("PORT", "8000") or "8000"
    return f"http://{host}:{port}"

_TEST_TEXT = (
    "Marie Curie was born in Warsaw, Poland, and later conducted her research "
    "in Paris, France. Together with her husband Pierre Curie, she discovered "
    "polonium and radium."
)

_MINIMAL_SCHEMA = {
    "elements": [
        {"tag": "persName", "description": "a person's name"},
        {"tag": "placeName", "description": "a geographical place name"},
    ],
    "rules": [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")
    except urllib.error.URLError as exc:
        print(
            f"\nERROR: Cannot reach the webservice at {url}\n"
            f"       {exc.reason}\n"
            f"       Is the server running?  Start it with:\n"
            f"         cd webservice && uvicorn main:app --reload",
            file=sys.stderr,
        )
        sys.exit(2)


def _post_json(url: str, payload: dict) -> tuple[int, dict | str]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")
    except urllib.error.URLError as exc:
        print(
            f"\nERROR: Cannot reach the webservice at {url}\n"
            f"       {exc.reason}\n"
            f"       Is the server running?  Start it with:\n"
            f"         cd webservice && uvicorn main:app --reload",
            file=sys.stderr,
        )
        sys.exit(2)


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return ok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_html_ui(base_url: str) -> bool:
    status, body = _get(f"{base_url}/")
    return _check(
        "GET / returns HTTP 200 with HTML form",
        status == 200 and "<form" in body,
        f"status={status}",
    )


def test_api_annotate(base_url: str) -> bool:
    status, data = _post_json(
        f"{base_url}/api/annotate",
        {"text": _TEST_TEXT, "schema": _MINIMAL_SCHEMA},
    )
    if not _check(
        "POST /api/annotate returns HTTP 200",
        status == 200,
        f"status={status}  body={str(data)[:200]}",
    ):
        return False

    xml = data.get("xml", "") if isinstance(data, dict) else ""
    plain = re.sub(r"<[^>]+>", "", xml)

    ok_nonempty = _check("Response xml is non-empty", bool(xml.strip()))
    ok_plain = _check(
        "Plain text preserved after stripping tags",
        plain == _TEST_TEXT,
        f"expected: {_TEST_TEXT!r}\n         got:      {plain!r}",
    )
    ok_fuzzy = _check(
        "Response contains fuzzy_spans list",
        isinstance(data.get("fuzzy_spans"), list) if isinstance(data, dict) else False,
    )
    return ok_nonempty and ok_plain and ok_fuzzy


def test_api_no_schema(base_url: str) -> bool:
    """Omitting schema should fall back to the built-in BLBL schema."""
    bibl = "Curie, Marie. 1898. Sur une nouvelle substance radioactive. Paris."
    status, data = _post_json(f"{base_url}/api/annotate", {"text": bibl})
    return _check(
        "POST /api/annotate without schema uses BLBL default (HTTP 200)",
        status == 200 and isinstance(data, dict) and bool(data.get("xml")),
        f"status={status}",
    )


def test_api_unknown_provider(base_url: str) -> bool:
    status, _ = _post_json(
        f"{base_url}/api/annotate",
        {"text": "hello", "provider": "nonexistent"},
    )
    return _check(
        "POST /api/annotate with unknown provider returns 400",
        status == 400,
        f"status={status}",
    )


def test_openapi_docs(base_url: str) -> bool:
    status, _ = _get(f"{base_url}/docs")
    return _check("GET /docs (OpenAPI UI) returns HTTP 200", status == 200)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    default_url = _default_base_url()
    parser = argparse.ArgumentParser(description="Smoke-test the TEI Annotator webservice.")
    parser.add_argument(
        "--base-url",
        default=default_url,
        help=f"Base URL of the running server (default: derived from webservice/.env, currently {default_url})",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print(f"Smoke-testing {base}\n")

    results = [
        test_html_ui(base),
        test_openapi_docs(base),
        test_api_unknown_provider(base),
        test_api_no_schema(base),
        test_api_annotate(base),
    ]

    passed = sum(results)
    total = len(results)
    print(f"\n{'═' * 50}")
    print(f"  {passed}/{total} checks passed")
    print(f"{'═' * 50}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
