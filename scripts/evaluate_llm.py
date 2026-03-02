#!/usr/bin/env python
"""
Evaluate tei-annotator annotation quality against the blbl-examples gold standard.

For each configured provider the script:
  1. Loads tests/fixtures/blbl-examples.tei.xml.
  2. Strips all tags from each <bibl> element → plain text.
  3. Runs the full annotate() pipeline.
  4. Compares the annotated output against the original markup.
  5. Prints precision, recall, and F1 — overall and per element type.

Providers:
  • Google Gemini 2.0 Flash     (GEMINI_API_KEY)
  • KISSKI llama-3.3-70b-instruct (KISSKI_API_KEY)

Usage:
    uv run scripts/evaluate_llm.py [--max-items N] [--match-mode text|exact|overlap] [--gliner-model MODEL]
    python scripts/evaluate_llm.py --max-items 10 --gliner-model numind/NuNER_Zero

API keys are read from .env in the project root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent
GOLD_FILE = _REPO / "tests" / "fixtures" / "blbl-examples.tei.xml"

# ---------------------------------------------------------------------------
# .env loader (stdlib-only, no python-dotenv needed)
# ---------------------------------------------------------------------------


def _load_env(path: str = ".env") -> None:
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), value)
    except FileNotFoundError:
        pass


_load_env(_REPO / ".env")


# ---------------------------------------------------------------------------
# HTTP helper (stdlib urllib)
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: dict, headers: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


# ---------------------------------------------------------------------------
# call_fn factories  (identical to smoke_test_llm.py)
# ---------------------------------------------------------------------------


def make_gemini_call_fn(api_key: str, model: str = "gemini-2.0-flash"):
    """Return a call_fn that sends a prompt to Gemini and returns the text reply."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models"
        f"/{model}:generateContent?key={api_key}"
    )

    def call_fn(prompt: str) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        result = _post_json(url, payload, {"Content-Type": "application/json"})
        return result["candidates"][0]["content"]["parts"][0]["text"]

    call_fn.__name__ = f"gemini/{model}"
    return call_fn


def make_kisski_call_fn(
    api_key: str,
    base_url: str = "https://chat-ai.academiccloud.de/v1",
    model: str = "llama-3.3-70b-instruct",
):
    """Return a call_fn for a KISSKI-hosted OpenAI-compatible model."""
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    def call_fn(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        result = _post_json(url, payload, headers)
        return result["choices"][0]["message"]["content"]

    call_fn.__name__ = f"kisski/{model}"
    return call_fn


# ---------------------------------------------------------------------------
# Schema — focused on the elements that appear in blbl-examples.tei.xml
# ---------------------------------------------------------------------------


def _build_schema():
    from tei_annotator.models.schema import TEIAttribute, TEIElement, TEISchema

    def attr(name: str, desc: str, allowed: list[str] | None = None) -> TEIAttribute:
        return TEIAttribute(name=name, description=desc, allowed_values=allowed)

    return TEISchema(
        elements=[
            TEIElement(
                tag="label",
                description="The number of a reference or of a footnote, preceeding the reference",
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="author",
                description="Name of an author of the cited work.",
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="editor",
                description="Name of an editor of the cited work.",
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="title",
                description="Title of the cited work.",
                allowed_children=[],
                attributes=[
                    attr(
                        "level",
                        "Publication level: 'a'=article/chapter, 'm'=monograph/book, "
                        "'j'=journal, 's'=series.",
                        ["a", "m", "j", "s"],
                    )
                ],
            ),
            TEIElement(
                tag="date",
                description="Publication date or year.",
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="publisher",
                description="Name of the publisher.",
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="pubPlace",
                description="Place of publication.",
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="biblScope",
                description="Scope reference within the cited item (page range, volume, issue).",
                allowed_children=[],
                attributes=[
                    attr(
                        "unit",
                        "Unit of the scope reference.",
                        ["page", "volume", "issue"],
                    )
                ],
            ),
            TEIElement(
                tag="idno",
                description="Bibliographic identifier such as DOI, ISBN, or ISSN.",
                allowed_children=[],
                attributes=[attr("type", "Identifier type, e.g. DOI, ISBN, ISSN.")],
            ),
            TEIElement(
                tag="note",
                description="Editorial note or annotation about the cited item.",
                allowed_children=[],
                attributes=[attr("type", "Type of note, e.g. 'report'.")],
            ),
            TEIElement(
                tag="ptr",
                description="Pointer to an external resource such as a URL.",
                allowed_children=[],
                attributes=[attr("type", "Type of pointer, e.g. 'web'.")],
            ),
            TEIElement(
                tag="orgName",
                description="Name of an organisation.",
                allowed_children=[],
                attributes=[],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_evaluation(
    provider_name: str,
    call_fn,
    match_mode_str: str,
    max_items: int | None,
    gliner_model: str | None = None,
    show_annotations: bool = False,
    output_file: Path | None = None,
) -> bool:
    """
    Evaluate one provider: iterate over gold records with live progress,
    then print overall and per-element metrics.

    When *output_file* is set all text output is written to that file and a
    tqdm progress bar is shown in the terminal instead of per-record lines.

    Returns True on success, False if a fatal exception occurred.
    """
    import contextlib
    import io
    import warnings
    from lxml import etree

    from tei_annotator import preload_gliner_model
    from tei_annotator.evaluation import evaluate_element, aggregate, MatchMode
    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    _TEI_NS = "http://www.tei-c.org/ns/1.0"

    mode_map = {
        "text": MatchMode.TEXT,
        "exact": MatchMode.EXACT,
        "overlap": MatchMode.OVERLAP,
    }
    match_mode = mode_map[match_mode_str]

    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=call_fn,
    )
    schema = _build_schema()

    # --- load gold records --------------------------------------------------
    tree = etree.parse(str(GOLD_FILE))
    containers = tree.findall(f".//{{{_TEI_NS}}}listBibl") or tree.findall(".//listBibl")
    all_bibls: list[etree._Element] = []
    for c in containers:
        children = c.findall(f"{{{_TEI_NS}}}bibl") or c.findall("bibl")
        all_bibls.extend(children)
    if max_items is not None:
        all_bibls = all_bibls[:max_items]
    n_total = len(all_bibls)

    # --- output destination and progress display ----------------------------
    # When --output-file: buffer all prints → file; show tqdm bar on stderr.
    # Otherwise: print to stdout and show manual per-record progress lines.
    _buf = io.StringIO() if output_file else None
    _pbar = (
        _tqdm(total=n_total, desc="Annotating", unit="rec", file=sys.stderr)
        if output_file and _tqdm
        else None
    )
    if output_file and not _tqdm:
        print("WARNING: tqdm not installed — no progress bar. Run: pip install tqdm",
              file=sys.stderr)

    _ok = False
    with contextlib.redirect_stdout(_buf) if _buf else contextlib.nullcontext():
        sep = "─" * 64
        print(f"\n{sep}")
        print(f"  Provider  : {provider_name}")
        print(f"  Gold file : {GOLD_FILE.relative_to(_REPO)}")
        print(f"  Records   : {n_total}   match-mode: {match_mode_str}")
        print(f"  GLiNER    : {gliner_model or 'disabled'}")
        print(sep)

        if gliner_model:
            print(f"  Loading GLiNER model '{gliner_model}'...", flush=True)
            preload_gliner_model(gliner_model)
            print(f"  GLiNER model ready.")

        per_record = []
        failed = 0
        for i, bibl in enumerate(all_bibls, 1):
            plain_text = "".join(bibl.itertext())
            snippet = plain_text[:60].replace("\n", " ")
            if _pbar:
                _pbar.set_description(snippet[:45])
            else:
                print(f"  [{i:3d}/{n_total}] {snippet}...", end="\r\n", flush=True)
            try:
                # Suppress the pipeline's best-effort XML validation warning here;
                # it surfaces again in the evaluator warning if parsing fails.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Output XML validation failed",
                    )
                    result = evaluate_element(
                        gold_element=bibl,
                        schema=schema,
                        endpoint=endpoint,
                        gliner_model=gliner_model,
                        match_mode=match_mode,
                    )
                if show_annotations and result.annotation_xml is not None:
                    sep60 = "─" * 60
                    gold_parts = [bibl.text or ""]
                    for child in bibl:
                        child_xml = etree.tostring(child, encoding="unicode", with_tail=True)
                        gold_parts.append(re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', "", child_xml))
                    gold_xml = "".join(gold_parts)
                    print(f"\n  {sep60}")
                    print(f"  Gold:       {gold_xml}")
                    print(f"  Annotation: {result.annotation_xml}")
                    print(f"  F1={result.micro_f1:.3f}  "
                          f"missed={[s.element for s in result.unmatched_gold]}  "
                          f"spurious={[s.element for s in result.unmatched_pred]}")
                    print(f"  {sep60}\n")
                per_record.append(result)
                if _pbar:
                    _pbar.update(1)
                    _pbar.set_postfix(F1=f"{result.micro_f1:.3f}")
            except Exception as exc:
                print(f"\n  [{i:3d}/{n_total}] ERROR — {exc}")
                failed += 1
                if _pbar:
                    _pbar.update(1)

        if _pbar:
            _pbar.close()
        else:
            # Clear the progress line
            print(" " * 70, end="\r")

        if not per_record:
            print("  ✗ All records failed — no results to report.")
        else:
            overall = aggregate(per_record)
            n_ok = len(per_record)
            print(f"\n  Completed: {n_ok}/{n_total} records"
                  + (f"  ({failed} failed)" if failed else "") + "\n")
            print(overall.report(title=f"Overall — {provider_name}"))

            # Show the five worst records (by F1) for diagnostics
            worst = sorted(enumerate(per_record, 1), key=lambda x: x[1].micro_f1)[:5]
            if worst and worst[0][1].micro_f1 < 1.0:
                print(f"\n  Lowest-F1 records (top 5):")
                for idx, r in worst:
                    gold_bibl = all_bibls[idx - 1]
                    snippet = "".join(gold_bibl.itertext())[:55].replace("\n", " ")
                    fn_tags = [s.element for s in r.unmatched_gold]
                    fp_tags = [s.element for s in r.unmatched_pred]
                    print(
                        f"    #{idx:3d}  F1={r.micro_f1:.3f}"
                        f"  missed={fn_tags}  spurious={fp_tags}"
                    )
                    print(f'         "{snippet}..."')

            _ok = True

    if _buf is not None:
        output_file.write_text(_buf.getvalue(), encoding="utf-8")
        print(f"\n  Output written to: {output_file}")

    return _ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate tei-annotator against blbl-examples.tei.xml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N <bibl> records (useful for quick runs).",
    )
    p.add_argument(
        "--match-mode",
        choices=["text", "exact", "overlap"],
        default="text",
        help="Span matching criterion.",
    )
    p.add_argument(
        "--gliner-model",
        default=None,
        metavar="MODEL",
        help=(
            "HuggingFace GLiNER model ID for the optional pre-detection pass "
            "(e.g. 'numind/NuNER_Zero'). Omit to disable."
        ),
    )
    p.add_argument(
        "--show-annotations",
        action="store_true",
        default=False,
        help="Print the annotated XML output for each record (useful for inspection runs).",
    )
    p.add_argument(
        "--output-file",
        default=None,
        metavar="PATH",
        help=(
            "Write all evaluation output to this file. "
            "A tqdm progress bar is shown in the terminal instead of per-record lines."
        ),
    )
    p.add_argument(
        "--provider",
        choices=["gemini", "kisski", "all"],
        default="all",
        help="Which provider(s) to evaluate.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    kisski_key = os.environ.get("KISSKI_API_KEY", "")

    providers: list[tuple[str, object]] = []

    if args.provider in ("gemini", "all"):
        if not gemini_key:
            print("ERROR: GEMINI_API_KEY not set (check .env)", file=sys.stderr)
            if args.provider == "gemini":
                return 1
        else:
            providers.append(("Gemini 2.0 Flash", make_gemini_call_fn(gemini_key)))

    if args.provider in ("kisski", "all"):
        if not kisski_key:
            print("ERROR: KISSKI_API_KEY not set (check .env)", file=sys.stderr)
            if args.provider == "kisski":
                return 1
        else:
            providers.append(
                ("KISSKI / llama-3.3-70b-instruct", make_kisski_call_fn(kisski_key))
            )

    if not providers:
        print("ERROR: No providers configured — check your .env file.", file=sys.stderr)
        return 1

    results: list[bool] = []
    for name, fn in providers:
        ok = run_evaluation(
            provider_name=name,
            call_fn=fn,
            match_mode_str=args.match_mode,
            max_items=args.max_items,
            gliner_model=args.gliner_model,
            show_annotations=args.show_annotations,
            output_file=Path(args.output_file) if args.output_file else None,
        )
        results.append(ok)

    print(f"\n{'═' * 64}")
    passed = sum(results)
    total = len(results)
    print(f"  Result: {passed}/{total} providers completed successfully")
    print(f"{'═' * 64}\n")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
