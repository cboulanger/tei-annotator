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
    python scripts/evaluate_llm.py --max-items 10 --gliner-model numind/NuNER_Zero --verbose

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

# Separator used to join multiple bibl plain-texts into a single annotate() call.
# Triple-pipe never appears in bibliographic text; inject_xml() never modifies
# text characters, so this string is guaranteed to survive the annotation pass.
_BATCH_SEP = "\n---RECORD|||SEP|||BOUNDARY---\n"

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


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


# ---------------------------------------------------------------------------
# call_fn factories  (identical to smoke_test_llm.py)
# ---------------------------------------------------------------------------


def make_gemini_call_fn(api_key: str, model: str = "gemini-2.0-flash", timeout: int = 120):
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
        result = _post_json(url, payload, {"Content-Type": "application/json"}, timeout)
        return result["candidates"][0]["content"]["parts"][0]["text"]

    call_fn.__name__ = f"gemini/{model}"
    return call_fn


def make_kisski_call_fn(
    api_key: str,
    base_url: str = "https://chat-ai.academiccloud.de/v1",
    model: str = "llama-3.3-70b-instruct",
    timeout: int = 120,
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
        result = _post_json(url, payload, headers, timeout)
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
        rules=[
            "For each person's name, emit an 'author' or 'editor' span covering the full name "
            "AND separate 'surname', 'forename', or 'orgName' spans for the individual name "
            "parts within that span.",
            "Never emit 'surname', 'forename', or 'orgName' without a corresponding enclosing "
            "'author' or 'editor' span.",
            "When an organisation acts as author or editor, emit BOTH an 'orgName' span AND an "
            "enclosing 'author' (or 'editor') span. The 'author'/'editor' span MUST enclose the "
            "'orgName' span — NEVER put an 'author' or 'editor' span inside an 'orgName' span.",
            "CRITICAL: All name parts for all contiguous authors MUST always be placed inside a "
            "SINGLE 'author' (or 'editor') span — conjunctions ('and', '&', 'et') and commas "
            "between names do NOT create separate spans. Emit a new 'author' span only when "
            "the authors are separated by a title, date, or other non-name bibliographic field.",
            "In a bibliography, a dash or underscore may stand for a repeated author or editor "
            "name — tag it as 'author' or 'editor' accordingly.",
            "CRITICAL: When a parenthesised location appears immediately after a title "
            "(e.g. 'Title (City, Region)'), end the 'title' span BEFORE the opening parenthesis "
            "and emit a separate 'pubPlace' span covering only 'City, Region' (not the parentheses). "
            "Never include a parenthesised location inside a 'title' span.",
        ],
        elements=[
            TEIElement(
                tag="label",
                description=(
                    "A numeric or alphanumeric reference label appearing at the very start of a "
                    "bibliographic entry, before any author or title. Typical forms: a plain number "
                    "('17'), a number with a trailing period ('17.'), a number in square brackets "
                    "('[77]', '[ACL30]'), or a compound number ('5,6'). The separator that follows "
                    "the label (period, dash, or space) is NOT part of the label. "
                    "A label is always a number or short code — never a word or name. "
                    "An ALL-CAPS word at the start of an entry is an author surname, not a label."
                ),
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="author",
                description=(
                    "Name(s) of the author(s) of the cited work. "
                    "Names appearing at the start of a bibliographic entry before the title and "
                    "date are authors."
                ),
                allowed_children=['surname', 'forename', 'orgName'],
                attributes=[],
            ),
            TEIElement(
                tag="editor",
                description=(
                    "Name of an editor of the cited work. "
                    "An editor's name typically follows keywords such as 'in', 'ed.', 'éd.', "
                    "'Hrsg.', 'dir.', '(ed.)', '(eds.)'. "
                    "CRITICAL: A person's name (or surname alone) that follows 'in' is an editor — "
                    "emit an 'editor' span (plus name-part spans), never a 'title' span."
                ),
                allowed_children=['surname', 'forename', 'orgName'],
                attributes=[],
            ),
            TEIElement(
                tag="surname",
                description="The inherited (family) name of a person.",
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="forename",
                description="The given (first) name or initials of a person.",
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="orgName",
                description=(
                    "Name of an organisation that acts as author or editor. "
                    "Do NOT emit an 'orgName' span inside a 'publisher' span — "
                    "when an organisation is the publisher, use 'publisher' alone."
                ),
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="title",
                description=(
                    "Title of the cited work. "
                    "Do NOT split a title at an internal period or subtitle separator — "
                    "e.g. 'Classical Literary Criticism. Oxford World Classics' is ONE title span; "
                    "a city name embedded in a subtitle (e.g. 'Oxford' in 'Oxford World Classics') "
                    "is NOT a pubPlace — do not interrupt the title span with a pubPlace span. "
                    "CRITICAL: The title span ends BEFORE any parenthesised location — "
                    "e.g. in 'Title (City, Region)', only 'Title' is the title span; "
                    "'City, Region' is a separate pubPlace span. "
                    "A journal or series title may appear after keywords such as 'in', 'dans', 'in:' — "
                    "emit a 'title' span for it; do NOT tag it as 'note'."
                ),
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
                description=(
                    "Publication date or year. "
                    "When two dates appear in sequence — e.g. '1989 [1972]' (reprint year and "
                    "original year) — emit a SEPARATE 'date' span for each individual date."
                ),
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="publisher",
                description=(
                    "Name of the publisher. "
                    "When multiple publishers are connected by 'and', emit a SINGLE 'publisher' "
                    "span covering the full text (e.g. 'Cambridge University Press and the Russell "
                    "Sage Foundation' is one span). Do NOT nest 'orgName' inside 'publisher'."
                ),
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="pubPlace",
                description=(
                    "Place of publication. "
                    "CRITICAL: When a location appears in parentheses immediately after the title "
                    "(e.g. 'Title (City, Region)'), the parenthesised location is the pubPlace — "
                    "emit a 'pubPlace' span covering only 'City, Region' (without parentheses), "
                    "and end the 'title' span BEFORE the opening parenthesis. "
                    "Only tag a city name as pubPlace when it appears OUTSIDE and AFTER the title, "
                    "typically before a colon and publisher name (e.g. 'Oxford: Oxford UP'). "
                    "A city name that is part of a subtitle or series name within a title is NOT a pubPlace."
                ),
                allowed_children=[],
                attributes=[],
            ),
            TEIElement(
                tag="biblScope",
                description=(
                    "Scope reference within the cited item (page range, volume, issue). "
                    "Emit a separate 'biblScope' span for volume and for issue. "
                    "The span text contains ONLY the bare number — do not include labels "
                    "('Vol.', 'No.', 'n°', 't.') or surrounding punctuation/parentheses. "
                    "E.g. for 'Vol. 12(3)', emit '12' as unit='volume' and '3' as unit='issue'. "
                    "E.g. for 'n°198', emit '198' as unit='volume'. "
                    "Do NOT absorb a volume or issue number into a preceding title span."
                ),
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
                description=(
                    "Editorial note or annotation about the cited item. "
                    "Institutional or series report designations — such as 'Amok Internal Report', "
                    "'USGS Open-File Report 97-123', or 'Technical Report No. 5' — must be tagged "
                    "as 'note' with type='report', NOT as 'orgName' or 'title'."
                ),
                allowed_children=[],
                attributes=[attr("type", "Type of note, e.g. 'report'.")],
            ),
            TEIElement(
                tag="ptr",
                description="Pointer to an external resource such as a URL.",
                allowed_children=[],
                attributes=[attr("type", "Type of pointer, e.g. 'web'.")],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------


def _evaluate_batch(
    batch_bibls,
    schema,
    endpoint,
    match_mode,
    gliner_model=None,
    overlap_threshold: float = 0.5,
):
    """
    Annotate *batch_bibls* in a single annotate() call and evaluate each record.

    Returns a list of (EvaluationResult, annotation_xml_fragment | None) tuples
    in the same order as *batch_bibls*.  annotation_xml_fragment is the portion
    of the combined annotated XML that corresponds to that record (None if the
    record was empty or if the separator split failed).
    """
    import warnings
    from lxml import etree

    from tei_annotator.evaluation.evaluator import _escape_nonschema_brackets
    from tei_annotator.evaluation.extractor import extract_spans
    from tei_annotator.evaluation.metrics import compute_metrics
    from tei_annotator.pipeline import annotate

    n = len(batch_bibls)
    results = [None] * n

    # Step 1 — extract gold spans and plain text for every record
    plain_texts = []
    gold_spans_list = []
    for bibl in batch_bibls:
        pt, gs = extract_spans(bibl)
        plain_texts.append(pt)
        gold_spans_list.append(gs)

    # Step 2 — separate empty records (no text to annotate)
    non_empty_indices = [i for i, t in enumerate(plain_texts) if t.strip()]
    for i in range(n):
        if plain_texts[i].strip() == "":
            results[i] = (compute_metrics([], []), None)

    if not non_empty_indices:
        return results

    # Step 3 — guard: separator must not appear in any record's text
    for i in non_empty_indices:
        if _BATCH_SEP in plain_texts[i]:
            warnings.warn(
                f"Batch record {i} contains the batch separator; "
                "falling back to empty predictions for this batch.",
                stacklevel=2,
            )
            for j in non_empty_indices:
                results[j] = (compute_metrics(gold_spans_list[j], []), None)
            return results

    # Step 4 — build combined text and annotate in one call
    combined = _BATCH_SEP.join(plain_texts[i] for i in non_empty_indices)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Output XML validation failed")
        annotation_result = annotate(
            text=combined,
            schema=schema,
            endpoint=endpoint,
            gliner_model=gliner_model,
        )
    combined_xml = annotation_result.xml

    # Step 5 — split annotated XML back into per-record fragments
    pieces = combined_xml.split(_BATCH_SEP)
    if len(pieces) != len(non_empty_indices):
        warnings.warn(
            f"Batch split mismatch: expected {len(non_empty_indices)} pieces, "
            f"got {len(pieces)}. Returning empty predictions for this batch.",
            stacklevel=2,
        )
        for i in non_empty_indices:
            results[i] = (compute_metrics(gold_spans_list[i], []), None)
        return results

    # Step 6 — build EvaluationResult for each fragment
    allowed_tags = frozenset(e.tag for e in schema.elements)
    for k, i in enumerate(non_empty_indices):
        fragment = pieces[k]
        gold_spans = gold_spans_list[i]

        safe_xml = _escape_nonschema_brackets(fragment, allowed_tags)
        try:
            pred_root = etree.fromstring(f"<_root>{safe_xml}</_root>".encode())
            _, pred_spans = extract_spans(pred_root)
        except etree.XMLSyntaxError as exc:
            warnings.warn(
                f"Could not parse batch fragment {i} as XML; treating as empty: {exc}",
                stacklevel=2,
            )
            pred_spans = []

        eval_result = compute_metrics(
            gold_spans,
            pred_spans,
            mode=match_mode,
            overlap_threshold=overlap_threshold,
        )
        eval_result.annotation_xml = fragment
        results[i] = (eval_result, fragment)

    return results


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_evaluation(
    provider_name: str,
    call_fn,
    match_mode_str: str,
    max_items: int | None,
    gliner_model: str | None = None,
    verbose: bool = False,
    output_file: Path | None = None,
    grep: str | None = None,
    inverse_grep: str | None = None,
    batch_size: int = 1,
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
    records: list[etree._Element] = []
    for c in containers:
        children = c.findall(f"{{{_TEI_NS}}}bibl") or c.findall("bibl")
        records.extend(children)
    if grep:
        _grep_re = re.compile(grep)
        records = [r for r in records if _grep_re.search("".join(r.itertext()))]
    if inverse_grep:
        _igrep_re = re.compile(inverse_grep)
        records = [r for r in records if not _igrep_re.search("".join(r.itertext()))]
    if max_items is not None:
        records = records[:max_items]
    n_total = len(records)

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
        print(f"  Batch size: {batch_size}")
        print(f"  GLiNER    : {gliner_model or 'disabled'}")
        print(sep)

        if gliner_model:
            print(f"  Loading GLiNER model '{gliner_model}'...", flush=True)
            preload_gliner_model(gliner_model)
            print(f"  GLiNER model ready.")

        def _batched(lst, size):
            for start in range(0, len(lst), size):
                yield lst[start : start + size]

        per_record = []
        failed = 0
        item_idx = 0
        sep60 = "─" * 60
        for batch in _batched(records, batch_size):
            batch_start = item_idx + 1
            batch_end = item_idx + len(batch)
            snippet = "".join(batch[0].itertext())[:60].replace("\n", " ")
            if _pbar:
                _pbar.set_description(snippet[:45])
            else:
                range_str = (
                    f"{batch_start:3d}"
                    if batch_size == 1
                    else f"{batch_start}-{batch_end}"
                )
                print(f"  [{range_str}/{n_total}] {snippet}...", end="\r\n", flush=True)
            try:
                # Suppress the pipeline's best-effort XML validation warning here;
                # it surfaces again in the evaluator warning if parsing fails.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Output XML validation failed",
                    )
                    if batch_size == 1:
                        result = evaluate_element(
                            gold_element=batch[0],
                            schema=schema,
                            endpoint=endpoint,
                            gliner_model=gliner_model,
                            match_mode=match_mode,
                        )
                        batch_results = [(result, result.annotation_xml)]
                    else:
                        batch_results = _evaluate_batch(
                            batch_bibls=batch,
                            schema=schema,
                            endpoint=endpoint,
                            match_mode=match_mode,
                            gliner_model=gliner_model,
                        )
                for k, (result, annotation_frag) in enumerate(batch_results):
                    bibl = batch[k]
                    if verbose and annotation_frag is not None and result.micro_f1 < 1.0:
                        gold_parts = [bibl.text or ""]
                        for child in bibl:
                            child_xml = etree.tostring(child, encoding="unicode", with_tail=True)
                            gold_parts.append(re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', "", child_xml))
                        gold_xml = "".join(gold_parts)
                        print(f"  {sep60}")
                        print(f"  Gold:       {gold_xml}")
                        print(f"  Annotation: {annotation_frag}")
                        print(f"  F1={result.micro_f1:.3f}  "
                              f"missed={[s.element for s in result.unmatched_gold]}  "
                              f"spurious={[s.element for s in result.unmatched_pred]}")
                    per_record.append(result)
                item_idx += len(batch)
                if _pbar:
                    _pbar.update(len(batch))
                    _pbar.set_postfix(F1=f"{batch_results[0][0].micro_f1:.3f}")
            except Exception as exc:
                print(f"\n  [{batch_start}-{batch_end}/{n_total}] ERROR — {exc}")
                failed += len(batch)
                item_idx += len(batch)
                if _pbar:
                    _pbar.update(len(batch))

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
            worst = sorted(
                [(i, r) for i, r in enumerate(per_record, 1) if r.micro_f1 < 1.0],
                key=lambda x: x[1].micro_f1,
            )[:5]
            if worst:
                print(f"\n  Lowest-F1 records (top 5):")
                for idx, r in worst:
                    record = records[idx - 1]
                    snippet = "".join(record.itertext())[:55].replace("\n", " ")
                    fn_tags = [s.element for s in r.unmatched_gold]
                    fp_tags = [s.element for s in r.unmatched_pred]
                    print(
                        f"    #{idx:3d}  F1={r.micro_f1:.3f}"
                        f"  missed={fn_tags}  spurious={fp_tags}"
                    )
                    print(f'         "{snippet}..."')

            _ok = True

    if _buf is not None:
        with open(output_file, "a", encoding="utf-8") as _fh:
            _fh.write(_buf.getvalue())
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
        "--verbose",
        action="store_true",
        default=False,
        help="Print annotated XML for each record where F1 < 1.0 (useful for inspection runs).",
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
    p.add_argument(
        "--grep",
        default=None,
        metavar="PATTERN",
        help="Only evaluate records whose plain text matches this regex pattern.",
    )
    p.add_argument(
        "--inverse-grep",
        default=None,
        metavar="PATTERN",
        help="Only evaluate records whose plain text does NOT match this regex pattern.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of <bibl> records to annotate in a single LLM call. "
            "Default=1 (original one-record-per-call behavior). "
            "Use 5-20 to reduce latency at a potential quality cost "
            "(\"lost in the middle\" effect for items in large batches)."
        ),
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=120,
        metavar="SECONDS",
        help=(
            "HTTP read timeout in seconds for each LLM API call. "
            "Default=120. Increase when using large batch sizes with slow models "
            "(e.g. --timeout 600 --batch-size 10 for KISSKI Llama)."
        ),
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
            providers.append(("Gemini 2.0 Flash", make_gemini_call_fn(gemini_key, timeout=args.timeout)))

    if args.provider in ("kisski", "all"):
        if not kisski_key:
            print("ERROR: KISSKI_API_KEY not set (check .env)", file=sys.stderr)
            if args.provider == "kisski":
                return 1
        else:
            providers.append(
                ("KISSKI / llama-3.3-70b-instruct", make_kisski_call_fn(kisski_key, timeout=args.timeout))
            )

    if not providers:
        print("ERROR: No providers configured — check your .env file.", file=sys.stderr)
        return 1

    if args.output_file:
        Path(args.output_file).write_text("", encoding="utf-8")

    results: list[bool] = []
    for name, fn in providers:
        ok = run_evaluation(
            provider_name=name,
            call_fn=fn,
            match_mode_str=args.match_mode,
            max_items=args.max_items,
            gliner_model=args.gliner_model,
            verbose=args.verbose,
            output_file=Path(args.output_file) if args.output_file else None,
            grep=args.grep,
            inverse_grep=args.inverse_grep,
            batch_size=args.batch_size,
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
