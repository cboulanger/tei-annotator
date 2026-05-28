#!/usr/bin/env python
"""
Evaluate tei-annotator annotation quality against a gold-standard TEI file.

For each configured provider the script:
  1. Loads the gold TEI file (auto-detected from schema name, or --gold-file).
  2. Strips all tags from each record element → plain text.
  3. Runs the full annotate() pipeline.
  4. Compares the annotated output against the original markup.
  5. Prints precision, recall, and F1 — overall and per element type.
  6. Optionally logs results to an experiment tracker (W&B or MLFlow).

Providers:
  • Google Gemini 2.0 Flash     (GEMINI_API_KEY)
  • KISSKI llama-3.3-70b-instruct (KISSKI_API_KEY)

Usage:
    uv run scripts/evaluate_llm.py [--schema SCHEMA] [--gold-file PATH]
        [--max-items N] [--match-mode text|exact|overlap] [--gliner-model MODEL]
        [--track wandb|mlflow|all] [--run-name NAME] [--log-prompts]

    # bibl schema (default) against its auto-detected corpus file:
    uv run scripts/evaluate_llm.py --schema bibl

    # referenceSegmenter schema against its auto-detected corpus file:
    uv run scripts/evaluate_llm.py --schema bibl-reference-segmenter

    # any schema with an explicit gold file:
    uv run scripts/evaluate_llm.py --schema bibl --gold-file path/to/file.xml

    # log results to Weights & Biases:
    uv run scripts/evaluate_llm.py --schema bibl --track wandb

    # log results to MLFlow (GitLab), capturing prompts in the per-record table:
    uv run scripts/evaluate_llm.py --schema bibl --track mlflow --log-prompts

Gold-file auto-detection rule:
    data/corpus/<schema-name>.default.tei.xml
    e.g. schema "bibl" → data/corpus/bibl.default.tei.xml

API keys are read from .env in the project root.

Experiment tracking env vars:
    WANDB_API_KEY              — enables Weights & Biases tracker
    WANDB_PROJECT              — W&B project name (default: "tei-annotator")
    WANDB_ENTITY               — W&B team/user (optional)
    MLFLOW_TRACKING_URI        — enables MLFlow tracker (GitLab or local)
    MLFLOW_TRACKING_TOKEN      — auth token (GitLab PAT)
    MLFLOW_EXPERIMENT_NAME     — experiment name (default: "tei-annotator")
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from tei_annotator.providers import _ALL_CONNECTORS, get_available_connectors, get_connector
from tei_annotator.schemas.registry import build_schema, get_schema_config, get_schema_names
from tei_annotator.tracking import _ALL_TRACKERS, get_available_trackers, get_tracker

# ---------------------------------------------------------------------------
# Paths & schema registry
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent
_CORPUS = _REPO / "data" / "corpus"

# Separator used to join multiple records into a single annotate() call.
# Triple-pipe never appears in bibliographic text; inject_xml() never modifies
# text characters, so this string is guaranteed to survive the annotation pass.
_BATCH_SEP = "\n---RECORD|||SEP|||BOUNDARY---\n"

load_dotenv(_REPO / ".env")


def _default_gold_file(schema_name: str) -> Path:
    """Default corpus path: data/corpus/<schema-name>.default.tei.xml."""
    return _CORPUS / f"{schema_name}.default.tei.xml"


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

    # Step 1 — extract gold spans and plain text for every record.
    # Filter to schema-defined elements only (mirrors evaluate_element behaviour).
    schema_tags = frozenset(e.tag for e in schema.elements)
    plain_texts = []
    gold_spans_list = []
    for bibl in batch_bibls:
        pt, gs = extract_spans(bibl)
        plain_texts.append(pt)
        gold_spans_list.append([s for s in gs if s.element in schema_tags])

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
    import time as _time
    _batch_t0 = _time.monotonic()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Output XML validation failed")
        annotation_result = annotate(
            text=combined,
            schema=schema,
            endpoint=endpoint,
            gliner_model=gliner_model,
        )
    _elapsed_per_record = round((_time.monotonic() - _batch_t0) / len(non_empty_indices), 3)
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
        eval_result.elapsed_seconds = _elapsed_per_record
        results[i] = (eval_result, fragment)

    return results


# ---------------------------------------------------------------------------
# Record loader
# ---------------------------------------------------------------------------


def _load_records(
    gold_file: Path,
    root_element: str,
    child_element: str,
    grep: str | None = None,
    inverse_grep: str | None = None,
    shuffle: bool = False,
    max_items: int | None = None,
):
    import random
    from lxml import etree

    _TEI_NS = "http://www.tei-c.org/ns/1.0"
    tree = etree.parse(str(gold_file))
    containers = (
        tree.findall(f".//{{{_TEI_NS}}}{root_element}")
        or tree.findall(f".//{root_element}")
    )
    records: list[etree._Element] = []
    for c in containers:
        children = (
            c.findall(f"{{{_TEI_NS}}}{child_element}")
            or c.findall(child_element)
        )
        records.extend(children)
    if grep:
        _grep_re = re.compile(grep)
        records = [r for r in records if _grep_re.search("".join(r.itertext()))]
    if inverse_grep:
        _igrep_re = re.compile(inverse_grep)
        records = [r for r in records if not _igrep_re.search("".join(r.itertext()))]
    if shuffle:
        random.shuffle(records)
    if max_items is not None:
        records = records[:max_items]
    return records


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_evaluation(
    provider_name: str,
    call_fn,
    match_mode_str: str,
    schema,
    gold_file: Path,
    records: list,
    gliner_model: str | None = None,
    verbose: bool = False,
    output_file: Path | None = None,
    batch_size: int = 1,
    tracker=None,
    run_name: str | None = None,
    log_prompts: bool = False,
) -> bool:
    """
    Evaluate one provider: iterate over gold records with live progress,
    then print overall and per-element metrics.

    When *output_file* is set all text output is written to that file and a
    tqdm progress bar is shown in the terminal instead of per-record lines.

    When *tracker* is set (an ExperimentTracker instance), each run is logged
    to the configured backend (W&B, MLFlow, …).  Pass *log_prompts=True* to
    include the full LLM prompt in the per-record artifact table.

    Returns True on success, False if a fatal exception occurred.
    """
    import contextlib
    import datetime
    import io
    import warnings
    from lxml import etree

    from tei_annotator import preload_gliner_model
    from tei_annotator.evaluation import evaluate_element, aggregate, MatchMode
    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.tracking.base import RecordEntry, _NullRunContext

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    mode_map = {
        "text": MatchMode.TEXT,
        "exact": MatchMode.EXACT,
        "overlap": MatchMode.OVERLAP,
    }
    match_mode = mode_map[match_mode_str]

    n_total = len(records)

    # --- tracking run context -----------------------------------------------
    _run_name = run_name or (
        f"{schema.name}"
        f"-{provider_name.replace('/', '-').replace(' ', '_')}"
        f"-{datetime.datetime.now():%Y%m%dT%H%M%S}"
    )
    _params = {
        "schema": schema.name,
        "provider": provider_name,
        "match_mode": match_mode_str,
        "batch_size": batch_size,
        "n_records": n_total,
        "gliner_model": gliner_model or "disabled",
    }
    ctx = tracker.start_run(_run_name, _params) if tracker else _NullRunContext()
    if log_prompts:
        call_fn = ctx.wrap_call_fn(call_fn)

    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=call_fn,
    )

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
    with ctx:
        with contextlib.redirect_stdout(_buf) if _buf else contextlib.nullcontext():
            sep = "─" * 64
            print(f"\n{sep}")
            print(f"  Provider  : {provider_name}")
            print(f"  Gold file : {gold_file.relative_to(_REPO)}")
            print(f"  Records   : {n_total}   match-mode: {match_mode_str}")
            print(f"  Batch size: {batch_size}")
            print(f"  GLiNER    : {gliner_model or 'disabled'}")
            if tracker:
                print(f"  Tracking  : {tracker.name}  run={_run_name!r}")
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
                        # Log per-record entry to tracker
                        _snippet_tr = "".join(bibl.itertext())[:80].replace("\n", " ")
                        ctx.log_record(RecordEntry(
                            idx=item_idx + k + 1,
                            snippet=_snippet_tr,
                            micro_f1=result.micro_f1,
                            missed_tags=[s.element for s in result.unmatched_gold],
                            spurious_tags=[s.element for s in result.unmatched_pred],
                            elapsed_seconds=result.elapsed_seconds,
                            prompt=ctx._last_prompt if log_prompts else None,
                        ))
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
                ctx.log_summary(overall)
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
        description="Evaluate tei-annotator against a gold-standard TEI file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--schema",
        choices=sorted(get_schema_names()),
        default="bibl",
        help=(
            "Annotation schema to evaluate.  The gold file is auto-detected as "
            "data/corpus/<schema>.default.tei.xml unless --gold-file is given."
        ),
    )
    p.add_argument(
        "--gold-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to the gold-standard TEI XML file.  Overrides the auto-detected "
            "path derived from --schema."
        ),
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N records (useful for quick runs).",
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
        "--model",
        default=None,
        metavar="MODEL_ID",
        help="Model ID to use (default: each connector's default_model).",
    )
    p.add_argument(
        "--provider",
        choices=[c.id for c in _ALL_CONNECTORS] + ["all"],
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
    p.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Randomly shuffle the evaluation set before applying --max-items.",
    )
    _tracker_ids = [t.id for t in _ALL_TRACKERS]
    p.add_argument(
        "--track",
        choices=_tracker_ids + ["all"],
        default=None,
        metavar="BACKEND",
        help=(
            f"Log results to an experiment tracker. "
            f"Available backends: {', '.join(_tracker_ids)} or 'all'. "
            "Requires the corresponding env vars and optional package to be installed."
        ),
    )
    p.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help=(
            "Custom name for this experiment run. "
            "Default: <schema>-<provider>-<timestamp>."
        ),
    )
    p.add_argument(
        "--log-prompts",
        action="store_true",
        default=False,
        help=(
            "Capture and include the full LLM prompt in the per-record artifact table. "
            "Only has effect when --track is set."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    providers: list[tuple[str, object]] = []

    if args.provider == "all":
        connectors = get_available_connectors()
        if not connectors:
            print("ERROR: No providers configured — check your .env file.", file=sys.stderr)
            return 1
    else:
        connector = get_connector(args.provider)
        if connector is None:
            print(f"ERROR: Unknown provider {args.provider!r}", file=sys.stderr)
            return 1
        if not connector.is_available():
            print(f"ERROR: Provider {args.provider!r} not available — check your .env file.", file=sys.stderr)
            return 1
        connectors = [connector]

    for connector in connectors:
        model = args.model or connector.default_model
        providers.append((
            f"{connector.name} / {model}",
            connector.make_call_fn(model, timeout=args.timeout),
        ))

    # Resolve schema and gold file
    schema_cfg = get_schema_config(args.schema)
    schema = build_schema(args.schema)
    gold_file = Path(args.gold_file) if args.gold_file else _default_gold_file(args.schema)
    if not gold_file.exists():
        print(
            f"ERROR: Gold file not found: {gold_file}\n"
            f"       Create it or pass --gold-file PATH.",
            file=sys.stderr,
        )
        return 1

    if args.output_file:
        Path(args.output_file).write_text("", encoding="utf-8")

    records = _load_records(
        gold_file=gold_file,
        root_element=schema_cfg["root_element"],
        child_element=schema_cfg["child_element"],
        grep=args.grep,
        inverse_grep=args.inverse_grep,
        shuffle=args.shuffle,
        max_items=args.max_items,
    )

    # --- resolve experiment tracker(s) -------------------------------------
    tracker = None
    if args.track:
        if args.track == "all":
            available = get_available_trackers()
            if not available:
                print(
                    "WARNING: --track all specified but no trackers are configured "
                    "(check WANDB_API_KEY / MLFLOW_TRACKING_URI).",
                    file=sys.stderr,
                )
            elif len(available) > 1:
                # Multiple trackers: use the first available; log a note about others
                print(
                    f"NOTE: Multiple trackers available "
                    f"({', '.join(t.name for t in available)}). "
                    "Currently only one tracker per run is supported; "
                    f"using {available[0].name}.",
                    file=sys.stderr,
                )
                tracker = available[0]
            else:
                tracker = available[0]
        else:
            t = get_tracker(args.track)
            if t is None:
                print(f"ERROR: Unknown tracker {args.track!r}", file=sys.stderr)
                return 1
            if not t.is_available():
                print(
                    f"ERROR: Tracker {args.track!r} is not available — "
                    "check your environment variables.",
                    file=sys.stderr,
                )
                return 1
            tracker = t

    results: list[bool] = []
    for name, fn in providers:
        ok = run_evaluation(
            provider_name=name,
            call_fn=fn,
            match_mode_str=args.match_mode,
            schema=schema,
            gold_file=gold_file,
            records=records,
            gliner_model=args.gliner_model,
            verbose=args.verbose,
            output_file=Path(args.output_file) if args.output_file else None,
            batch_size=args.batch_size,
            tracker=tracker,
            run_name=args.run_name,
            log_prompts=args.log_prompts,
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
