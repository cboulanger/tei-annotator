#!/usr/bin/env python3
"""
Interactive annotation debugger — runs the annotation pipeline step-by-step
on a provided text snippet and prints every intermediate result.

Designed for a debugging agent: each stage shows inputs, outputs, and what
was lost (with reasons where determinable).

Usage:
    # Annotate text passed directly
    uv run scripts/debug_annotation.py --text "Bugnon (A.-L.), Le mobilier..."

    # Annotate text from a file
    uv run scripts/debug_annotation.py --file path/to/snippet.txt

    # Pipe text in
    echo "Bugnon (A.-L.), Le mobilier..." | uv run scripts/debug_annotation.py

    # Override provider/model/schema
    uv run scripts/debug_annotation.py --text "..." \\
        --provider kisski --model Qwen3-235B-A22B \\
        --schema bibl-reference-segmenter

    # Show full prompt (can be very long)
    uv run scripts/debug_annotation.py --text "..." --show-prompt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO = Path(__file__).parent.parent
load_dotenv(_REPO / ".env")

# ── ANSI helpers ────────────────────────────────────────────────────────────

_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RESET  = "\033[0m"


def _h1(title: str) -> None:
    line = "═" * 80
    print(f"\n{_BOLD}{_CYAN}{line}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{line}{_RESET}\n")


def _h2(title: str) -> None:
    pad = max(0, 78 - len(title))
    print(f"\n{_BOLD}{_YELLOW}── {title} {'─' * pad}{_RESET}\n")


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET}  {msg}")


def _err(msg: str) -> None:
    print(f"  {_RED}✗{_RESET}  {msg}", file=sys.stderr)


def _dim(text: str) -> str:
    return f"{_DIM}{text}{_RESET}"


# ── Span display helpers ─────────────────────────────────────────────────────

def _fmt_attrs(attrs: dict) -> str:
    if not attrs:
        return ""
    return " " + " ".join(f'{k}="{v}"' for k, v in attrs.items())


def _print_span_descriptors(spans: list, source: str | None = None) -> None:
    for i, s in enumerate(spans, 1):
        attrs = _fmt_attrs(s.attrs)
        ctx_preview = repr(s.context[:60]) if s.context else "—"
        print(f"  {i:2d}. <{s.element}{attrs}>")
        print(f"       text    = {repr(s.text[:80])}")
        print(f"       context = {ctx_preview}")
        if source is not None:
            pos = source.find(s.text)
            if pos == -1:
                print(f"       {_RED}text not found in source (resolver will reject){_RESET}")


def _print_resolved_spans(spans: list, source: str) -> None:
    for i, s in enumerate(spans, 1):
        attrs = _fmt_attrs(s.attrs)
        text_at_pos = source[s.start:s.end]
        fuzzy = f"  {_YELLOW}[fuzzy]{_RESET}" if s.fuzzy_match else ""
        print(f"  {i:2d}. <{s.element}{attrs}>  [{s.start}:{s.end}]  {repr(text_at_pos[:80])}{fuzzy}")


def _find_rejected_descriptors(
    descriptors: list, resolved: list, source: str
) -> list[tuple]:
    """Return (descriptor, reason) for each descriptor not represented in resolved."""
    resolved_ranges = {(s.start, s.end) for s in resolved}
    rejected = []
    for s in descriptors:
        pos = source.find(s.text)
        if pos == -1:
            rejected.append((s, "text not found in source"))
        elif (pos, pos + len(s.text)) not in resolved_ranges:
            # Likely context mismatch
            rejected.append((s, "context anchor not found or ambiguous"))
    return rejected


def _find_rejected_resolved(
    before: list, after: list, source: str, schema
) -> list[tuple]:
    """Return (span, reason) for each resolved span dropped by validate_spans."""
    after_keys = {(s.element, s.start, s.end) for s in after}
    rejected = []
    for s in before:
        key = (s.element, s.start, s.end)
        if key not in after_keys:
            # Determine reason
            elem = schema.get(s.element)
            if elem is None:
                reason = f"element <{s.element}> not in schema"
            elif s.start < 0 or s.end > len(source) or s.start >= s.end:
                reason = f"bounds [{s.start}:{s.end}] out of range"
            else:
                bad_attrs = []
                allowed_names = {a.name for a in elem.attributes}
                for attr_name, attr_value in s.attrs.items():
                    if attr_name not in allowed_names:
                        bad_attrs.append(f"unknown attr '{attr_name}'")
                    else:
                        attr_def = next((a for a in elem.attributes if a.name == attr_name), None)
                        if attr_def and attr_def.allowed_values is not None:
                            if attr_value not in attr_def.allowed_values:
                                bad_attrs.append(
                                    f"'{attr_name}={attr_value}' not in {attr_def.allowed_values}"
                                )
                reason = "; ".join(bad_attrs) if bad_attrs else "unknown reason"
            rejected.append((s, reason))
    return rejected


# ── Core debug pipeline ──────────────────────────────────────────────────────

def run_debug(
    text: str,
    schema,
    endpoint,
    chunk_size: int,
    chunk_overlap: int,
    show_prompt: bool,
    prompt_preview_chars: int = 400,
) -> None:
    from tei_annotator.chunking.chunker import chunk_text
    from tei_annotator.inference.endpoint import EndpointCapability
    from tei_annotator.models.spans import ResolvedSpan
    from tei_annotator.pipeline import (
        _escape_bare_ampersands,
        _strip_existing_tags,
    )
    from tei_annotator.postprocessing.injector import inject_xml
    from tei_annotator.postprocessing.parser import parse_response
    from tei_annotator.postprocessing.resolver import resolve_spans
    from tei_annotator.postprocessing.validator import validate_spans
    from tei_annotator.prompting.builder import build_prompt, make_correction_prompt

    # ── Step 1: strip tags ─────────────────────────────────────────────────
    _h2("STEP 1  Strip existing XML tags")
    plain_text, restore_map = _strip_existing_tags(text)
    if restore_map:
        _warn(f"Stripped {len(restore_map)} existing tag(s) — will be re-inserted later")
    else:
        _ok("No existing XML tags in input")
    print(f"\n  Plain text ({len(plain_text)} chars):")
    print(f"  {_dim(repr(plain_text[:200]) + ('...' if len(plain_text) > 200 else ''))}\n")

    # ── Step 2: chunking ───────────────────────────────────────────────────
    _h2("STEP 2  Chunking")
    chunks = chunk_text(plain_text, chunk_size=chunk_size, overlap=chunk_overlap)
    print(f"  {len(chunks)} chunk(s)  (chunk_size={chunk_size}, overlap={chunk_overlap})\n")

    # ── Steps 3–5: per-chunk pipeline ─────────────────────────────────────
    all_resolved: list[ResolvedSpan] = []

    for ci, chunk in enumerate(chunks, 1):
        _h2(f"STEP 3  Chunk {ci}/{len(chunks)}  [offset={chunk.start_offset}, len={len(chunk.text)}]")
        print(f"  {_dim(repr(chunk.text[:120]) + ('...' if len(chunk.text) > 120 else ''))}\n")

        # 3a: build prompt
        if endpoint.capability == EndpointCapability.EXTRACTION:
            prompt = None
            raw_response = endpoint.call_fn(chunk.text)
        else:
            prompt = build_prompt(
                source_text=chunk.text,
                schema=schema,
                capability=endpoint.capability,
            )

            print(f"  Prompt: {len(prompt)} chars")
            if show_prompt:
                print()
                for line in prompt.splitlines():
                    print(f"  {line}")
                print()
            else:
                preview = prompt[:prompt_preview_chars].replace("\n", " ↵ ")
                ellipsis = f"  {_dim(f'[...{len(prompt)-prompt_preview_chars} more chars — pass --show-prompt to see full prompt]')}" if len(prompt) > prompt_preview_chars else ""
                print(f"  {_dim(preview)}")
                if ellipsis:
                    print(ellipsis)
                print()

            # 3b: call LLM
            print(f"  Calling LLM...", end="", flush=True)
            try:
                raw_response = endpoint.call_fn(prompt)
                print(f" done ({len(raw_response)} chars)\n")
            except Exception as exc:
                print()
                _err(f"LLM call failed: {exc}")
                continue

        print(f"  {_BOLD}Raw LLM response:{_RESET}")
        print()
        for line in raw_response.splitlines():
            print(f"  {line}")
        print()

        # 3c: parse
        retry_fn = endpoint.call_fn if endpoint.capability == EndpointCapability.TEXT_GENERATION else None
        correction_fn = make_correction_prompt if endpoint.capability == EndpointCapability.TEXT_GENERATION else None
        try:
            span_descs = parse_response(
                raw_response,
                call_fn=retry_fn,
                make_correction_prompt=correction_fn,
            )
        except ValueError as exc:
            _err(f"parse_response failed: {exc}")
            continue

        print(f"  {_BOLD}Parsed spans:{_RESET}  {len(span_descs)} span descriptor(s)")
        print()
        _print_span_descriptors(span_descs, chunk.text)
        print()

        # 3d: resolve
        chunk_resolved = resolve_spans(chunk.text, span_descs)
        n_rejected_resolve = len(span_descs) - len(chunk_resolved)

        print(f"  {_BOLD}Resolved spans:{_RESET}  {len(chunk_resolved)}/{len(span_descs)}"
              + (f"  {_RED}({n_rejected_resolve} rejected){_RESET}" if n_rejected_resolve else ""))
        print()
        _print_resolved_spans(chunk_resolved, chunk.text)

        if n_rejected_resolve:
            print()
            rejected_descs = _find_rejected_descriptors(span_descs, chunk_resolved, chunk.text)
            for s, reason in rejected_descs:
                print(f"  {_RED}✗{_RESET} <{s.element}> {repr(s.text[:60])} — {reason}")

        print()

        # 3e: shift to global offsets
        for s in chunk_resolved:
            s.start += chunk.start_offset
            s.end += chunk.start_offset

        # 3f: validate
        before_validate = list(chunk_resolved)
        chunk_validated = validate_spans(chunk_resolved, schema, plain_text)
        n_rejected_validate = len(before_validate) - len(chunk_validated)

        print(f"  {_BOLD}Validated spans:{_RESET}  {len(chunk_validated)}/{len(before_validate)}"
              + (f"  {_RED}({n_rejected_validate} rejected by schema){_RESET}" if n_rejected_validate else ""))

        if n_rejected_validate:
            print()
            rejected_resolved = _find_rejected_resolved(before_validate, chunk_validated, plain_text, schema)
            for s, reason in rejected_resolved:
                print(f"  {_RED}✗{_RESET} <{s.element}> [{s.start}:{s.end}] — {reason}")

        print()
        all_resolved.extend(chunk_validated)

    # ── Dedup / merge ──────────────────────────────────────────────────────
    _h2("STEP 4  Deduplication & merge")
    seen: set[tuple] = set()
    deduped = []
    for s in all_resolved:
        key = (s.element, s.start, s.end)
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    dupes = len(all_resolved) - len(deduped)
    if dupes:
        _warn(f"Removed {dupes} duplicate span(s)")

    # Merge overlapping same-element spans
    processed: set[int] = set()
    merged: list[ResolvedSpan] = []
    for i, span in enumerate(deduped):
        if i in processed:
            continue
        overlapping = [span]
        for j, other in enumerate(deduped[i + 1:], start=i + 1):
            if j in processed:
                continue
            if other.element == span.element and not (other.start >= span.end or span.start >= other.end):
                overlapping.append(other)
                processed.add(j)
        if len(overlapping) > 1:
            ms = min(s.start for s in overlapping)
            me = max(s.end for s in overlapping)
            _warn(f"Merged {len(overlapping)} overlapping <{span.element}> spans → [{ms}:{me}]")
            merged.append(ResolvedSpan(
                element=span.element, start=ms, end=me,
                attrs=span.attrs.copy(), children=[],
                fuzzy_match=any(s.fuzzy_match for s in overlapping),
            ))
        else:
            merged.append(span)

    n_final = len(merged)
    print(f"  {n_final} span(s) after dedup/merge:")
    print()
    _print_resolved_spans(merged, plain_text)
    print()

    # ── Inject XML ────────────────────────────────────────────────────────
    _h2("STEP 5  inject_xml")
    annotated = inject_xml(plain_text, merged)
    final_xml = _escape_bare_ampersands(annotated)

    fuzzy = [s for s in merged if s.fuzzy_match]
    if fuzzy:
        _warn(f"{len(fuzzy)} span(s) used fuzzy context matching (review carefully):")
        for s in fuzzy:
            print(f"    <{s.element}> [{s.start}:{s.end}]  {repr(plain_text[s.start:s.end][:60])}")
        print()

    # ── Final XML ─────────────────────────────────────────────────────────
    _h2("FINAL OUTPUT")
    print(final_xml)
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    from tei_annotator.providers import _ALL_CONNECTORS
    from tei_annotator.schemas.registry import get_schema_names

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", metavar="TEXT", help="Text snippet to annotate")
    src.add_argument("--file", metavar="PATH", help="File containing the text snippet")

    p.add_argument(
        "--schema",
        choices=sorted(get_schema_names()),
        default="bibl",
        help="Annotation schema (default: bibl)",
    )
    p.add_argument(
        "--provider",
        choices=[c.id for c in _ALL_CONNECTORS],
        default=None,
        help="Provider ID (default: first available, preferring gemini)",
    )
    p.add_argument(
        "--model",
        default=None,
        metavar="MODEL_ID",
        help="Model ID override (default: gemini-2.5-flash for gemini, else connector default)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        metavar="N",
        help="Max chars per chunk (default: 1500)",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        metavar="N",
        help="Overlap between chunks (default: 200)",
    )
    p.add_argument(
        "--show-prompt",
        action="store_true",
        default=False,
        help="Print the full LLM prompt (can be very long)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=120,
        metavar="SECONDS",
        help="HTTP timeout in seconds (default: 120)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.providers import _ALL_CONNECTORS, get_available_connectors, get_connector
    from tei_annotator.schemas.registry import build_schema

    # ── Resolve text input ────────────────────────────────────────────────
    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("ERROR: provide text via --text, --file, or stdin", file=sys.stderr)
        return 1

    text = text.strip()
    if not text:
        print("ERROR: empty input", file=sys.stderr)
        return 1

    # ── Resolve provider ──────────────────────────────────────────────────
    if args.provider:
        connector = get_connector(args.provider)
        if connector is None:
            _err(f"Unknown provider {args.provider!r}")
            return 1
        if not connector.is_available():
            _err(f"Provider {args.provider!r} not available — check .env")
            return 1
    else:
        # Prefer gemini, then first available
        available = get_available_connectors()
        if not available:
            _err("No providers available — check your .env file")
            return 1
        connector = next((c for c in available if c.id == "gemini"), available[0])

    model = args.model or ("gemini-2.5-flash" if connector.id == "gemini" else connector.default_model)
    call_fn = connector.make_call_fn(model, timeout=args.timeout)

    schema = build_schema(args.schema)
    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=call_fn,
    )

    # ── Header ────────────────────────────────────────────────────────────
    _h1(f"ANNOTATION DEBUG  |  schema: {args.schema}  |  {connector.name} / {model}")
    print(f"  Input text: {len(text)} chars\n")
    print(f"  {text[:200]}{'...' if len(text) > 200 else ''}\n")

    # ── Run debug pipeline ────────────────────────────────────────────────
    try:
        run_debug(
            text=text,
            schema=schema,
            endpoint=endpoint,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            show_prompt=args.show_prompt,
        )
    except KeyboardInterrupt:
        print("\n\n  [interrupted]")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
