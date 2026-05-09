#!/usr/bin/env python
"""
Collect challenging examples for a given schema by evaluating mini-batches
against a provider and retaining items the model handles poorly.

Strategy
--------
1. Load all child elements (e.g. <bibl>) from source TEI files.
2. Group consecutive elements from the same source container into mini-batches
   of --batch-size (default 10).
3. Wrap each mini-batch in a synthetic container element and call evaluate_element.
4. Skip batches where micro-F1 >= --f1-threshold (well-handled; not interesting).
5. From failing batches keep up to --max-per-batch gold spans that were missed,
   together with --context neighbouring elements for segmentation context.
6. Stop once --limit examples have been collected.
7. Write a TEI file ready to use as a gold fixture.

Usage
-----
    # bibl-reference-segmenter, KISSKI gemma-4-31b-it, 30 hard examples:
    uv run scripts/collect_hard_examples.py \\
        --provider kisski --model gemma-4-31b-it \\
        --output data/hard-bibl-refseg-gemma.tei.xml

    # same but bibl schema, Gemini, 20 examples, verbose:
    uv run scripts/collect_hard_examples.py \\
        --schema bibl --provider gemini --limit 20 --verbose \\
        --output data/hard-bibl-gemini.tei.xml

API keys are read from .env in the project root.
"""

from __future__ import annotations

import argparse
import copy
import glob
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv
from lxml import etree

_REPO = Path(__file__).parent.parent
_TEI_NS = "http://www.tei-c.org/ns/1.0"

load_dotenv(_REPO / ".env")


# ---------------------------------------------------------------------------
# Source data loader
# ---------------------------------------------------------------------------


def _load_source_groups(
    data_glob: str,
    root_element: str,
    child_element: str,
    shuffle: bool = False,
) -> list[tuple[list[etree._Element], str]]:
    """
    Return a list of (children, source_filepath) pairs, one per container element
    found across all files matching *data_glob*.
    """
    import random

    files = sorted(glob.glob(data_glob, recursive=True))
    if not files:
        return []

    groups: list[tuple[list[etree._Element], str]] = []
    for filepath in files:
        tree = etree.parse(filepath)
        containers = tree.findall(f".//{{{_TEI_NS}}}{root_element}")
        if not containers:
            containers = tree.findall(f".//{root_element}")
        for container in containers:
            children = container.findall(f"{{{_TEI_NS}}}{child_element}")
            if not children:
                children = container.findall(child_element)
            if children:
                groups.append((children, filepath))

    if shuffle:
        random.shuffle(groups)
    return groups


# ---------------------------------------------------------------------------
# Synthetic container builder
# ---------------------------------------------------------------------------


def _make_container(tag: str, children: list[etree._Element]) -> etree._Element:
    """Return a new element <tag> with deep-copies of *children* appended."""
    el = etree.Element(f"{{{_TEI_NS}}}{tag}")
    for child in children:
        el.append(copy.deepcopy(child))
    return el


# ---------------------------------------------------------------------------
# Span → XML element matcher
# ---------------------------------------------------------------------------


def _match_span_to_element(
    span_text: str, candidates: list[etree._Element]
) -> etree._Element | None:
    """Return the first candidate whose normalised plain text equals *span_text*."""
    norm = " ".join(span_text.split())
    for el in candidates:
        if " ".join("".join(el.itertext()).split()) == norm:
            return el
    return None


# ---------------------------------------------------------------------------
# TEI output builder
# ---------------------------------------------------------------------------


def _build_output_tei(
    bad_examples: list[dict],
    schema_name: str,
    provider_name: str,
    child_element: str,
) -> etree._Element:
    """
    Assemble a <TEI> element containing one <listBibl> per collected example,
    each with the context window of neighbouring elements.

    Each entry in *bad_examples* is a dict with keys:
        element    – the bad XML element (lxml)
        neighbors  – list of XML elements forming the context window
        f1         – batch micro-F1 where this element failed
        source     – source file path string
    """
    tei = etree.Element("TEI", nsmap={None: _TEI_NS})
    tei.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    header = etree.SubElement(tei, f"{{{_TEI_NS}}}teiHeader")
    fdesc = etree.SubElement(header, f"{{{_TEI_NS}}}fileDesc")
    fdesc.set("{http://www.w3.org/XML/1998/namespace}id", f"_{schema_name}-hard-examples")

    text_el = etree.SubElement(tei, f"{{{_TEI_NS}}}text")
    text_el.text = "\n        "

    for ex in bad_examples:
        bad_el = ex["element"]
        neighbors = ex["neighbors"]
        source = ex["source"]
        f1 = ex["f1"]

        lb = etree.SubElement(text_el, f"{{{_TEI_NS}}}listBibl")
        comment_text = (
            f" F1={f1:.3f}  source={Path(source).name} "
        )
        lb.append(etree.Comment(comment_text))
        lb.text = "\n"

        bad_text = " ".join("".join(bad_el.itertext()).split())

        for neighbor in neighbors:
            neighbor_text = " ".join("".join(neighbor.itertext()).split())
            if neighbor_text == bad_text:
                lb.append(etree.Comment(" ↓ missed by model "))
            el_copy = copy.deepcopy(neighbor)
            lb.append(el_copy)
            el_copy.tail = "\n"

        lb.tail = "\n\n        "

    return tei


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    from tei_annotator.evaluation import evaluate_element, MatchMode
    from tei_annotator.inference.endpoint import EndpointCapability, EndpointConfig
    from tei_annotator.providers import _ALL_CONNECTORS, get_available_connectors, get_connector
    from tei_annotator.schemas.registry import build_schema, get_schema_config

    # --- provider setup ---
    if args.provider == "all":
        connectors = get_available_connectors()
        if not connectors:
            print("ERROR: No providers available — check .env", file=sys.stderr)
            return 1
        connector = connectors[0]
        print(f"Using first available provider: {connector.name}")
    else:
        connector = get_connector(args.provider)
        if connector is None or not connector.is_available():
            print(
                f"ERROR: Provider {args.provider!r} not available — check .env",
                file=sys.stderr,
            )
            return 1

    model = args.model or connector.default_model
    call_fn = connector.make_call_fn(model, timeout=args.timeout)
    endpoint = EndpointConfig(
        capability=EndpointCapability.TEXT_GENERATION,
        call_fn=call_fn,
    )

    mode_map = {"text": MatchMode.TEXT, "exact": MatchMode.EXACT, "overlap": MatchMode.OVERLAP}
    match_mode = mode_map[args.match_mode]

    schema = build_schema(args.schema)
    cfg = get_schema_config(args.schema)
    # For bibl-reference-segmenter: root=text/child=listBibl → we evaluate listBibl records
    # whose *children* are the bibl elements. We want to find hard *child* elements.
    #
    # Generic mapping:
    #   container_element = cfg["child_element"]   (the record element, e.g. listBibl)
    #   item_element      = one level deeper        (e.g. bibl inside listBibl)
    #
    # We load (container_element, item_element) pairs from source files.
    # For schemas where the record IS the smallest unit (e.g. bibl schema where
    # record=bibl and items are sub-elements), we treat each record independently.
    container_element = cfg["child_element"]  # e.g. "listBibl" for bibl-reference-segmenter
    # The item-level element: for bibl-reference-segmenter it is "bibl";
    # fall back to child_element itself when the schema has no deeper level.
    item_element = args.item_element or _infer_item_element(args.schema, cfg)

    data_glob = str(_REPO / args.data_dir / args.data_glob)
    groups = _load_source_groups(
        data_glob=data_glob,
        root_element=container_element,
        child_element=item_element,
        shuffle=args.shuffle,
    )
    if not groups:
        print(f"ERROR: No source files found matching {data_glob}", file=sys.stderr)
        return 1

    total_items = sum(len(g) for g, _ in groups)
    print(
        f"\n  Schema   : {args.schema}"
        f"\n  Provider : {connector.name} / {model}"
        f"\n  Glob     : {data_glob}"
        f"\n  Source   : {len(groups)} containers, {total_items} items"
        f"\n  Batch    : {args.batch_size} items/batch"
        f"\n  Target   : {args.limit} hard examples (F1 < {args.f1_threshold})"
        f"\n  Output   : {args.output}\n"
    )

    bad_examples: list[dict] = []
    # Track (source_file, item_text) pairs to avoid duplicates
    seen: set[tuple[str, str]] = set()

    batches_evaluated = 0
    batches_skipped = 0

    for group_items, source_file in groups:
        if len(bad_examples) >= args.limit:
            break

        # Split this container's items into mini-batches
        for batch_start in range(0, len(group_items), args.batch_size):
            if len(bad_examples) >= args.limit:
                break

            batch = group_items[batch_start : batch_start + args.batch_size]
            if not batch:
                continue

            synthetic = _make_container(container_element, batch)

            snippet = "".join(batch[0].itertext())[:50].replace("\n", " ")
            print(
                f"  [{batches_evaluated + 1}] {Path(source_file).name}"
                f"  items {batch_start + 1}-{batch_start + len(batch)}"
                f"  «{snippet}»",
                end="  ",
                flush=True,
            )

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Output XML validation failed")
                    result = evaluate_element(
                        gold_element=synthetic,
                        schema=schema,
                        endpoint=endpoint,
                        match_mode=match_mode,
                    )
                batches_evaluated += 1
            except Exception as exc:
                print(f"ERROR — {exc}")
                batches_evaluated += 1
                continue

            f1 = result.micro_f1
            print(f"F1={f1:.3f}", flush=True)

            if f1 >= args.f1_threshold:
                batches_skipped += 1
                continue

            # Collect missed item spans (element name = item_element)
            missed = [s for s in result.unmatched_gold if s.element == item_element]
            if not missed:
                continue

            # Sort by position in the text (start offset)
            missed.sort(key=lambda s: s.start)

            collected_this_batch = 0
            for span in missed:
                if collected_this_batch >= args.max_per_batch:
                    break
                if len(bad_examples) >= args.limit:
                    break

                bad_el = _match_span_to_element(span.text, batch)
                if bad_el is None:
                    continue

                dedup_key = (source_file, " ".join(span.text.split()))
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Build context window from original group (not synthetic batch copy)
                try:
                    item_idx = group_items.index(bad_el)
                except ValueError:
                    # bad_el is a copy; find by text
                    item_idx = next(
                        (
                            i
                            for i, el in enumerate(group_items)
                            if " ".join("".join(el.itertext()).split())
                            == " ".join("".join(bad_el.itertext()).split())
                        ),
                        None,
                    )
                if item_idx is None:
                    continue

                ctx_start = max(0, item_idx - args.context)
                ctx_end = min(len(group_items), item_idx + args.context + 1)
                neighbors = group_items[ctx_start:ctx_end]

                bad_examples.append(
                    {
                        "element": group_items[item_idx],
                        "neighbors": neighbors,
                        "f1": f1,
                        "source": source_file,
                    }
                )
                collected_this_batch += 1

                if args.verbose:
                    print(
                        f"    ✗ [{len(bad_examples)}/{args.limit}]"
                        f" «{''.join(bad_el.itertext())[:60].replace(chr(10), ' ')}»"
                    )

    print(
        f"\n  Evaluated {batches_evaluated} batches"
        f"  ({batches_skipped} skipped, F1 >= {args.f1_threshold})"
    )
    print(f"  Collected {len(bad_examples)} hard examples\n")

    if not bad_examples:
        print("  Nothing to write — no hard examples found.")
        return 0

    # --- build and write output TEI ---
    tei = _build_output_tei(
        bad_examples=bad_examples,
        schema_name=args.schema,
        provider_name=f"{connector.name}/{model}",
        child_element=item_element,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = etree.ElementTree(tei)
    tree.write(
        str(out_path),
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
    )
    print(f"  Written → {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Schema introspection helper
# ---------------------------------------------------------------------------


def _infer_item_element(schema_name: str, cfg: dict) -> str:
    """
    Infer the item-level element tag for a schema.

    For bibl-reference-segmenter: container=listBibl, items=bibl.
    For bibl: container=bibl, items also=bibl (each bibl is its own record).
    """
    # bibl-reference-segmenter: records are listBibl, items inside are bibl
    _ITEM_MAP = {
        "bibl-reference-segmenter": "bibl",
        "bibl": "bibl",
    }
    return _ITEM_MAP.get(schema_name, cfg["child_element"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    from tei_annotator.providers import _ALL_CONNECTORS
    from tei_annotator.schemas.registry import get_schema_names

    p = argparse.ArgumentParser(
        description="Collect hard examples for a TEI annotation schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--schema",
        choices=sorted(get_schema_names()),
        default="bibl-reference-segmenter",
        help="Annotation schema to evaluate.",
    )
    p.add_argument(
        "--provider",
        choices=[c.id for c in _ALL_CONNECTORS] + ["all"],
        default="kisski",
        help="LLM provider to use.",
    )
    p.add_argument(
        "--model",
        default=None,
        metavar="MODEL_ID",
        help="Model ID (default: provider's default_model).",
    )
    p.add_argument(
        "--data-dir",
        default="data/raw",
        metavar="DIR",
        help="Root directory for source TEI files (joined with --data-glob).",
    )
    p.add_argument(
        "--data-glob",
        default="**/tei/*.xml",
        metavar="GLOB",
        help="Glob pattern appended to --data-dir to locate source files.",
    )
    p.add_argument(
        "--item-element",
        default=None,
        metavar="TAG",
        help=(
            "XML element tag for the items to collect (e.g. 'bibl'). "
            "Auto-detected from schema when omitted."
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output TEI file path.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=30,
        metavar="N",
        help="Stop after collecting this many hard examples.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        metavar="N",
        help="Number of items per synthetic evaluation batch.",
    )
    p.add_argument(
        "--f1-threshold",
        type=float,
        default=0.95,
        metavar="F",
        help="Skip batches with micro-F1 >= this value.",
    )
    p.add_argument(
        "--max-per-batch",
        type=int,
        default=3,
        metavar="N",
        help="Maximum hard examples to collect from a single batch.",
    )
    p.add_argument(
        "--context",
        type=int,
        default=2,
        metavar="N",
        help="Number of neighbouring items to include around each hard example in the output.",
    )
    p.add_argument(
        "--match-mode",
        choices=["text", "exact", "overlap"],
        default="text",
        help="Span matching criterion.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=120,
        metavar="SECONDS",
        help="HTTP read timeout per API call.",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Randomly shuffle source containers before processing.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print each collected example as it is found.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
