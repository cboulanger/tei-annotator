"""
evaluator.py — High-level evaluation entry points.

evaluate_bibl(gold_element, schema, endpoint, ...)
    Evaluate annotation of a single XML element against its gold standard.

evaluate_file(gold_xml_path, schema, endpoint, ...)
    Evaluate annotation of every <bibl> in a TEI file's <listBibl>.
    Returns per-record results and corpus-level aggregated metrics.

Both functions follow the same pipeline:
  1. Extract gold spans from the gold element (character offsets in plain text).
  2. Strip all tags → plain text (same text the annotator will see).
  3. Run annotate() on the plain text.
  4. Wrap the annotated XML in a synthetic root, parse it, extract spans.
  5. Match predicted spans against gold spans.
  6. Return an EvaluationResult with P/R/F1.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from lxml import etree

# Matches a well-formed XML tag (open/close/self-closing) or comment.
# Used to identify tag boundaries when escaping non-schema angle-brackets.
_XML_TAG_RE = re.compile(
    r"<(/?)([a-zA-Z_][\w:.-]*)(\s[^<>\"']*(?:(?:\"[^\"]*\"|'[^']*')[^<>\"']*)*)?/?>|<!--.*?-->",
    re.DOTALL,
)


def _escape_nonschema_brackets(fragment: str, allowed_tags: frozenset[str]) -> str:
    """
    Escape ``<`` / ``>`` characters in *fragment* whose element name is NOT in
    *allowed_tags* (i.e. not injected by the annotator).

    This prevents spurious elements from literal text like ``<italic>`` that
    lxml converts from XML entities (``&lt;italic&gt;``) in gold-standard files.
    Tags with names in *allowed_tags* (schema elements) are left untouched.
    Comments (``<!-- ... -->``) are also left untouched.
    """
    parts: list[str] = []
    last = 0
    for m in _XML_TAG_RE.finditer(fragment):
        text = fragment[last : m.start()]
        parts.append(text.replace("<", "&lt;").replace(">", "&gt;"))
        # m.group(2) is the tag name; group(2) is None for comments
        tag_name = m.group(2)
        if tag_name is None or tag_name in allowed_tags:
            parts.append(m.group())   # keep it as real XML
        else:
            parts.append(m.group().replace("<", "&lt;").replace(">", "&gt;"))
        last = m.end()
    text = fragment[last:]
    parts.append(text.replace("<", "&lt;").replace(">", "&gt;"))
    return "".join(parts)

from ..inference.endpoint import EndpointConfig
from ..models.schema import TEISchema
from ..pipeline import annotate
from .extractor import extract_spans
from .metrics import EvaluationResult, MatchMode, aggregate, compute_metrics

# TEI namespace used in documents like blbl-examples.tei.xml
_TEI_NS = "http://www.tei-c.org/ns/1.0"


def evaluate_bibl(
    gold_element: etree._Element,
    schema: TEISchema,
    endpoint: EndpointConfig,
    gliner_model: str | None = None,
    match_mode: MatchMode = MatchMode.TEXT,
    overlap_threshold: float = 0.5,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> EvaluationResult:
    """
    Evaluate annotation quality for a single XML element.

    Parameters
    ----------
    gold_element :
        An lxml element with manually annotated child tags (the gold standard).
    schema :
        TEISchema describing the elements that the annotator should produce.
    endpoint :
        Injected inference dependency passed unchanged to :func:`annotate`.
    gliner_model :
        GLiNER model ID for the optional pre-detection pass.
        Defaults to ``None`` (disabled) — enable for real-world runs.
    match_mode :
        How to decide whether a predicted span matches a gold span.
    overlap_threshold :
        IoU threshold when *match_mode* is OVERLAP.
    chunk_size, chunk_overlap :
        Chunking parameters forwarded to :func:`annotate`.

    Returns
    -------
    :class:`~tei_annotator.evaluation.metrics.EvaluationResult`
    """
    # Step 1 — extract gold spans (and the plain text they are anchored to)
    plain_text, gold_spans = extract_spans(gold_element)

    if not plain_text.strip():
        return compute_metrics([], [])

    # Step 2 — annotate the plain text
    result = annotate(
        text=plain_text,
        schema=schema,
        endpoint=endpoint,
        gliner_model=gliner_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Step 3 — parse the annotated XML output
    # annotate() returns a fragment (no root element), so we wrap it.
    # Escape any '<'/'>' whose tag name is not in the schema — these are
    # literal text characters that lxml would otherwise parse as elements
    # (e.g. &lt;italic&gt; in gold-standard bibls becomes raw '<italic>').
    allowed_tags = frozenset(e.tag for e in schema.elements)
    safe_xml = _escape_nonschema_brackets(result.xml, allowed_tags)
    try:
        pred_root = etree.fromstring(f"<_root>{safe_xml}</_root>".encode())
    except etree.XMLSyntaxError as exc:
        warnings.warn(
            f"Could not parse annotator output as XML; treating as empty: {exc}",
            stacklevel=2,
        )
        return compute_metrics(gold_spans, [])

    # Step 4 — extract predicted spans from the parsed output
    _, pred_spans = extract_spans(pred_root)

    # Step 5 — match and compute metrics
    return compute_metrics(
        gold_spans,
        pred_spans,
        mode=match_mode,
        overlap_threshold=overlap_threshold,
    )


def evaluate_file(
    gold_xml_path: str | Path,
    schema: TEISchema,
    endpoint: EndpointConfig,
    root_element: str = "listBibl",
    child_element: str = "bibl",
    gliner_model: str | None = None,
    match_mode: MatchMode = MatchMode.TEXT,
    overlap_threshold: float = 0.5,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    max_items: int | None = None,
) -> tuple[list[EvaluationResult], EvaluationResult]:
    """
    Evaluate annotation quality against a gold-standard TEI XML file.

    Finds every ``<child_element>`` inside ``<root_element>``, strips its
    tags to obtain plain text, runs :func:`annotate`, and compares the result
    to the original annotation.

    Parameters
    ----------
    gold_xml_path :
        Path to a TEI XML file (e.g. ``tests/fixtures/blbl-examples.tei.xml``).
    schema :
        TEISchema to use for annotation.
    endpoint :
        Inference endpoint configuration.
    root_element :
        Container element name to search for (default: ``"listBibl"``).
    child_element :
        Individual record element name to annotate (default: ``"bibl"``).
    gliner_model :
        GLiNER model ID, or ``None`` to disable.
    match_mode :
        Span matching criterion.
    overlap_threshold :
        IoU threshold for OVERLAP mode.
    chunk_size, chunk_overlap :
        Chunking parameters forwarded to :func:`annotate`.
    max_items :
        If set, only the first *max_items* child elements are evaluated.
        Useful for quick smoke runs.

    Returns
    -------
    (per_record_results, aggregated_result)
        *per_record_results* — one :class:`EvaluationResult` per child element.
        *aggregated_result*  — corpus-level metrics (TP/FP/FN summed across
        all records, then P/R/F1 computed from those totals).
    """
    tree = etree.parse(str(gold_xml_path))

    def _find(tag: str) -> list[etree._Element]:
        """Search with TEI namespace first, then without."""
        elems = tree.findall(f".//{{{_TEI_NS}}}{tag}")
        return elems or tree.findall(f".//{tag}")

    containers = _find(root_element)
    all_children: list[etree._Element] = []
    for container in containers:
        children = container.findall(f"{{{_TEI_NS}}}{child_element}")
        if not children:
            children = container.findall(child_element)
        all_children.extend(children)

    if max_items is not None:
        all_children = all_children[:max_items]

    per_record: list[EvaluationResult] = []
    for element in all_children:
        result = evaluate_bibl(
            gold_element=element,
            schema=schema,
            endpoint=endpoint,
            gliner_model=gliner_model,
            match_mode=match_mode,
            overlap_threshold=overlap_threshold,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        per_record.append(result)

    aggregated = aggregate(per_record)
    return per_record, aggregated
