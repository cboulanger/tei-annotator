from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

from .chunking.chunker import chunk_text
from .inference.endpoint import EndpointCapability, EndpointConfig
from .models.schema import TEISchema
from .models.spans import ResolvedSpan, SpanDescriptor
from .postprocessing.injector import inject_xml
from .postprocessing.parser import parse_response
from .postprocessing.resolver import resolve_spans
from .postprocessing.validator import validate_spans
from .prompting.builder import build_prompt, make_correction_prompt


@dataclass
class AnnotationResult:
    xml: str
    fuzzy_spans: list[ResolvedSpan] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _TagEntry:
    plain_offset: int  # position in plain text before which this tag should be re-inserted
    tag: str


def _strip_existing_tags(text: str) -> tuple[str, list[_TagEntry]]:
    """
    Remove XML tags from *text*.

    Returns (plain_text, restore_map) where restore_map records each stripped
    tag and the plain-text offset at which it should be re-inserted.
    """
    plain: list[str] = []
    restore: list[_TagEntry] = []
    i = 0
    while i < len(text):
        if text[i] == "<":
            j = text.find(">", i)
            if j != -1:
                restore.append(_TagEntry(plain_offset=len(plain), tag=text[i : j + 1]))
                i = j + 1
            else:
                plain.append(text[i])
                i += 1
        else:
            plain.append(text[i])
            i += 1
    return "".join(plain), restore


def _restore_existing_tags(annotated_xml: str, restore_map: list[_TagEntry]) -> str:
    """
    Re-insert original XML tags into *annotated_xml*.

    The tags are keyed by their position in the *plain text* (before annotation),
    so we walk the annotated XML tracking plain-text position (i.e. advancing only
    on non-tag characters).
    """
    if not restore_map:
        return annotated_xml

    inserts: dict[int, list[str]] = {}
    for entry in restore_map:
        inserts.setdefault(entry.plain_offset, []).append(entry.tag)

    result: list[str] = []
    plain_pos = 0
    i = 0

    while i < len(annotated_xml):
        # Flush any original tags due at the current plain position
        for tag in inserts.pop(plain_pos, []):
            result.append(tag)

        if annotated_xml[i] == "<":
            # Existing (newly injected) tag — copy verbatim, don't advance plain_pos
            j = annotated_xml.find(">", i)
            if j != -1:
                result.append(annotated_xml[i : j + 1])
                i = j + 1
            else:
                result.append(annotated_xml[i])
                plain_pos += 1
                i += 1
        else:
            result.append(annotated_xml[i])
            plain_pos += 1
            i += 1

    # Flush any remaining original tags (e.g. trailing tags in the original)
    for pos in sorted(inserts.keys()):
        for tag in inserts[pos]:
            result.append(tag)

    return "".join(result)


_ENTITY_RE = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);")


def _escape_bare_ampersands(xml: str) -> str:
    """Replace bare & (not part of a valid entity reference) with &amp; in text nodes."""
    result: list[str] = []
    i = 0
    while i < len(xml):
        if xml[i] == "<":
            j = xml.find(">", i)
            if j != -1:
                result.append(xml[i : j + 1])
                i = j + 1
            else:
                result.append(xml[i])
                i += 1
        elif xml[i] == "&":
            m = _ENTITY_RE.match(xml, i)
            if m:
                result.append(m.group())
                i += len(m.group())
            else:
                result.append("&amp;")
                i += 1
        else:
            result.append(xml[i])
            i += 1
    return "".join(result)


def _run_gliner(
    text: str,
    schema: TEISchema,
    model_id: str,
) -> list[SpanDescriptor]:
    """Run GLiNER detection; returns [] if the optional dependency is missing."""
    try:
        from .detection.gliner_detector import detect_spans

        return detect_spans(text, schema, model_id)
    except ImportError:
        warnings.warn(
            "gliner is not installed; skipping GLiNER pre-detection pass. "
            "Install it with: pip install tei-annotator[gliner]",
            stacklevel=3,
        )
        return []


def preload_gliner_model(model_id: str) -> None:
    """
    Load and cache a GLiNER model before the first :func:`annotate` call.

    Calling this explicitly avoids paying the model-loading cost inside the
    first annotation of a batch run.  Safe to call multiple times for the same
    *model_id* — subsequent calls are no-ops.

    Raises a :class:`UserWarning` (rather than an error) if the ``gliner``
    extra is not installed.
    """
    try:
        from .detection.gliner_detector import preload_model

        preload_model(model_id)
    except ImportError:
        warnings.warn(
            "gliner is not installed; cannot preload GLiNER model. "
            "Install it with: pip install tei-annotator[gliner]",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate(
    text: str,
    schema: TEISchema,
    endpoint: EndpointConfig,
    gliner_model: str | None = "numind/NuNER_Zero",
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> AnnotationResult:
    """
    Annotate *text* with TEI XML tags using a two-stage LLM pipeline.

    The source text is **never modified** — models only contribute tag positions
    and attribute values.  All text in the output comes from the original input.

    Parameters
    ----------
    text:
        Input text, which may already contain partial XML markup.
    schema:
        A TEISchema describing the elements (and their attributes) in scope.
    endpoint:
        Injected inference dependency (wraps any call_fn: str → str).
    gliner_model:
        HuggingFace model ID for the optional GLiNER pre-detection pass.
        Pass None to disable.
    chunk_size:
        Maximum characters per chunk sent to the LLM.
    chunk_overlap:
        Characters of overlap between consecutive chunks.
    """
    # ------------------------------------------------------------------ #
    # STEP 1  Strip existing XML tags; save restoration map               #
    # ------------------------------------------------------------------ #
    plain_text, restore_map = _strip_existing_tags(text)

    # ------------------------------------------------------------------ #
    # STEP 2  Optional GLiNER pre-detection pass                          #
    # ------------------------------------------------------------------ #
    gliner_candidates: list[SpanDescriptor] = []
    if (
        gliner_model is not None
        and endpoint.capability != EndpointCapability.EXTRACTION
        and len(plain_text) > 200
    ):
        gliner_candidates = _run_gliner(plain_text, schema, gliner_model)

    # ------------------------------------------------------------------ #
    # STEPS 3–5  Chunk → prompt → infer → postprocess                     #
    # ------------------------------------------------------------------ #
    chunks = chunk_text(plain_text, chunk_size=chunk_size, overlap=chunk_overlap)
    log.debug("annotate: plain_text length=%d, %d chunk(s)", len(plain_text), len(chunks))
    all_resolved: list[ResolvedSpan] = []

    for chunk in chunks:
        # Narrow GLiNER candidates to those plausibly within this chunk
        chunk_candidates: list[SpanDescriptor] | None = None
        if gliner_candidates:
            chunk_candidates = [
                c
                for c in gliner_candidates
                if c.context and chunk.text.find(c.context[:30]) != -1
            ] or None

        log.debug(
            "chunk offset=%d len=%d",
            chunk.start_offset, len(chunk.text),
        )

        # 3. Build prompt / raw request
        if endpoint.capability == EndpointCapability.EXTRACTION:
            raw_response = endpoint.call_fn(chunk.text)
        else:
            prompt = build_prompt(
                source_text=chunk.text,
                schema=schema,
                capability=endpoint.capability,
                candidates=chunk_candidates,
            )
            raw_response = endpoint.call_fn(prompt)

        # 4. Parse response → SpanDescriptors
        retry_fn = (
            endpoint.call_fn
            if endpoint.capability == EndpointCapability.TEXT_GENERATION
            else None
        )
        correction_fn = (
            make_correction_prompt
            if endpoint.capability == EndpointCapability.TEXT_GENERATION
            else None
        )
        try:
            span_descs = parse_response(
                raw_response,
                call_fn=retry_fn,
                make_correction_prompt=correction_fn,
            )
        except ValueError:
            warnings.warn(
                f"Could not parse LLM response for chunk at offset "
                f"{chunk.start_offset}; skipping chunk.",
                stacklevel=2,
            )
            continue

        log.debug(
            "chunk offset=%d: LLM returned %d span descriptor(s): %s",
            chunk.start_offset,
            len(span_descs),
            [(s.element, repr(s.text[:40])) for s in span_descs],
        )

        # 5a. Resolve within chunk text → positions relative to chunk
        chunk_resolved = resolve_spans(chunk.text, span_descs)

        rejected = len(span_descs) - len(chunk_resolved)
        log.debug(
            "chunk offset=%d: resolved %d/%d span(s), %d rejected by resolver",
            chunk.start_offset, len(chunk_resolved), len(span_descs), rejected,
        )
        for s in chunk_resolved:
            log.debug(
                "  resolved: <%s> [%d:%d] text=%r",
                s.element, s.start, s.end, chunk.text[s.start:s.end][:60],
            )
        if rejected:
            # Find which SpanDescriptors weren't resolved by checking coverage
            resolved_ranges = {(s.start, s.end) for s in chunk_resolved}
            for s in span_descs:
                # Locate where this descriptor's text appears in the chunk
                pos = chunk.text.find(s.text)
                if pos == -1 or (pos, pos + len(s.text)) not in resolved_ranges:
                    log.debug(
                        "  rejected: <%s> text=%r context=%r",
                        s.element, s.text[:80], s.context[:80],
                    )

        # Warn if many spans were rejected (likely resolver context mismatch)
        if len(span_descs) > 0 and len(chunk_resolved) < len(span_descs) * 0.5:
            warnings.warn(
                f"Chunk at offset {chunk.start_offset}: {len(span_descs)} spans detected "
                f"but only {len(chunk_resolved)} resolved. This may indicate context mismatch "
                f"(LLM context includes XML tags stripped from plain text).",
                stacklevel=2,
            )

        # 5b. Shift to global (plain_text) offsets
        for span in chunk_resolved:
            span.start += chunk.start_offset
            span.end += chunk.start_offset

        # 5c. Validate against schema
        before_validate = len(chunk_resolved)
        chunk_resolved = validate_spans(chunk_resolved, schema, plain_text)
        if len(chunk_resolved) < before_validate:
            log.debug(
                "chunk offset=%d: %d span(s) dropped by schema validator",
                chunk.start_offset, before_validate - len(chunk_resolved),
            )

        all_resolved.extend(chunk_resolved)

    # ------------------------------------------------------------------ #
    # Deduplicate and merge spans from overlapping chunks                 #
    # ------------------------------------------------------------------ #
    log.debug("post-chunk: %d total span(s) across all chunks before dedup", len(all_resolved))

    # First pass: deduplicate identical spans
    seen: set[tuple[str, int, int]] = set()
    deduped: list[ResolvedSpan] = []
    for span in all_resolved:
        key = (span.element, span.start, span.end)
        if key not in seen:
            seen.add(key)
            deduped.append(span)

    if len(deduped) < len(all_resolved):
        log.debug("dedup: removed %d exact duplicate(s)", len(all_resolved) - len(deduped))

    # Second pass: merge overlapping spans with the same element
    merged: list[ResolvedSpan] = []
    processed = set()

    for i, span in enumerate(deduped):
        if i in processed:
            continue

        # Find all spans that overlap with this one and have the same element
        overlapping = [span]
        for j, other in enumerate(deduped[i+1:], start=i+1):
            if j in processed:
                continue
            if other.element == span.element:
                # Check if they overlap
                if not (other.start >= span.end or span.start >= other.end):
                    overlapping.append(other)
                    processed.add(j)

        if len(overlapping) > 1:
            # Merge overlapping spans by extending boundaries
            merged_start = min(s.start for s in overlapping)
            merged_end = max(s.end for s in overlapping)
            log.debug(
                "merge: %d overlapping <%s> spans → [%d:%d] (was %s)",
                len(overlapping), span.element, merged_start, merged_end,
                [(s.start, s.end) for s in overlapping],
            )
            merged_span = ResolvedSpan(
                element=span.element,
                start=merged_start,
                end=merged_end,
                attrs=span.attrs.copy(),
                children=[],
                fuzzy_match=any(s.fuzzy_match for s in overlapping),
            )
            merged.append(merged_span)
        else:
            merged.append(span)

    deduped = merged
    log.debug("final: %d span(s) after dedup+merge: %s", len(deduped), [(s.element, s.start, s.end) for s in deduped])

    # ------------------------------------------------------------------ #
    # STEP 5d  Inject XML tags into the plain text                        #
    # ------------------------------------------------------------------ #
    log.debug("inject_xml: injecting %d span(s) into plain text", len(deduped))
    annotated_text = inject_xml(plain_text, deduped)
    log.debug("inject_xml: done, annotated length=%d", len(annotated_text))

    # ------------------------------------------------------------------ #
    # STEP 5d (cont.)  Restore original XML tags                          #
    # ------------------------------------------------------------------ #
    log.debug("restore_tags: %d original tag(s) to restore", len(restore_map))
    final_xml = _restore_existing_tags(annotated_text, restore_map)
    log.debug("restore_tags: done, final length=%d", len(final_xml))

    # ------------------------------------------------------------------ #
    # STEP 5d (cont.)  Escape bare & in text nodes                        #
    # ------------------------------------------------------------------ #
    final_xml = _escape_bare_ampersands(final_xml)

    # ------------------------------------------------------------------ #
    # STEP 5e  Final XML validation (best-effort)                         #
    # ------------------------------------------------------------------ #
    try:
        from lxml import etree

        try:
            etree.fromstring(f"<_root>{final_xml}</_root>".encode())
        except etree.XMLSyntaxError as exc:
            warnings.warn(f"Output XML validation failed: {exc}", stacklevel=2)
    except ImportError:
        pass

    return AnnotationResult(
        xml=final_xml,
        fuzzy_spans=[s for s in deduped if s.fuzzy_match],
    )
