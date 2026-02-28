from __future__ import annotations

import warnings
from dataclasses import dataclass, field

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

        # 5a. Resolve within chunk text → positions relative to chunk
        chunk_resolved = resolve_spans(chunk.text, span_descs)

        # 5b. Shift to global (plain_text) offsets
        for span in chunk_resolved:
            span.start += chunk.start_offset
            span.end += chunk.start_offset

        # 5c. Validate against schema
        chunk_resolved = validate_spans(chunk_resolved, schema, plain_text)

        all_resolved.extend(chunk_resolved)

    # ------------------------------------------------------------------ #
    # Deduplicate spans that appeared in overlapping chunks               #
    # ------------------------------------------------------------------ #
    seen: set[tuple[str, int, int]] = set()
    deduped: list[ResolvedSpan] = []
    for span in all_resolved:
        key = (span.element, span.start, span.end)
        if key not in seen:
            seen.add(key)
            deduped.append(span)

    # ------------------------------------------------------------------ #
    # STEP 5d  Inject XML tags into the plain text                        #
    # ------------------------------------------------------------------ #
    annotated_text = inject_xml(plain_text, deduped)

    # ------------------------------------------------------------------ #
    # STEP 5d (cont.)  Restore original XML tags                          #
    # ------------------------------------------------------------------ #
    final_xml = _restore_existing_tags(annotated_text, restore_map)

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
