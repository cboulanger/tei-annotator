"""
tei.py — Parse a RELAX NG schema and produce a TEISchema for use with annotate().
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from .models.schema import TEIAttribute, TEIElement, TEISchema

# Namespace URIs
_RNG_NS = "http://relaxng.org/ns/structure/1.0"
_A_NS = "http://relaxng.org/ns/compatibility/annotations/1.0"

_RNG = f"{{{_RNG_NS}}}"


def _local(tag: str) -> str:
    """Strip the Clark-notation namespace from a tag, returning just the local name."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def _get_doc(node: etree._Element) -> str:
    """Return the text of the first a:documentation child of *node*, or ''."""
    doc_tag = f"{{{_A_NS}}}documentation"
    for child in node:
        if isinstance(child.tag, str) and child.tag == doc_tag:
            return (child.text or "").strip()
    return ""


# ---------------------------------------------------------------------------
# Build lookup tables from the parsed RNG tree
# ---------------------------------------------------------------------------


def _build_defines(root: etree._Element) -> dict[str, etree._Element]:
    """Map every <define name="..."> in the schema to its element node."""
    defines: dict[str, etree._Element] = {}
    for define in root.iter(f"{_RNG}define"):
        name = define.get("name")
        if name:
            defines[name] = define
    return defines


def _build_elem_to_define(defines: dict[str, etree._Element]) -> dict[str, str]:
    """Map TEI element names (e.g. 'persName') to their RNG define names."""
    result: dict[str, str] = {}
    for def_name, def_node in defines.items():
        elem = def_node.find(f"{_RNG}element")
        if elem is not None:
            tei_name = elem.get("name")
            if tei_name:
                result[tei_name] = def_name
    return result


# ---------------------------------------------------------------------------
# Content-model traversal: collect child element names
# ---------------------------------------------------------------------------


def _collect_child_elements(
    node: etree._Element,
    defines: dict[str, etree._Element],
    visited: set[str],
) -> list[str]:
    """
    Walk *node* (an RNG define or structural node) and return the names of all
    TEI elements that can appear as immediate children according to the content
    model.

    - Follows <ref> nodes recursively to expand macros and model classes.
    - Stops at <element> boundaries (their inner content is not traversed).
    - Skips attribute-group refs (those whose name contains ".att" or "att.").
    """
    result: list[str] = []
    for child in node:
        if not isinstance(child.tag, str):
            continue
        local = _local(child.tag)

        if local == "element":
            # This RNG <element> defines a TEI child element — record its name.
            name = child.get("name")
            if name:
                result.append(name)
            # Do NOT recurse into the element's own content (those are grandchildren).

        elif local == "ref":
            ref_name = child.get("name", "")
            if not ref_name:
                continue
            # Skip attribute-group references — they contribute no child elements.
            if "att." in ref_name or ref_name not in defines:
                continue
            target = defines[ref_name]
            # If the define directly wraps a TEI <element>, record its name and stop —
            # do NOT recurse into the element's own content (those are grandchildren).
            # This also handles self-referential elements (e.g. idno containing idno).
            elem_child = target.find(f"{_RNG}element")
            if elem_child is not None:
                name = elem_child.get("name")
                if name:
                    result.append(name)
            elif ref_name not in visited:
                # It's a model/macro group — recurse to expand it.
                visited.add(ref_name)
                result.extend(_collect_child_elements(target, defines, visited))

        elif local in ("notAllowed", "empty", "text", "data", "param", "value", "attribute"):
            # Terminal or attribute nodes — no child elements here.
            continue

        else:
            # Structural wrappers: choice, group, interleave, optional,
            # zeroOrMore, oneOrMore, grammar, …
            result.extend(_collect_child_elements(child, defines, visited))

    return result


# ---------------------------------------------------------------------------
# Attribute traversal: collect TEIAttribute instances
# ---------------------------------------------------------------------------


def _collect_attributes(
    node: etree._Element,
    defines: dict[str, etree._Element],
    visited: set[str],
) -> list[TEIAttribute]:
    """
    Walk *node* and collect all <attribute> elements, following only
    attribute-group <ref> nodes (those whose name contains "att.").
    Inline <attribute> elements directly in the element definition are also
    picked up.
    """
    result: list[TEIAttribute] = []
    for child in node:
        if not isinstance(child.tag, str):
            continue
        local = _local(child.tag)

        if local == "attribute":
            attr = _parse_attribute(child, node)
            if attr is not None:
                result.append(attr)

        elif local == "ref":
            ref_name = child.get("name", "")
            if not ref_name or ref_name in visited:
                continue
            # Only follow attribute-group refs.
            if "att." not in ref_name:
                continue
            if ref_name not in defines:
                continue
            visited.add(ref_name)
            result.extend(_collect_attributes(defines[ref_name], defines, visited))

        elif local == "element":
            # Don't descend into nested element definitions.
            continue

        else:
            # Structural wrappers: optional, choice, group, etc.
            result.extend(_collect_attributes(child, defines, visited))

    return result


def _parse_attribute(
    attr_node: etree._Element,
    parent: etree._Element,
) -> TEIAttribute | None:
    """
    Convert an RNG <attribute> element to a TEIAttribute.

    *parent* is the direct parent of *attr_node* in the RNG tree; it is used
    to determine whether the attribute is required (i.e. not wrapped in
    <optional> or <zeroOrMore>).
    """
    name = attr_node.get("name")
    if not name:
        return None

    description = _get_doc(attr_node)
    required = _local(parent.tag) not in ("optional", "zeroOrMore")

    # Collect explicit enumerated values from a <choice> inside the attribute.
    allowed_values: list[str] | None = None
    for choice_node in attr_node.iter(f"{_RNG}choice"):
        values = [
            v.text.strip()
            for v in choice_node.findall(f"{_RNG}value")
            if v.text and v.text.strip()
        ]
        if values:
            allowed_values = values
            break

    return TEIAttribute(
        name=name,
        description=description,
        required=required,
        allowed_values=allowed_values,
    )


# ---------------------------------------------------------------------------
# Build a single TEIElement
# ---------------------------------------------------------------------------


def _build_tei_element(
    elem_name: str,
    defines: dict[str, etree._Element],
    elem_to_def: dict[str, str],
) -> TEIElement | None:
    """Construct a TEIElement for the named TEI element, or None if not found."""
    def_name = elem_to_def.get(elem_name)
    if not def_name:
        return None

    def_node = defines[def_name]
    elem_node = def_node.find(f"{_RNG}element")
    if elem_node is None:
        return None

    description = _get_doc(elem_node)

    # Child elements (content-model expansion).
    child_visited: set[str] = {def_name}
    children = _collect_child_elements(elem_node, defines, child_visited)

    # Attributes (attribute-group expansion + inline attributes).
    attr_visited: set[str] = {def_name}
    attributes = _collect_attributes(elem_node, defines, attr_visited)

    # Deduplicate children and attributes while preserving order.
    seen_children: set[str] = set()
    unique_children: list[str] = []
    for c in children:
        if c not in seen_children:
            seen_children.add(c)
            unique_children.append(c)

    seen_attrs: set[str] = set()
    unique_attrs: list[TEIAttribute] = []
    for a in attributes:
        if a.name not in seen_attrs:
            seen_attrs.add(a.name)
            unique_attrs.append(a)

    return TEIElement(
        tag=elem_name,
        description=description,
        allowed_children=unique_children,
        attributes=unique_attrs,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_schema(
    schema_path: str | Path,
    element: str = "text",
    depth: int = 1,
) -> TEISchema:
    """
    Parse a RELAX NG (.rng) schema and build a :class:`TEISchema` suitable for
    passing to :func:`tei_annotator.annotate`.

    Parameters
    ----------
    schema_path:
        Path to the ``.rng`` file.
    element:
        Name of the root TEI element to start from (default: ``"text"``).
    depth:
        How many levels of descendant elements to include in the returned
        schema.  ``depth=1`` includes the root element **and** its direct
        children; ``depth=2`` also includes their children; and so on.

    Returns
    -------
    TEISchema
        A :class:`TEISchema` containing :class:`TEIElement` entries for the
        root element and all descendants up to *depth* levels deep.

    Raises
    ------
    ValueError
        If *element* is not defined in the schema.
    """
    tree = etree.parse(str(schema_path))
    root = tree.getroot()

    defines = _build_defines(root)
    elem_to_def = _build_elem_to_define(defines)

    if element not in elem_to_def:
        raise ValueError(
            f"Element '{element}' not found in schema '{schema_path}'. "
            f"Available elements: {sorted(elem_to_def)[:10]} …"
        )

    # BFS: process level by level up to `depth` levels deep.
    tei_elements: list[TEIElement] = []
    seen: set[str] = set()
    current_level: list[str] = [element]

    for level in range(depth + 1):
        next_level: list[str] = []
        for elem_name in current_level:
            if elem_name in seen:
                continue
            seen.add(elem_name)

            tei_elem = _build_tei_element(elem_name, defines, elem_to_def)
            if tei_elem is None:
                continue
            tei_elements.append(tei_elem)

            # Queue children for the next level (if we haven't hit the depth limit).
            if level < depth:
                next_level.extend(tei_elem.allowed_children)

        current_level = next_level

    return TEISchema(elements=tei_elements)
