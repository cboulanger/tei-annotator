# Writing effective TEIElement descriptions

`TEIElement.description` is the primary signal the LLM uses to decide whether
and how to annotate a span of text.  A poorly-worded description is usually the
first thing to fix when annotation quality is low.  The guidelines below are
derived from empirical evaluation against a gold-standard bibliography corpus
using Gemini 2.0 Flash and Llama-3.3-70b-instruct.

---

## Core principle: think in spans, not in XML tags

The LLM is asked to **emit spans** — tuples of *(element name, verbatim text,
surrounding context)*.  It never writes raw XML.  Descriptions therefore should
be phrased in terms of *emitting a span*, not *wrapping text in a tag*.

| Avoid                                   | Prefer                                                                    |
| --------------------------------------- | ------------------------------------------------------------------------- |
| "Wrap the author name in `<author>`."   | "Emit an `author` span covering the full name text."                      |
| "Nest `<surname>` inside `<author>`."   | "The `surname` span must fall within the enclosing `author` span's text." |

---

## Guidelines

### 1. Say "emit a span" explicitly

Models respond well to imperative phrasing that mirrors their task:

> "Emit a separate `author` span for each distinct author."

Vague descriptions like *"the author's name"* leave models guessing whether to
act at all.

---

### 2. State multiplicity and grouping rules explicitly

Without explicit instructions, models may make wrong choices about how many
spans to emit.

**Bad:** "Name(s) of the author(s) of the cited work."

**Good:** "All name parts (`surname`, `forename`, `orgName`) for one or more
contiguous authors may be placed inside a single `author` span. Emit a separate
`author` span only when authors are separated by non-name text (e.g. a title or
date between them)."

This applies to any element that can appear more than once: `author`, `editor`,
`biblScope` (volume vs. issue), `date` (publication year vs. original year), etc.

---

### 3. Describe parent–child span pairs from both sides

When two span types must always appear together (e.g. `author` wrapping
`orgName`), describe the relationship in **both** element descriptions and state
explicitly which span encloses the other.  Also provide a concrete example,
because models often invert the nesting direction from an ambiguous description.

**In `author`:**
> "When an organisation is the author, emit both an `author` span and an
> `orgName` span covering the same text — the `author` span encloses the
> `orgName` span.  For example, if 'Acme Research Group' is an author, emit an
> `author` span AND an `orgName` span both covering 'Acme Research Group'."

**In `orgName`:**
> "When the organisation acts as author or editor, you MUST emit both the
> `orgName` span and an enclosing `author` (or `editor`) span.  Never emit
> `orgName` alone in that role."

The same pattern applies to `surname`/`forename` inside `author`/`editor`.

---

### 4. Include negative constraints ("never … as …")

*What not to annotate* is as important as what to annotate.  Models will
confidently mis-label text unless told otherwise.

Examples of effective negative constraints:

> "A person's name (or surname alone) that follows 'in' is an editor — emit an
> `editor` span, **never** a `title` span."
>
> "An institutional report name (e.g. 'Amok Internal Report') must be tagged as
> `note` with type='report', **NOT** as `orgName` or `title`."
>
> "A label is always a number or short code — **never** a word or name.  An
> ALL-CAPS word at the start of an entry is an author surname, not a label."

Prefix the most critical constraints with **CRITICAL:** to draw model attention:

> "CRITICAL: A person's name (or surname alone) that follows 'in' is an editor
> — emit an `editor` span (plus name-part spans), never a `title` span."

---

### 5. Describe textual triggers and positional cues

Tell the model *what text signals the presence of the span*, not just what the
span represents semantically.

> "An editor's name typically follows keywords such as 'in', 'ed.', 'éd.',
> 'Hrsg.', 'dir.', '(ed.)', '(eds.)'."
>
> "A label appears at the very start of a bibliographic entry, before any author
> or title."
>
> "The place of publication may appear in parentheses immediately after the
> title, e.g. 'Title (City, Region)' — the parenthesised location is the
> pubPlace."

---

### 6. Use inline examples for ambiguous surface forms

Short quoted examples in the description remove ambiguity about what the span
text looks like:

> "Typical label forms: a plain number ('17'), a number with a trailing period
> ('17.'), a number in square brackets ('[77]', '[ACL30]'), or a compound number
> ('5,6')."
>
> "Institutional report designations — such as 'Amok Internal Report', 'USGS
> Open-File Report 97-123', or 'Technical Report No. 5' — must be tagged as
> `note`."

---

### 7. Define span boundaries explicitly

State what is *included* and what is *excluded* from the span text, especially
when surrounding punctuation could reasonably be included:

> "The separator that follows the label (period, dash, or space) is NOT part of
> the label."
>
> "Do not include the surrounding parentheses in the pubPlace span."

---

### 8. Use `TEISchema.rules` for cross-element constraints

When the same constraint applies to **multiple element types**, put it in
`TEISchema.rules` rather than copying it into every element description.
The prompt builder renders `rules` as a numbered **"General Rules"** section
that appears before all per-element descriptions.

Good candidates for `rules`:

- Parent–child pairing constraints shared by several elements (e.g. "`surname`
  and `forename` must always appear inside an enclosing `author` or `editor`
  span")
- Constraints that span the same surface form from both sides (e.g. the rule
  that `orgName` requires a sibling `author`/`editor` span, stated for both
  `author` and `orgName`)
- Bibliographic conventions that apply across multiple roles (e.g. "a dash or
  underscore may stand for a repeated author **or editor** name")

Keep the individual element `description` focused on element-specific cues
(triggers, surface forms, boundaries, negative constraints) and let `rules`
carry the shared structural invariants.

**Example** — in `_build_schema()`:

```python
TEISchema(
    rules=[
        "For each person's name, emit an 'author' or 'editor' span covering "
        "the full name AND separate 'surname', 'forename', or 'orgName' spans "
        "for the individual name parts within that span.",
        "Never emit 'surname', 'forename', or 'orgName' without a corresponding "
        "enclosing 'author' or 'editor' span.",
    ],
    elements=[
        TEIElement(tag="author", description="Names appearing at the start …"),
        TEIElement(tag="surname", description="The inherited (family) name …"),
        # 'surname' description no longer repeats the parent-span constraint
    ],
)
```

---

## Quick checklist

Before finalising a description, ask:

- [ ] Is it phrased as "emit a span" rather than "use a tag"?
- [ ] Does it say how many spans to emit when the element can repeat?
- [ ] If the span must enclose another span type, is the direction stated with an example?
- [ ] Are the most common mis-labellings ruled out with a "never … as …" clause?
- [ ] Are there positional or keyword triggers that help the model find the span?
- [ ] Are edge-case surface forms illustrated with a quoted example?
- [ ] Are span boundaries (what's in / what's out) unambiguous?
- [ ] Are cross-element constraints factored into `TEISchema.rules` rather than duplicated across descriptions?
