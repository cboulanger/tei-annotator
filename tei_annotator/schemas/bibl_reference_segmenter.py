"""
GROBID referenceSegmenter schema for TEI annotation.

This schema covers the elements used in the GROBID referenceSegmenter training
data (files named *.referenceSegmenter.tei.xml).  The task is different from
the blbl schema: the *input* is the plain text of a whole reference list
(a <listBibl>) and the *output* segments that text into individual <bibl>
spans, each optionally beginning with a <label> span.

Evaluation usage
----------------
    from tei_annotator.evaluation import evaluate_file
    from tei_annotator.schemas.bibl_reference_segmenter import (
        build_bibl_reference_segmenter_schema,
    )

    schema = build_bibl_reference_segmenter_schema()
    per_record, agg = evaluate_file(
        gold_xml_path="data/grobid-batch-1/tei/...",
        schema=schema,
        endpoint=endpoint,
        root_element="text",
        child_element="listBibl",
    )
"""

from __future__ import annotations


def build_bibl_reference_segmenter_schema():
    from tei_annotator.models.schema import TEIAttribute, TEIElement, TEISchema

    def attr(name: str, desc: str, allowed: list[str] | None = None) -> TEIAttribute:
        return TEIAttribute(name=name, description=desc, allowed_values=allowed)

    return TEISchema(
        rules=[
            "Mark each distinct bibliographic reference as a 'bibl' span.  A new reference "
            "typically begins with an author's last name (often in ALL-CAPS or inverted "
            "'SURNAME, First' form) or with an introductory phrase such as 'Cf.', 'See', "
            "'Ver também:', 'Nesse sentido:', 'Ibidem', 'op. cit.'.",
            "CRITICAL: A footnote or endnote that cites multiple separate works — typically "
            "separated by a semicolon followed by a new author name, or by a period followed "
            "by a new author name in inverted/capitalised form — produces MULTIPLE 'bibl' "
            "spans, one per cited work.  Only the FIRST 'bibl' in the footnote carries the "
            "'label'; the remaining 'bibl' spans for the same footnote have no label.  "
            "Step-by-step example — '1. Robins (2013); Boss (2000); Kovras (2017).' → "
            "bibl span 1: text = '1. Robins (2013);', with a nested 'label' span text = '1'; "
            "bibl span 2: text = 'Boss (2000);' (no label); "
            "bibl span 3: text = 'Kovras (2017).' (no label).  "
            "ALL THREE cited works need their own 'bibl' span.  After wrapping the first bibl, "
            "continue wrapping every remaining citation into its own bibl.  "
            "Do NOT stop after 1 or 2 — wrap every cited work until the end of the footnote.  "
            "EXCEPTION: a semicolon that appears *within* an author list (e.g. 'COIMBRA, "
            "Marcelo; Manzi, Vanessa') is NOT a reference separator — it separates "
            "co-authors of the same work.",
            "CRITICAL: When a footnote entry begins with a label, ALL text in that entry — "
            "from the label to the end of the last citation — must be divided into one or more "
            "'bibl' spans.  No text between the opening label and the end of the footnote entry "
            "may be left as bare unwrapped text.  If the text immediately following the label "
            "is commentary rather than a formal citation, wrap it in a 'bibl' span anyway.  "
            "Step-by-step example — "
            "'57 Article 28 of the ICESCR expressly provides that \"...\". See also HUMAN RIGHTS COMMITTEE, GENERAL COMMENT 31: ..., [4] (2004).' → "
            "the leading '57' is a label; the following sentence is pure commentary (not a formal citation); "
            "the 'See also' clause introduces a citation but does NOT start a new bibl because the "
            "preceding commentary contains no complete formal citation.  "
            "Correct output: ONE 'bibl' span covering the entire entry from '57' to '(2004).', "
            "with a nested 'label' span text = '57'.  "
            "WRONG: leaving '57 Article 28...' as bare text and emitting only 'See also HUMAN RIGHTS COMMITTEE...' as a bibl.",
            "If a reference begins with a numeric or alphanumeric label (footnote number, "
            "endnote number, or reference key), emit a 'label' span covering that label — "
            "including any brackets, parentheses, or trailing period that are part of the "
            "label format — as the very first span inside the enclosing 'bibl' span.  "
            "The whitespace or dash that separates the label from the first author is NOT "
            "part of the label span.  "
            "IMPORTANT: A plain integer at the very start of a footnote or endnote entry is "
            "ALWAYS a label, even when what follows is substantive prose rather than an "
            "immediate author name or formal citation.  The number still identifies the "
            "entry and must be tagged as a 'label' inside the enclosing 'bibl'.",
            "Labels take many forms: plain integers ('1', '42'), integers with a trailing "
            "period ('1.', '42.'), integers in square brackets ('[1]', '[42]'), integers in "
            "parentheses ('(1)', '(42)'), letter-number codes ('5a'), or special characters "
            "such as '*'.  ALL of these forms are valid labels and must be tagged.  "
            "CRITICAL: The label span text MUST include ALL formatting characters — the "
            "trailing period, enclosing brackets, and enclosing parentheses belong INSIDE "
            "the span text.  Examples: '17.' → span text '17.' (NOT '17'); "
            "'[1]' → span text '[1]' (NOT '1'); '(1)' → span text '(1)' (NOT '1').",
            "CRITICAL — ORPHAN LABELS ARE FORBIDDEN: A label token MUST ALWAYS be enclosed "
            "inside a 'bibl' span — NEVER as bare text or as a 'label' span that appears "
            "OUTSIDE a 'bibl' span.  The 'bibl' span must OPEN BEFORE the label "
            "character(s) and CLOSE AFTER the last word of that entry.  This rule applies "
            "to ALL label forms without exception: integers, period-integers, bracket "
            "labels '[N]', parenthesis labels '(N)', and special characters '*'.  "
            "When a labeled entry begins with an introductory phrase such as 'See also', "
            "'Cf.', or 'Ver também:', that phrase is PART OF that labeled entry's bibl — "
            "it does NOT start a separate bibl.  The label defines where the bibl begins.  "
            "Anti-patterns that are ALWAYS WRONG:  "
            "(a) A 'label' span text='*' emitted with NO enclosing 'bibl' — correct is a "
            "'bibl' span '* R. Diana, ...' with label='*' as its first nested span.  "
            "(b) '[3]' left as bare text before '<bibl>See also C Coglianese...</bibl>' — "
            "correct is ONE 'bibl' span '[3] See also C Coglianese...' with label='[3]' "
            "as its first nested span.  "
            "(c) Emitting ONLY label spans with no bibl spans at all — for example, "
            "'<label>[1]</label> B Morgan... <label>[2]</label> R Baldwin... "
            "<label>[3]</label> See also C Coglianese...' produces ZERO bibl spans, "
            "which is completely wrong.  Correct output: THREE bibl spans, each containing "
            "the bracket label as its first nested child: "
            "'<bibl><label>[1]</label> B Morgan...</bibl>' "
            "'<bibl><label>[2]</label> R Baldwin...</bibl>' "
            "'<bibl><label>[3]</label> See also C Coglianese...</bibl>'.",
            "A single cited work that spans multiple OCR line breaks is still ONE 'bibl' "
            "span.  Do NOT split a single citation at a line break.",
            "Include the trailing separator of each reference (the semicolon or period that "
            "terminates it) INSIDE that reference's 'bibl' span, not at the start of the "
            "next one.",
            "Introductory commentary that immediately precedes a reference and directs the "
            "reader to it — e.g. 'See', 'Cf.', 'Nesse sentido:', 'For a contrary view, "
            "see', 'siehe auch', 'ver também' — belongs INSIDE that reference's 'bibl' "
            "span.  When such commentary introduces two or more consecutive references, "
            "attach it to the immediately following reference.  "
            "When 'see also', 'cf.', or similar phrases appear in the MIDDLE of a "
            "multi-citation footnote (after one or more bibls containing a COMPLETE formal "
            "citation — i.e. author + title + publication — have already been emitted), "
            "they introduce a new 'bibl' span.  "
            "EXCEPTION: if the immediately preceding text is pure commentary that contains "
            "NO complete formal citation (e.g. a sentence mentioning an author in passing "
            "without a title or publisher), do NOT start a new bibl at 'see' or 'see also' — "
            "include that phrase and what follows in the SAME bibl that covers the commentary.  "
            "The bibl must begin at the very start of the commentary (or at the label if one "
            "precedes it), NOT at the 'see also' phrase.  "
            "Leaving the commentary as bare unwrapped text while making only the 'see also' "
            "clause into a bibl is always wrong.",
            "Standalone commentary that does not directly refer to any specific reference, "
            "or that bridges two different references, should be included in the span that "
            "covers the FOLLOWING reference.",
            "Commentary that immediately follows a reference and elaborates on it — e.g. "
            "parenthetical remarks such as '(arguing that …)', brief paraphrases — belongs "
            "INSIDE that reference's 'bibl' span.",
            "Short self-contained cross-references such as 'Id.', 'Ibid.', 'Idem.', "
            "'Op. cit.', 'supra note N' each form their own individual 'bibl' span "
            "(with a 'label' if a label precedes them).",
            "Cover as much of the text as possible with 'bibl' spans.  Do not leave "
            "whitespace or punctuation gaps between spans.",
            "Do NOT nest 'bibl' spans inside other 'bibl' spans.",
            "Text that is NOT a bibliographic reference — section headings such as "
            "'References', 'Bibliography', 'Notes', or editorial annotations — must NOT "
            "be wrapped in a 'bibl' span.  Only actual reference entries get a 'bibl' span.",
            "Some reference lists use a purely alphabetical (author-date) format with no "
            "numeric labels.  In that case, every reference still gets a 'bibl' span, but "
            "no 'label' spans are emitted.",
            "Do NOT emit a 'label' span when the leading text is an author's surname "
            "(ALL-CAPS or mixed-case) rather than a numeric or alphanumeric code.",
            "CRITICAL: A number or character that is embedded inside a URL, a word, "
            "or any non-whitespace alphanumeric string is NEVER a label.  "
            "A label must be a completely standalone token at the very start of a "
            "reference entry — it must be preceded by whitespace or the very start "
            "of the input, and followed by whitespace before the reference text.  "
            "Example of what is NOT a label: the digit '7' inside the URL path "
            "'https://perma.cc/4SZ2-47A9' — even if a line break appears nearby, "
            "that '7' is part of the URL string and must never be tagged as a label.  "
            "Only tag '7' as a label when it appears as a free-standing token "
            "at the start of a footnote entry, e.g. '7 Feinstein & Wood, ...'.",
        ],
        elements=[
            TEIElement(
                tag="bibl",
                description=(
                    "MANDATORY: Every bibliographic entry in the reference list MUST be "
                    "wrapped in a 'bibl' span — this is the primary required output of the "
                    "task.  Emitting only 'label' spans with no 'bibl' spans at all is "
                    "always wrong, regardless of the label format (integer, period-integer, "
                    "bracket '[N]', or special character '*').  "
                    "A span covering one complete bibliographic reference, including any "
                    "commentary that directly qualifies or elaborates on that specific "
                    "reference.  Commentary that immediately precedes a reference (e.g. "
                    "'See also', 'For a different view see', 'Cf.', and similar expressions "
                    "in other languages such as 'siehe auch', 'ver também') and belongs to "
                    "it must be included in the span.  Commentary that immediately follows a "
                    "reference and is clearly about that reference (e.g. '(arguing that …)', "
                    "'who first demonstrated that …') must also be included.  A 'bibl' span "
                    "must contain at minimum one verifiable bibliographic item — an author "
                    "name, title, publication, or a short-form citation ('Ibid.', 'op. cit.', "
                    "a bare page number following a prior citation).  An in-text mention of an "
                    "author by name (e.g. 'resembles Louis Althusser's distinction') qualifies "
                    "as a bibliographic item even without a publication title or date — wrap "
                    "such commentary in a 'bibl' span, especially when it follows a label or "
                    "precedes a formal citation.  Sources without named authors, such as "
                    "websites (title + URL), are also valid bibliographic references.  "
                    "Standalone commentary that refers to no specific reference "
                    "or bridges two references should be included in the FOLLOWING reference's "
                    "span.  If the reference begins with a numeric or alphanumeric label "
                    "(including a special-character label such as '*'), the very first nested "
                    "span inside this 'bibl' span MUST be a 'label' span — never emit the "
                    "label text as bare untagged text.  "
                    "CRITICAL: A reference that starts with a special-character label such "
                    "as '*' still requires a full 'bibl' span.  "
                    "Example: '* R. Diana, Migrations of Concepts, Brepols, 2023.' → "
                    "bibl span = '* R. Diana, Migrations of Concepts, Brepols, 2023.' "
                    "containing label span = '*'.  Never emit only the label span without "
                    "an enclosing bibl span."
                ),
                allowed_children=["label"],
                attributes=[],
            ),
            TEIElement(
                tag="label",
                description=(
                    "A numeric or alphanumeric label at the very start of a reference that "
                    "identifies or numbers it.  Typical forms: a plain integer ('17'), an "
                    "integer with a trailing period ('17.'), an integer in square brackets "
                    "('[77]', '[ACL30]'), an integer in parentheses ('(3)'), a letter-number "
                    "code ('5a'), or a special character ('*').  "
                    "CRITICAL: A lone asterisk '*' at the very start of a reference IS a "
                    "valid label and MUST be tagged.  "
                    "Example: '* R. Diana, Migrations of Concepts...' → emit a 'label' span "
                    "with text='*' as the first span inside the enclosing 'bibl' span.  "
                    "The separator that follows the label (period, dash, space, closing "
                    "bracket) is NOT part of the label.  "
                    "A label is always a standalone number or short code at the very "
                    "beginning of a reference — never a word, name, or sentence fragment, "
                    "and never a number embedded inside a URL or alphanumeric string "
                    "(see general rules for the URL constraint).  "
                    "A plain integer at the very start of a footnote entry is ALWAYS a label "
                    "even when what follows is substantive prose rather than an immediate "
                    "author name or citation — e.g. '57 Article 28 of the ICESCR...' → "
                    "label = '57'.  "
                    "CRITICAL: A 'label' span MUST ALWAYS appear as the first nested span "
                    "inside a 'bibl' span.  Emitting a label as bare text outside a 'bibl' "
                    "span is always wrong.  If you are unsure how to divide the content "
                    "following the label, wrap the label AND all remaining text of that "
                    "footnote entry in a single 'bibl' span."
                ),
                allowed_children=[],
                attributes=[],
            ),
        ],
    )
