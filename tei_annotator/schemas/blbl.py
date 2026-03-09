"""
BLBL bibliographic schema for TEI annotation.

This schema covers the elements that appear in blbl-examples.tei.xml and is
shared between the evaluation script and the webservice.
"""

from __future__ import annotations


def build_blbl_schema():
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
