from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    start_offset: int  # position of text[0] in the original source


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[Chunk]:
    """
    Split text into overlapping chunks, never splitting inside an XML tag.

    Each chunk's start_offset satisfies:
        original_text[chunk.start_offset : chunk.start_offset + len(chunk.text)] == chunk.text
    """
    if len(text) <= chunk_size:
        return [Chunk(text=text, start_offset=0)]

    # Build a set of character positions that are inside XML tags (inclusive).
    tag_positions: set[int] = set()
    for m in re.finditer(r"<[^>]*>", text):
        tag_positions.update(range(m.start(), m.end()))

    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Step back out of any XML tag
            candidate = end
            while candidate > start and candidate in tag_positions:
                candidate -= 1

            # Try to break at a whitespace boundary near the target
            break_pos = candidate
            for i in range(candidate, max(start, candidate - 100), -1):
                if i not in tag_positions and text[i].isspace():
                    break_pos = i + 1
                    break

            end = max(start + 1, break_pos)  # guarantee forward progress

        chunks.append(Chunk(text=text[start:end], start_offset=start))

        if end >= len(text):
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks
