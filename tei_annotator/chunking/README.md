# Chunking

Long texts exceed the context window (or the practical attention span) of most LLMs. The chunker splits the source text into overlapping windows so that each LLM call sees a manageable piece of text while entity boundaries between windows are never lost.

---

## API

```python
from tei_annotator.chunking.chunker import chunk_text, Chunk

chunks: list[Chunk] = chunk_text(text, chunk_size=1500, chunk_overlap=200)
```

Each `Chunk` is a dataclass:

```python
@dataclass
class Chunk:
    text: str          # content of this window
    start_offset: int  # character position of chunk[0] in the original text
```

`start_offset` is used by the resolver to convert chunk-local character positions back to positions in the full source text.

---

## Splitting algorithm

1. Walk the text left-to-right, accumulating characters.
2. When the accumulated length reaches `chunk_size`, scan backwards for the last whitespace character at or before that position and cut there. The boundary is always between words, never mid-token.
3. The **next chunk begins** `chunk_overlap` characters before the cut point, so consecutive chunks share a strip of text.
4. Repeat until all text is consumed.

The final chunk is whatever remains after the last cut, even if it is shorter than `chunk_size`.

---

## XML safety

If the source text contains existing XML markup, the splitter is tag-aware: it never places a cut inside an XML tag. Tags are treated as **zero-width** for length accounting — only visible text characters count toward `chunk_size`. This ensures that pre-existing markup is always preserved intact within whichever chunk it falls in.

---

## Why overlap matters

Without overlap, an entity that straddles the boundary between two chunks could be missed by both LLM calls — the first call sees the entity's opening but not its close, and vice versa. With `chunk_overlap` characters of shared context, the entity appears complete in at least one chunk with enough surrounding text for the model to recognise and annotate it.

Because the overlap causes the same entity to appear in two consecutive chunks, the pipeline deduplicates resolved spans after collecting results from all chunks: identical `(element, start, end)` triples are merged and only one instance is kept.

---

## Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `chunk_size` | `1500` | Larger values send more text per LLM call (cheaper) but risk exceeding context limits. Reduce for models with small windows. |
| `chunk_overlap` | `200` | Larger values make boundary entities safer but increase the number of duplicate detections that must be deduplicated. |

A `chunk_size` of 1500 characters corresponds to roughly 375–500 tokens for Latin-script text, leaving ample room for the prompt preamble and JSON response within a 4 k-token limit.
