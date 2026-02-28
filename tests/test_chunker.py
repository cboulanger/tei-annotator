from tei_annotator.chunking.chunker import Chunk, chunk_text


def test_short_text_single_chunk():
    text = "Short text."
    chunks = chunk_text(text, chunk_size=1500)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start_offset == 0


def test_long_text_multiple_chunks():
    text = "word " * 400  # 2000 chars
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.start_offset >= 0
        if i > 0:
            assert chunk.start_offset > chunks[i - 1].start_offset


def test_chunk_start_offsets_correct():
    """Every chunk's text must match a slice of the original at start_offset."""
    text = "hello world " * 200
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    for chunk in chunks:
        assert (
            text[chunk.start_offset : chunk.start_offset + len(chunk.text)]
            == chunk.text
        )


def test_long_text_covers_all_characters():
    """Union of all chunk ranges must cover the entire source text."""
    text = "abcdefghij" * 200  # 2000 chars
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    covered: set[int] = set()
    for chunk in chunks:
        for j in range(chunk.start_offset, chunk.start_offset + len(chunk.text)):
            covered.add(j)
    assert covered == set(range(len(text)))


def test_chunk_boundary_does_not_split_xml_tag():
    """A chunk boundary must never fall inside an XML tag."""
    # Place a tag that straddles the natural 500-char boundary
    prefix = "a" * 495
    tag = "<someElement>"
    suffix = "b" * 600
    text = prefix + tag + suffix

    chunks = chunk_text(text, chunk_size=500, overlap=0)

    for chunk in chunks:
        # Each chunk must be self-consistent XML-tag-wise:
        # count of '<' must equal count of '>' within the chunk text
        # (a split tag would have an unbalanced '<' or '>')
        assert chunk.text.count("<") == chunk.text.count(">"), (
            f"Chunk at offset {chunk.start_offset} has unbalanced angle brackets: "
            f"{chunk.text!r}"
        )


def test_exact_chunk_size_no_overflow():
    text = "x" * 1500
    chunks = chunk_text(text, chunk_size=1500, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_overlap_produces_repeated_content():
    """With positive overlap, the end of chunk N overlaps with the start of chunk N+1."""
    text = "word " * 300  # 1500 chars
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) >= 2
    # The end of chunk 0 and the start of chunk 1 must share content
    c0_end = chunks[0].start_offset + len(chunks[0].text)
    c1_start = chunks[1].start_offset
    assert c1_start < c0_end, "Expected overlapping content between consecutive chunks"
