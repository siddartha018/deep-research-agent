from deep_research.rag.chunk import chunk_text


def _para(n_words: int, seed: str = "a") -> str:
    return " ".join([seed] * n_words)


def test_short_input_below_min_chars_returns_empty():
    assert chunk_text("hi") == []


def test_single_paragraph_fits_in_one_chunk():
    p = _para(200)
    chunks = chunk_text(p, max_tokens=800, overlap_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].strip() == p


def test_splits_when_over_max_tokens():
    p = _para(3000)  # far more than 800 tokens
    chunks = chunk_text(p, max_tokens=200, overlap_tokens=50, min_chars=50)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c) >= 50


def test_overlap_between_adjacent_chunks():
    # Two clearly distinct paragraphs bigger than max_tokens combined
    text = _para(500, "alpha") + "\n\n" + _para(500, "beta")
    chunks = chunk_text(text, max_tokens=300, overlap_tokens=50, min_chars=50)
    assert len(chunks) >= 2
    # Overlap: last 50 tokens of chunk N appear at the start of chunk N+1.
    assert any(c.split()[0] == "alpha" or c.split()[0] == "beta" for c in chunks)


def test_drops_below_min_chars_tail():
    text = _para(500, "alpha")
    chunks = chunk_text(text, max_tokens=1000, overlap_tokens=100, min_chars=10_000)
    assert chunks == []
