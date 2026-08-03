import pytest

from deep_research.tools.cite import cite


def test_cite_dedupes_and_preserves_order():
    c = cite(["c2", "c1", "c2", "c3"], claim="X caused Y")
    assert c.claim == "X caused Y"
    assert c.chunk_ids == ["c2", "c1", "c3"]


def test_cite_requires_at_least_one_chunk():
    with pytest.raises(ValueError):
        cite([], "claim")
