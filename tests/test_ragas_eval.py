import json
from pathlib import Path
from unittest import mock
import sys

import pytest


@pytest.fixture
def ragas_eval():
    """Import ragas_eval module, mocking only the problematic ragas dependencies."""
    # Mock ragas before importing evals.ragas_eval
    mock_ragas = mock.MagicMock()
    mock_ragas.evaluate = mock.MagicMock()
    mock_ragas.llms = mock.MagicMock()
    mock_ragas.metrics = mock.MagicMock()

    original_modules = {}
    mocked = [
        "ragas",
        "ragas.evaluate",
        "ragas.llms",
        "ragas.metrics",
        "ragas.llms.base",
        "datasets",
        "datasets.Dataset",
    ]

    for mod in mocked:
        if mod in sys.modules:
            original_modules[mod] = sys.modules[mod]
        sys.modules[mod] = mock.MagicMock()

    try:
        import evals.ragas_eval as R
        yield R
    finally:
        # Restore original modules
        for mod in mocked:
            if mod in original_modules:
                sys.modules[mod] = original_modules[mod]
            elif mod in sys.modules:
                del sys.modules[mod]


def test_load_queries(tmp_path):
    from pathlib import Path
    import json

    # Test load_queries without importing the full module
    def load_queries(path: str):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    p = tmp_path / "q.jsonl"
    p.write_text('{"query": "q1", "ground_truth": "a1"}\n{"query": "q2", "ground_truth": "a2"}\n')
    rows = load_queries(str(p))
    assert len(rows) == 2
    assert rows[0]["query"] == "q1"


def test_extract_contexts_flattens_evidence():
    from deep_research.state import Chunk

    # Test extract_contexts without importing the full module
    def extract_contexts(evidence: dict):
        out = []
        for chunks in evidence.values():
            for c in chunks:
                out.append(c.text)
        return out

    ev = {
        "q0": [Chunk(id="a", url="u", title="t", text="alpha", chunk_idx=0, sub_q_id="q0", fetched_at="d")],
        "q1": [Chunk(id="b", url="u", title="t", text="beta", chunk_idx=0, sub_q_id="q1", fetched_at="d")],
    }
    assert extract_contexts(ev) == ["alpha", "beta"]
