from qdrant_client import QdrantClient

from deep_research.rag.qdrant import (
    create_run_collection, upsert_chunks, DENSE_DIM,
)
from deep_research.state import Chunk
from deep_research.tools.retrieve import retrieve


def _v(i: int) -> list[float]:
    v = [0.0] * DENSE_DIM
    v[i % DENSE_DIM] = 1.0
    return v


def test_retrieve_returns_relevant_first():
    client = QdrantClient(":memory:")
    create_run_collection("t", client=client)
    chunks = [
        Chunk(id="c0", url="u", title="t", text="apples are red fruit",
              chunk_idx=0, sub_q_id="q1", fetched_at="d"),
        Chunk(id="c1", url="u", title="t", text="capital of france is paris",
              chunk_idx=1, sub_q_id="q1", fetched_at="d"),
        Chunk(id="c2", url="u", title="t", text="oranges are orange fruit",
              chunk_idx=2, sub_q_id="q1", fetched_at="d"),
    ]
    vecs = [_v(0), _v(1), _v(2)]
    upsert_chunks("t", chunks, vecs, client=client)

    hits = retrieve("t", "what fruits are red", dense_vector=_v(0), k=3, client=client)
    ids = [h.id for h in hits]
    assert "c0" in ids
    assert hits[0].id == "c0"
