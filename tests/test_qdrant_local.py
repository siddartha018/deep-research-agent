import pytest
from qdrant_client import QdrantClient

from deep_research.rag.qdrant import (
    collection_name, create_run_collection, delete_run_collection,
    upsert_chunks, DENSE_DIM,
)
from deep_research.state import Chunk


@pytest.fixture
def client():
    return QdrantClient(":memory:")


def test_collection_name_is_safe():
    n = collection_name("abc-123-def")
    assert n.startswith("run_")
    assert "-" not in n


def test_create_and_delete_collection(client):
    name = create_run_collection("trace1", client=client)
    assert client.collection_exists(name)
    delete_run_collection("trace1", client=client)
    assert not client.collection_exists(name)


def test_upsert_and_scroll(client):
    create_run_collection("trace2", client=client)
    chunks = [
        Chunk(id=f"c{i}", url="u", title="t", text=f"body {i}",
              chunk_idx=i, sub_q_id="q1", fetched_at="2026-07-31")
        for i in range(3)
    ]
    dense = [[0.1] * DENSE_DIM for _ in chunks]
    upsert_chunks("trace2", chunks, dense, client=client)
    name = collection_name("trace2")
    hits, _ = client.scroll(collection_name=name, limit=10, with_payload=True)
    assert len(hits) == 3
    assert {h.payload["chunk_idx"] for h in hits} == {0, 1, 2}
