from __future__ import annotations

from qdrant_client import QdrantClient, models

from ..obs import observed
from ..rag.qdrant import (
    DENSE_NAME, SPARSE_NAME, _bm25, collection_name, get_client,
)
from ..state import Chunk


@observed(name="tool.retrieve", as_type="retriever")
def retrieve(
    trace_id: str,
    query: str,
    *,
    dense_vector: list[float],
    k: int = 8,
    client: QdrantClient | None = None,
) -> list[Chunk]:
    """
    Hybrid retrieval with dense + sparse RRF fusion.

    Args:
        trace_id: Unique trace/run identifier for the collection
        query: Text query to retrieve relevant chunks for
        dense_vector: Dense embedding vector for the query
        k: Number of results to return (default 8)
        client: Optional QdrantClient; uses default if None

    Returns:
        List of Chunk objects sorted by relevance via RRF fusion
    """
    if not query.strip():
        return []

    c = client or get_client()
    name = collection_name(trace_id)

    # Compute sparse embedding via BM25
    sparse = next(iter(_bm25().embed([query])))

    # Hybrid query with RRF fusion
    result = c.query_points(
        collection_name=name,
        prefetch=[
            models.Prefetch(query=dense_vector, using=DENSE_NAME, limit=k * 3),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
                using=SPARSE_NAME,
                limit=k * 3,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )

    out: list[Chunk] = []
    for p in result.points:
        pl = p.payload or {}
        # Reconstruct chunk_id from payload if present, else use point id
        chunk_id = pl.get("chunk_id", str(p.id))
        out.append(Chunk(
            id=chunk_id,
            url=pl.get("url", ""),
            title=pl.get("title", ""),
            text=pl.get("text", ""),
            chunk_idx=int(pl.get("chunk_idx", 0)),
            sub_q_id=pl.get("sub_q_id", ""),
            fetched_at=pl.get("fetched_at", ""),
        ))
    return out
