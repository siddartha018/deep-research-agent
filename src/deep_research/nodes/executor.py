from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from ..llm import get_llm
from ..obs import observed
from ..rag.chunk import chunk_text
from ..rag.qdrant import create_run_collection, upsert_chunks
from ..state import Chunk, Citation, ResearchState, SubAnswer, SubQuestion
from ..tools.fetch import fetch as default_fetch
from ..tools.retrieve import retrieve as default_retrieve
from ..tools.search import web_search as default_search


_SYSTEM = (
    "You are a research executor. Answer the sub-question using ONLY the provided "
    "chunks. Every claim must cite chunk ids. Reply with strict JSON: "
    '{"answer": "string", "citations": [{"claim": "string", "chunk_ids": ["c0", ...]}]}'
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@observed(name="executor", as_type="agent")
def run_executor(
    state: ResearchState,
    sub_q: SubQuestion,
    *,
    llm=None,
    search_fn=None,
    fetch_fn=None,
    retrieve_fn=None,
    embed_fn=None,
    upsert_fn=None,
    create_fn=None,
    now_fn=None,
) -> tuple[list[Chunk], SubAnswer]:
    llm = llm or get_llm()
    search_fn = search_fn or default_search
    fetch_fn = fetch_fn or default_fetch
    retrieve_fn = retrieve_fn or default_retrieve
    embed_fn = embed_fn or llm.embed
    upsert_fn = upsert_fn or upsert_chunks
    create_fn = create_fn or create_run_collection
    now_fn = now_fn or _now_iso

    hits = search_fn(sub_q.question, k=5)[:3]
    now = now_fn()

    all_chunks: list[Chunk] = []
    for h in hits:
        try:
            body = fetch_fn(h.url)
        except Exception:
            continue
        for idx, text in enumerate(chunk_text(body)):
            all_chunks.append(Chunk(
                id=f"{sub_q.id}-{uuid.uuid4().hex[:8]}",
                url=h.url, title=h.title, text=text,
                chunk_idx=idx, sub_q_id=sub_q.id, fetched_at=now,
            ))

    create_fn(state["trace_id"])
    if all_chunks:
        vecs = embed_fn([c.text for c in all_chunks])
        upsert_fn(state["trace_id"], all_chunks, vecs)
        qvec = embed_fn([sub_q.question])[0]
        retrieved = retrieve_fn(state["trace_id"], sub_q.question, dense_vector=qvec, k=5)
    else:
        retrieved = []

    context = "\n\n".join(f"[{c.id}] {c.text}" for c in retrieved)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Sub-question: {sub_q.question}\n\nChunks:\n{context}"},
    ]
    resp = llm.chat(messages, model="primary", temperature=0.1)
    try:
        parsed = json.loads(resp.content)
        citations = [
            Citation(claim=c["claim"], chunk_ids=list(c["chunk_ids"]))
            for c in parsed.get("citations", [])
        ]
        answer = parsed["answer"]
    except (json.JSONDecodeError, KeyError):
        answer, citations = resp.content, []

    return all_chunks, SubAnswer(sub_q_id=sub_q.id, answer=answer, citations=citations)
