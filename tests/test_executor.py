import json

from deep_research.llm import ChatResult
from deep_research.nodes.executor import run_executor
from deep_research.state import Chunk, SubQuestion, new_state
from deep_research.tools.search import SearchHit


class FakeLLM:
    def __init__(self, chat_reply):
        self._reply = chat_reply
        self.chat_calls = 0
        self.embed_calls = 0

    def chat(self, messages, **kw):
        self.chat_calls += 1
        return ChatResult(content=self._reply, tool_calls=[], model="test")

    def embed(self, texts):
        self.embed_calls += 1
        return [[0.1] * 768 for _ in texts]


def test_executor_full_pipeline_records_chunks_and_answer():
    fetched = {
        "http://a": "Apples are red. " * 40,
        "http://b": "Oranges are orange. " * 40,
    }
    search_calls = []
    fetch_calls = []
    upsert_calls = []
    retrieve_calls = []

    def search_fn(q, k=5):
        search_calls.append((q, k))
        return [SearchHit("http://a", "A", "a"), SearchHit("http://b", "B", "b")]

    def fetch_fn(url):
        fetch_calls.append(url)
        return fetched[url]

    def upsert_fn(trace_id, chunks, embs, client=None):
        upsert_calls.append((trace_id, len(chunks)))

    def retrieve_fn(trace_id, q, *, dense_vector, k=5, client=None):
        retrieve_calls.append((trace_id, q, k))
        return [Chunk(id="cA0", url="http://a", title="A", text="Apples are red",
                      chunk_idx=0, sub_q_id="q0", fetched_at="d")]

    reply = json.dumps({
        "answer": "Apples are red.",
        "citations": [{"claim": "Apples are red.", "chunk_ids": ["cA0"]}],
    })
    llm = FakeLLM(reply)

    state = new_state("what fruits are red?", trace_id="t")
    sq = SubQuestion(id="q0", question="what fruits are red?", search_terms=["apples"])

    chunks, ans = run_executor(
        state, sq, llm=llm,
        search_fn=search_fn, fetch_fn=fetch_fn,
        retrieve_fn=retrieve_fn, upsert_fn=upsert_fn,
        create_fn=lambda trace_id: None,
        now_fn=lambda: "2026-07-31",
    )

    assert search_calls == [("what fruits are red?", 5)]
    assert set(fetch_calls) == {"http://a", "http://b"}
    assert upsert_calls and upsert_calls[0][0] == "t"
    assert retrieve_calls and retrieve_calls[0][0] == "t"
    assert ans.sub_q_id == "q0"
    assert ans.answer == "Apples are red."
    assert ans.citations[0].chunk_ids == ["cA0"]
    assert len(chunks) > 0
