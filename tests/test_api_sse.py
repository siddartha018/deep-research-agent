from fastapi.testclient import TestClient

from deep_research.api import build_app
from deep_research.state import (
    Chunk, Citation, Plan, SubAnswer, SubQuestion,
)


class FakeGraph:
    def stream(self, state, stream_mode="values"):
        yield {"plan": Plan(sub_questions=[SubQuestion(id="q0", question="?", search_terms=[])])}
        yield {"evidence": {"q0": [Chunk(id="c0", url="u", title="t", text="x",
                                         chunk_idx=0, sub_q_id="q0", fetched_at="d")]}}
        yield {"answers": {"q0": SubAnswer(sub_q_id="q0", answer="a", citations=[Citation("a", ["c0"])])}}
        yield {"report": "final [1]."}


def test_sse_stream_emits_all_stages(tmp_path):
    app = build_app(graph=FakeGraph(), db_path=str(tmp_path / "d.sqlite"))
    with TestClient(app) as client:
        with client.stream("POST", "/research", json={"query": "?"}) as r:
            body = "".join(r.iter_text())
    assert "event: plan" in body
    assert "event: evidence" in body
    assert "event: subanswer" in body
    assert "event: done" in body
