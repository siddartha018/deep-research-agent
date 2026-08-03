from deep_research.state import (
    Chunk, Citation, Plan, SubAnswer, SubQuestion, new_state,
)


def test_new_state_defaults():
    s = new_state("what is X?", trace_id="abc")
    assert s["query"] == "what is X?"
    assert s["trace_id"] == "abc"
    assert s["plan"] is None
    assert s["evidence"] == {}
    assert s["answers"] == {}
    assert s["gaps"] == []
    assert s["iterations"] == 0
    assert s["report"] is None
    assert s["cost"] == {"tokens": 0, "requests": 0}


def test_subquestion_status_defaults_open():
    q = SubQuestion(id="q1", question="?", search_terms=["a", "b"])
    assert q.status == "open"


def test_subanswer_carries_citations():
    c = Chunk(id="c1", url="u", title="t", text="x",
              chunk_idx=0, sub_q_id="q1", fetched_at="2026-07-31")
    a = SubAnswer(sub_q_id="q1", answer="ans", citations=[Citation("claim", [c.id])])
    assert a.citations[0].chunk_ids == ["c1"]


def test_plan_wraps_subquestions():
    p = Plan(sub_questions=[SubQuestion(id="q1", question="?", search_terms=[])])
    assert len(p.sub_questions) == 1
