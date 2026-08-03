import json

from deep_research.graph import build_graph
from deep_research.state import (
    Chunk, Citation, Plan, SubAnswer, SubQuestion, new_state,
)


def _fake_planner(state, **kw):
    return {"plan": Plan(sub_questions=[
        SubQuestion(id="q0", question="What is X?", search_terms=["x"]),
        SubQuestion(id="q1", question="Why does X exist?", search_terms=["x"]),
    ])}


def _fake_executor(state, sub_q, **kw):
    ch = [Chunk(id=f"{sub_q.id}-c0", url="http://x", title="t",
                text="X is a thing.", chunk_idx=0, sub_q_id=sub_q.id, fetched_at="d")]
    ans = SubAnswer(sub_q_id=sub_q.id, answer=f"Answer to {sub_q.id}.",
                    citations=[Citation(f"Answer to {sub_q.id}.", [ch[0].id])])
    return ch, ans


def _fake_critic(state, **kw):
    return {"gaps": [], "iterations": state["iterations"] + 1}


def _fake_writer(state, **kw):
    lines = [f"{ans.answer} [1]" for ans in state["answers"].values()]
    return {"report": ". ".join(lines) + "."}


def test_graph_smoke_end_to_end():
    g = build_graph(
        planner_fn=_fake_planner, executor_fn=_fake_executor,
        critic_fn=_fake_critic, writer_fn=_fake_writer,
    )
    state = new_state("what is X?", trace_id="t-smoke")
    out = g.invoke(state)
    assert out["report"] is not None
    assert "[1]" in out["report"]
    assert set(out["answers"].keys()) == {"q0", "q1"}
    assert len(out["evidence"]) == 2
