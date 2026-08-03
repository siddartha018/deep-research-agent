import json
import pytest

from deep_research.llm import ChatResult
from deep_research.nodes.planner import plan_query
from deep_research.state import new_state


class FakeLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        r = self._replies.pop(0)
        return ChatResult(content=r, tool_calls=[], model="test")


def _plan_json(qs):
    return json.dumps({
        "sub_questions": [
            {"id": f"q{i}", "question": q, "search_terms": [q.split()[0]]}
            for i, q in enumerate(qs)
        ]
    })


def test_plan_query_parses_and_returns_plan():
    llm = FakeLLM([_plan_json(["what is X", "why does X exist"])])
    s = new_state("what is X?", trace_id="t")
    update = plan_query(s, llm=llm)
    plan = update["plan"]
    assert len(plan.sub_questions) == 2
    assert plan.sub_questions[0].id == "q0"


def test_plan_query_retries_on_bad_json_then_succeeds():
    llm = FakeLLM(["not-json", _plan_json(["one"])])
    s = new_state("q?", trace_id="t")
    update = plan_query(s, llm=llm)
    assert len(update["plan"].sub_questions) == 1
    assert len(llm.calls) == 2


def test_plan_query_raises_after_two_bad_replies():
    llm = FakeLLM(["not-json", "also-not-json"])
    s = new_state("q?", trace_id="t")
    with pytest.raises(ValueError):
        plan_query(s, llm=llm)
