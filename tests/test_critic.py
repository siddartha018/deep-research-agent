import json

from deep_research.llm import ChatResult
from deep_research.nodes.critic import critique
from deep_research.state import (
    Plan, SubAnswer, SubQuestion, new_state,
)


class FakeLLM:
    def __init__(self, reply):
        self._reply = reply
    def chat(self, messages, **kw):
        return ChatResult(content=self._reply, tool_calls=[], model="t")


def _base_state(iters=0):
    s = new_state("q?", trace_id="t")
    s["plan"] = Plan(sub_questions=[
        SubQuestion(id="q0", question="?", search_terms=[]),
        SubQuestion(id="q1", question="?", search_terms=[]),
    ])
    s["answers"] = {
        "q0": SubAnswer(sub_q_id="q0", answer="ok", citations=[]),
        "q1": SubAnswer(sub_q_id="q1", answer="ok", citations=[]),
    }
    s["iterations"] = iters
    return s


def test_critic_returns_gaps():
    reply = json.dumps({"verdicts": [
        {"id": "q0", "verdict": "covered"},
        {"id": "q1", "verdict": "gap"},
    ]})
    out = critique(_base_state(), llm=FakeLLM(reply))
    assert out["gaps"] == ["q1"]
    assert out["iterations"] == 1


def test_critic_short_circuits_at_max_iterations(monkeypatch):
    from deep_research import config
    monkeypatch.setattr(config.get_settings(), "deep_research_max_iterations", 1)
    reply = json.dumps({"verdicts": [{"id": "q0", "verdict": "gap"}, {"id": "q1", "verdict": "gap"}]})
    out = critique(_base_state(iters=1), llm=FakeLLM(reply))
    assert out["gaps"] == []
