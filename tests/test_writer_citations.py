from deep_research.llm import ChatResult
from deep_research.nodes.writer import citation_coverage, write_report
from deep_research.state import (
    Citation, Plan, SubAnswer, SubQuestion, new_state,
)


class ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0
    def chat(self, messages, **kw):
        self.calls += 1
        return ChatResult(content=self._replies.pop(0), tool_calls=[], model="t")


def _prime(state):
    state["plan"] = Plan(sub_questions=[SubQuestion(id="q0", question="?", search_terms=[])])
    state["answers"] = {
        "q0": SubAnswer(
            sub_q_id="q0", answer="Apples are red.",
            citations=[Citation("Apples are red.", ["cA0"])],
        ),
    }
    return state


def test_citation_coverage_all_cited():
    txt = "Apples are red [1]. Oranges are orange [2]."
    assert citation_coverage(txt) == 1.0


def test_citation_coverage_partial():
    txt = "Apples are red [1]. Bananas are unclear."
    assert citation_coverage(txt) == 0.5


def test_writer_retries_when_coverage_low():
    s = _prime(new_state("q?", trace_id="t"))
    low = "Apples are red. Bananas are yellow."          # 0 cites
    high = "Apples are red [1]. Bananas are yellow [1]."  # both cited
    llm = ScriptedLLM([low, high])
    out = write_report(s, llm=llm, min_coverage=0.9)
    assert out["report"] == high
    assert llm.calls == 2


def test_writer_gives_up_after_two_retries():
    s = _prime(new_state("q?", trace_id="t"))
    bad = "No citations anywhere."
    llm = ScriptedLLM([bad, bad, bad])
    out = write_report(s, llm=llm, min_coverage=0.9)
    assert out["report"] == bad
    assert llm.calls == 3
