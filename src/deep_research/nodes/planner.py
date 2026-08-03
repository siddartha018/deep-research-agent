from __future__ import annotations

import json

from ..llm import get_llm
from ..obs import observed
from ..state import Plan, ResearchState, SubQuestion


PLAN_JSON_SCHEMA = """{
  "sub_questions": [
    {"id": "q0", "question": "string", "search_terms": ["string", ...]}
  ]
}"""

_SYSTEM = (
    "You are a research planner. Decompose the user's question into 3–7 focused "
    "sub-questions that together fully answer it. For each sub-question, give a "
    "short id (q0, q1, ...) and 2–4 search terms. Reply ONLY with JSON matching "
    f"this schema:\n{PLAN_JSON_SCHEMA}"
)


def _parse(raw: str) -> Plan:
    data = json.loads(raw)
    sqs = [
        SubQuestion(id=s["id"], question=s["question"], search_terms=list(s.get("search_terms", [])))
        for s in data["sub_questions"]
    ]
    if not sqs:
        raise ValueError("planner produced empty plan")
    return Plan(sub_questions=sqs)


@observed(name="planner", as_type="agent")
def plan_query(state: ResearchState, *, llm=None) -> dict:
    llm = llm or get_llm()
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": state["query"]},
    ]
    for attempt in range(2):
        result = llm.chat(messages, model="primary", temperature=0.0)
        try:
            plan = _parse(result.content)
            return {"plan": plan}
        except (json.JSONDecodeError, KeyError, ValueError):
            messages.append({"role": "user", "content": "Reply ONLY with JSON matching the schema."})
    raise ValueError("planner failed to produce valid JSON after 2 attempts")
