from __future__ import annotations

import json

from ..config import get_settings
from ..llm import get_llm
from ..obs import observed
from ..state import ResearchState


_SYSTEM = (
    "You are a research critic. For each sub-question, verdict its sub-answer as one of "
    "'covered', 'partial', or 'gap'. Reply strict JSON: "
    '{"verdicts": [{"id": "qN", "verdict": "covered|partial|gap"}]}'
)


@observed(name="critic", as_type="agent")
def critique(state: ResearchState, *, llm=None) -> dict:
    max_iter = get_settings().deep_research_max_iterations
    if state["iterations"] >= max_iter:
        return {"gaps": []}

    llm = llm or get_llm()
    plan = state["plan"]
    assert plan is not None
    payload = {
        "sub_answers": [
            {"id": sq.id, "question": sq.question,
             "answer": state["answers"].get(sq.id).answer if state["answers"].get(sq.id) else ""}
            for sq in plan.sub_questions
        ]
    }
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": json.dumps(payload)},
    ]
    result = llm.chat(messages, model="primary", temperature=0.0)
    try:
        data = json.loads(result.content)
        gaps = [v["id"] for v in data["verdicts"] if v["verdict"] != "covered"]
    except (json.JSONDecodeError, KeyError):
        gaps = []
    return {"gaps": gaps, "iterations": state["iterations"] + 1}
