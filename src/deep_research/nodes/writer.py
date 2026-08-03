from __future__ import annotations

import re

from ..llm import get_llm
from ..obs import observed
from ..state import ResearchState


CITATION_RE = re.compile(r"\[(\d+)\]")
_SENT_RE = re.compile(r"[.!?]+\s+")

_SYSTEM = (
    "You are a research writer. Synthesize the sub-answers into a single, cohesive "
    "report. EVERY sentence must contain at least one citation using the format [n], "
    "where n refers to the citation index provided. Do not invent citations. "
    "Return plain markdown."
)


def citation_coverage(report: str) -> float:
    sents = [s.strip() for s in _SENT_RE.split(report) if s.strip()]
    if not sents:
        return 0.0
    cited = sum(1 for s in sents if CITATION_RE.search(s))
    return cited / len(sents)


def _build_index(state: ResearchState) -> tuple[str, list[str]]:
    idx_lines = []
    ordered_chunk_ids: list[str] = []
    n = 1
    for sq in state["plan"].sub_questions:  # type: ignore[union-attr]
        ans = state["answers"].get(sq.id)
        if not ans:
            continue
        for cit in ans.citations:
            for cid in cit.chunk_ids:
                if cid not in ordered_chunk_ids:
                    ordered_chunk_ids.append(cid)
                    idx_lines.append(f"[{n}] chunk_id={cid} — claim: {cit.claim}")
                    n += 1
    return "\n".join(idx_lines), ordered_chunk_ids


@observed(name="writer", as_type="agent")
def write_report(state: ResearchState, *, llm=None, min_coverage: float = 0.9) -> dict:
    llm = llm or get_llm()
    index_block, _ = _build_index(state)
    sub_answers = "\n\n".join(
        f"### {sq.id}: {sq.question}\n{state['answers'][sq.id].answer}"
        for sq in state["plan"].sub_questions  # type: ignore[union-attr]
        if sq.id in state["answers"]
    )
    user = (
        f"Question: {state['query']}\n\n"
        f"Citation index:\n{index_block}\n\n"
        f"Sub-answers:\n{sub_answers}"
    )
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]

    report = ""
    for attempt in range(3):
        resp = llm.chat(messages, model="primary", temperature=0.2)
        report = resp.content
        if citation_coverage(report) >= min_coverage:
            return {"report": report}
        messages.append({
            "role": "user",
            "content": "Not every sentence has a [n] citation. Rewrite so every sentence contains at least one [n] citation drawn from the index above.",
        })
    return {"report": report}
