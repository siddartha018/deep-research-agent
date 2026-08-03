from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict


@dataclass
class SubQuestion:
    id: str
    question: str
    search_terms: list[str] = field(default_factory=list)
    status: Literal["open", "done"] = "open"


@dataclass
class Chunk:
    id: str
    url: str
    title: str
    text: str
    chunk_idx: int
    sub_q_id: str
    fetched_at: str


@dataclass
class Citation:
    claim: str
    chunk_ids: list[str]


@dataclass
class SubAnswer:
    sub_q_id: str
    answer: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Plan:
    sub_questions: list[SubQuestion]


class ResearchState(TypedDict):
    query: str
    plan: Plan | None
    evidence: dict[str, list[Chunk]]
    answers: dict[str, SubAnswer]
    gaps: list[str]
    iterations: int
    report: str | None
    trace_id: str
    cost: dict[str, int]


def new_state(query: str, trace_id: str) -> ResearchState:
    return ResearchState(
        query=query,
        plan=None,
        evidence={},
        answers={},
        gaps=[],
        iterations=0,
        report=None,
        trace_id=trace_id,
        cost={"tokens": 0, "requests": 0},
    )
