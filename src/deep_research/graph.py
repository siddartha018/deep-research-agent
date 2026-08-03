from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import obs
from .nodes.critic import critique as default_critic
from .nodes.executor import run_executor as default_executor
from .nodes.planner import plan_query as default_planner
from .nodes.writer import write_report as default_writer
from .rag.qdrant import delete_run_collection
from .state import ResearchState, new_state


def build_graph(*, planner_fn=None, executor_fn=None, critic_fn=None, writer_fn=None):
    """Build a LangGraph state machine for deep research.

    Args:
        planner_fn: Function that takes (state, **kw) and returns {"plan": Plan}
        executor_fn: Function that takes (state, sub_q, **kw) and returns (list[Chunk], SubAnswer)
        critic_fn: Function that takes (state, **kw) and returns {"gaps": list[str], "iterations": int}
        writer_fn: Function that takes (state, **kw) and returns {"report": str}

    Returns:
        CompiledGraph: A compiled LangGraph state machine ready to invoke.
    """
    planner_fn = planner_fn or default_planner
    executor_fn = executor_fn or default_executor
    critic_fn = critic_fn or default_critic
    writer_fn = writer_fn or default_writer

    def planner_node(state: ResearchState) -> dict:
        return planner_fn(state)

    def dispatcher_node(state: ResearchState) -> dict:
        """Dispatcher fans out executor invocations to all open or gap sub-questions."""
        plan = state["plan"]
        assert plan is not None
        targets = state["gaps"] or [sq.id for sq in plan.sub_questions]
        evidence = dict(state["evidence"])
        answers = dict(state["answers"])
        for sq in plan.sub_questions:
            if sq.id not in targets:
                continue
            chunks, ans = executor_fn(state, sq)
            evidence[sq.id] = chunks
            answers[sq.id] = ans
        return {"evidence": evidence, "answers": answers, "gaps": []}

    def critic_node(state: ResearchState) -> dict:
        return critic_fn(state)

    def writer_node(state: ResearchState) -> dict:
        return writer_fn(state)

    def route_after_critic(state: ResearchState) -> str:
        """Route to dispatcher if gaps exist, otherwise to writer."""
        return "dispatcher" if state["gaps"] else "writer"

    g = StateGraph(ResearchState)
    g.add_node("planner", planner_node)
    g.add_node("dispatcher", dispatcher_node)
    g.add_node("critic", critic_node)
    g.add_node("writer", writer_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "dispatcher")
    g.add_edge("dispatcher", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"dispatcher": "dispatcher", "writer": "writer"})
    g.add_edge("writer", END)
    return g.compile()


def run(query: str, trace_id: str) -> ResearchState:
    """Synchronous convenience wrapper for running the graph.

    After invoke completes, updates the trace with session_id and tags,
    then flushes Langfuse (no-op if disabled).

    Args:
        query: The research question to investigate.
        trace_id: Unique trace identifier for Langfuse session grouping.

    Returns:
        ResearchState: Final state after graph execution.
    """
    obs.init_langfuse()
    try:
        with obs.trace_context(session_id=trace_id, tags=["deep-research"]):
            result = build_graph().invoke(new_state(query, trace_id))
        obs.flush()
        return result
    finally:
        try:
            delete_run_collection(trace_id)
        except Exception:
            pass
