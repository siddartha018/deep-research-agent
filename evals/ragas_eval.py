from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from datasets import Dataset
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy, ContextPrecision, Faithfulness,
)

from deep_research import obs
from deep_research.config import get_settings
from deep_research.graph import build_graph
from deep_research.state import new_state


def load_queries(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def extract_contexts(evidence: dict) -> list[str]:
    out = []
    for chunks in evidence.values():
        for c in chunks:
            out.append(c.text)
    return out


def _run_one(query: str) -> tuple[str, list[str]]:
    graph = build_graph()
    trace_id = uuid.uuid4().hex
    state = graph.invoke(new_state(query, trace_id=trace_id))
    return state["report"] or "", extract_contexts(state["evidence"])


def main() -> None:
    obs.init_langfuse()
    obs.update_trace(name="ragas-eval", tags=["ragas-eval"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_queries(args.queries)
    records = []
    for r in rows:
        answer, contexts = _run_one(r["query"])
        records.append({
            "question": r["query"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": r["ground_truth"],
        })

    ds = Dataset.from_list(records)
    s = get_settings()
    llm = LangchainLLMWrapper(ChatGroq(
        api_key=s.groq_api_key, model="llama-3.3-70b-versatile", temperature=0.0,
    ))
    metrics = [
        Faithfulness(llm=llm),
        ContextPrecision(llm=llm),
        AnswerRelevancy(llm=llm),
    ]
    result = evaluate(dataset=ds, metrics=metrics)

    # Score each metric to Langfuse
    result_df = result.to_pandas()
    for metric_name in ["faithfulness", "context_precision", "answer_relevancy"]:
        if metric_name in result_df.columns:
            # Compute mean score for the metric
            scores = result_df[metric_name].dropna()
            if len(scores) > 0:
                mean_score = float(scores.mean())
                obs.score_current(name=metric_name, value=mean_score)

    Path(args.out).write_text(json.dumps({
        "generated_at": time.time(),
        "n_queries": len(rows),
        "scores": result_df.to_dict(orient="records"),
    }, indent=2))

    obs.flush()


if __name__ == "__main__":
    main()
