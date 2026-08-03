from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

import deep_research.obs
from deep_research.graph import build_graph
from deep_research.state import new_state


def main() -> None:
    deep_research.obs.init_langfuse()
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.queries).read_text().splitlines() if l.strip()]
    out_path = Path(args.out)
    per_row: list[dict] = []

    def _write():
        succeeded = [r for r in per_row if r.get("latency_sec") is not None]
        lats = [r["latency_sec"] for r in succeeded]
        result = {
            "generated_at": time.time(),
            "n_total": len(rows),
            "n_success": len(succeeded),
            "p50_sec": statistics.median(lats) if lats else None,
            "p95_sec": (
                statistics.quantiles(lats, n=20)[18]
                if len(lats) >= 20 else (max(lats) if lats else None)
            ),
            "per_row": per_row,
        }
        out_path.write_text(json.dumps(result, indent=2))

    for i, r in enumerate(rows, 1):
        t0 = time.monotonic()
        try:
            graph = build_graph()
            state = graph.invoke(new_state(r["query"], trace_id=uuid.uuid4().hex))
            per_row.append({
                "query": r["query"],
                "latency_sec": time.monotonic() - t0,
                "report_len": len(state["report"] or ""),
                "error": None,
            })
        except Exception as e:
            per_row.append({
                "query": r["query"],
                "latency_sec": None,
                "report_len": 0,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })
        _write()  # incremental checkpoint: resume-safe

    deep_research.obs.flush()


if __name__ == "__main__":
    main()
