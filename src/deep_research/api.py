from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .graph import build_graph
from .obs import init_langfuse, update_trace, flush
from .rag.qdrant import delete_run_collection
from .state import new_state
from .store import get_run, init_db, save_run


class ResearchRequest(BaseModel):
    query: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(v):
    """Recursively convert dataclass objects and other complex types to JSON-serializable dicts."""
    if hasattr(v, "__dataclass_fields__"):
        # It's a dataclass
        return {f: _to_jsonable(getattr(v, f)) for f in v.__dataclass_fields__}
    if isinstance(v, list):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    return v


def build_app(*, graph=None, db_path: str | None = None) -> FastAPI:
    app = FastAPI(title="deep-research-agent")
    the_graph = graph or build_graph()
    init_db(db_path)
    init_langfuse()

    @app.post("/research")
    async def research(req: ResearchRequest):
        trace_id = uuid.uuid4().hex
        started = _now()
        state = new_state(req.query, trace_id=trace_id)

        async def gen():
            try:
                from . import obs as obs_module
                with obs_module.trace_context(session_id=trace_id, tags=["deep-research", "api"]):
                    report = None
                    plan_json = "{}"
                    for update in the_graph.stream(state, stream_mode="values"):
                        if "plan" in update and update["plan"] is not None:
                            plan_json = json.dumps(_to_jsonable(update["plan"]))
                            yield {"event": "plan", "data": plan_json}
                        if "evidence" in update:
                            yield {"event": "evidence", "data": json.dumps(_to_jsonable(update["evidence"]))}
                        if "answers" in update:
                            yield {"event": "subanswer", "data": json.dumps(_to_jsonable(update["answers"]))}
                        if "report" in update and update["report"] is not None:
                            report = update["report"]
                            yield {"event": "token", "data": json.dumps({"t": report})}
                    finished = _now()
                    save_run(trace_id, req.query, report or "", plan_json, "{}", started, finished, path=db_path)

                    # Langfuse tracing: flush before yielding done
                    flush()

                    yield {"event": "done", "data": json.dumps({"trace_id": trace_id, "report": report})}
            finally:
                try:
                    delete_run_collection(trace_id)
                except Exception:
                    pass

        return EventSourceResponse(gen())

    @app.get("/runs/{trace_id}")
    async def get_run_endpoint(trace_id: str):
        row = get_run(trace_id, path=db_path)
        if not row:
            raise HTTPException(status_code=404)
        return row

    return app


app = build_app()
