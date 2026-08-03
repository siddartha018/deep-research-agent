from __future__ import annotations

from sqlalchemy import Column, Float, MetaData, String, Table, create_engine, insert, select

from .config import get_settings


_metadata = MetaData()

_runs = Table(
    "runs",
    _metadata,
    Column("trace_id", String, primary_key=True),
    Column("query", String, nullable=False),
    Column("report", String, nullable=False),
    Column("plan_json", String, nullable=False),
    Column("cost_json", String, nullable=False),
    Column("started_at", String, nullable=False),
    Column("finished_at", String, nullable=False),
    Column("faithfulness", Float, nullable=True),
    Column("context_precision", Float, nullable=True),
)


def _engine(path: str | None):
    path = path or get_settings().deep_research_db_path
    return create_engine(f"sqlite:///{path}", future=True)


def init_db(path: str | None = None) -> None:
    _metadata.create_all(_engine(path))


def save_run(
    trace_id: str,
    query: str,
    report: str,
    plan_json: str,
    cost_json: str,
    started_at: str,
    finished_at: str,
    *,
    path: str | None = None,
) -> None:
    eng = _engine(path)
    init_db(path)
    with eng.begin() as conn:
        conn.execute(insert(_runs).values(
            trace_id=trace_id, query=query, report=report,
            plan_json=plan_json, cost_json=cost_json,
            started_at=started_at, finished_at=finished_at,
        ))


def get_run(trace_id: str, *, path: str | None = None) -> dict | None:
    eng = _engine(path)
    init_db(path)
    with eng.connect() as conn:
        row = conn.execute(select(_runs).where(_runs.c.trace_id == trace_id)).mappings().first()
        return dict(row) if row else None
