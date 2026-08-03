from deep_research.store import get_run, init_db, save_run


def test_roundtrip(tmp_path):
    path = str(tmp_path / "r.sqlite")
    init_db(path)
    save_run(
        "t1", "what?", "the report [1].",
        plan_json='{"sub_questions": []}', cost_json='{"tokens": 42}',
        started_at="2026-07-31T00:00:00Z", finished_at="2026-07-31T00:01:00Z",
        path=path,
    )
    row = get_run("t1", path=path)
    assert row["trace_id"] == "t1"
    assert row["report"] == "the report [1]."
    assert row["cost_json"] == '{"tokens": 42}'


def test_missing_returns_none(tmp_path):
    path = str(tmp_path / "r.sqlite")
    init_db(path)
    assert get_run("nope", path=path) is None
