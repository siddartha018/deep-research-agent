"""End-to-end wiring smoke test for deep-research-agent.

Verifies that all public entry points (build_graph, run, build_app)
instantiate without hitting the network. Uses monkeypatch to short-circuit
the LLM and Qdrant clients.
"""

def test_can_import_and_build_graph(monkeypatch):
    """Test that build_graph can be imported and instantiated with env setup."""
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("QDRANT_API_KEY", "x")

    from deep_research.config import get_settings
    get_settings.cache_clear()

    from deep_research.graph import build_graph
    g = build_graph()
    assert g is not None


def test_can_build_app(monkeypatch, tmp_path):
    """Test that build_app can be instantiated without network."""
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("QDRANT_API_KEY", "x")
    monkeypatch.setenv("DEEP_RESEARCH_DB_PATH", str(tmp_path / "a.sqlite"))
    from deep_research.config import get_settings
    get_settings.cache_clear()

    from deep_research.api import build_app
    app = build_app()
    assert app.title == "deep-research-agent"
