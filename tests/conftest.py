import os
import pytest


@pytest.fixture(autouse=True)
def _test_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-qdrant")
    monkeypatch.setenv("DEEP_RESEARCH_DB_PATH", str(tmp_path / "dr.sqlite"))
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    from deep_research.config import get_settings
    get_settings.cache_clear()
    from deep_research import obs
    obs._INITIALIZED = False
    obs._ENABLED = False
