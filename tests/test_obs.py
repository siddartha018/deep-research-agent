from deep_research.obs import observed, init_langfuse


def test_observed_is_noop_without_keys():
    init_langfuse()  # no keys set in test env → must not raise

    @observed(name="thing", as_type="span")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_observed_preserves_async(monkeypatch):
    import asyncio

    @observed(name="async-thing", as_type="tool")
    async def hello():
        return "hi"

    assert asyncio.run(hello()) == "hi"
