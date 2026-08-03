from types import SimpleNamespace

from deep_research.tools import search as S


class FakeDDGS:
    def __init__(self, hits):
        self._hits = hits

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def text(self, query, max_results):
        return self._hits[:max_results]


def test_returns_up_to_k_hits():
    fake = FakeDDGS([
        {"href": f"http://x/{i}", "title": f"t{i}", "body": f"s{i}"} for i in range(20)
    ])
    hits = S.web_search("q", k=5, ddgs=fake)
    assert len(hits) == 5
    assert hits[0].url == "http://x/0"
    assert hits[0].title == "t0"
    assert hits[0].snippet == "s0"


def test_empty_query_returns_empty():
    assert S.web_search("   ", k=5, ddgs=FakeDDGS([])) == []
