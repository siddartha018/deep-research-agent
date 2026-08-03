from __future__ import annotations

import threading
import time
from dataclasses import dataclass

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover — legacy fallback
    from duckduckgo_search import DDGS  # type: ignore

from ..obs import observed


_MIN_INTERVAL = 1.0
_lock = threading.Lock()
_last_call = 0.0


@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str


def _pace() -> None:
    global _last_call
    with _lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


@observed(name="tool.search", as_type="tool")
def web_search(query: str, k: int = 8, *, ddgs=None) -> list[SearchHit]:
    q = query.strip()
    if not q:
        return []
    _pace()
    ctx = ddgs if ddgs is not None else DDGS()
    with ctx as ddg:
        raw = list(ddg.text(q, max_results=k) or [])
    return [
        SearchHit(url=r.get("href", ""), title=r.get("title", ""), snippet=r.get("body", ""))
        for r in raw
    ]
