from __future__ import annotations

from ..obs import observed
from ..state import Citation


@observed(name="tool.cite", as_type="tool")
def cite(chunk_ids: list[str], claim: str) -> Citation:
    if not chunk_ids:
        raise ValueError("cite() requires at least one chunk_id")
    seen: dict[str, None] = {}
    for cid in chunk_ids:
        seen.setdefault(cid, None)
    return Citation(claim=claim, chunk_ids=list(seen.keys()))
