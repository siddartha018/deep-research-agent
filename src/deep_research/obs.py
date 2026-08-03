from __future__ import annotations

import asyncio
import functools
import os
from contextlib import contextmanager
from typing import Any, Callable

from .config import get_settings

_INITIALIZED = False
_ENABLED = False
_propagate_attributes = None


def _keys_present() -> bool:
    s = get_settings()
    return bool(s.langfuse_public_key and s.langfuse_secret_key)


def init_langfuse() -> None:
    """Initialise the Langfuse SDK. Idempotent, no-op without keys.

    Best practice: called ONCE at process start, AFTER env vars are loaded
    (see references/instrumentation.md — 'Langfuse import before env vars
    loaded' common mistake).
    """
    global _INITIALIZED, _ENABLED, _propagate_attributes
    if _INITIALIZED:
        return
    _INITIALIZED = True
    if not _keys_present():
        return
    s = get_settings()
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", s.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", s.langfuse_host)
    try:
        from langfuse import get_client, propagate_attributes
        _propagate_attributes = propagate_attributes
    except ImportError:
        _propagate_attributes = None
    get_client()
    _ENABLED = True


def is_enabled() -> bool:
    return _ENABLED


def observed(
    name: str | None = None,
    *,
    as_type: str = "span",
    capture_input: bool | None = None,
    capture_output: bool | None = None,
) -> Callable:
    """Decorator wrapper around Langfuse's ``@observe``.

    Falls back to a no-op when Langfuse is disabled so tests and offline
    runs never need keys. ``as_type`` should be the MOST SPECIFIC type
    that fits (``agent``, ``retriever``, ``tool``, ``generation``,
    ``embedding``, ``chain``, ``guardrail``, ``evaluator``) — generic
    ``span`` hides the agent graph.

    Set ``capture_input=False`` on functions whose args contain state
    dicts, secrets, or long payloads; set inputs explicitly inside with
    ``set_span_io(input=...)``.
    """
    def decorator(fn: Callable) -> Callable:
        if not _keys_present():
            return fn
        from langfuse import observe as _observe
        kw: dict[str, Any] = {"name": name, "as_type": as_type}
        if capture_input is not None:
            kw["capture_input"] = capture_input
        if capture_output is not None:
            kw["capture_output"] = capture_output
        wrapped = _observe(**kw)(fn)
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def _async(*a, **kw2):
                return await wrapped(*a, **kw2)
            return _async
        @functools.wraps(fn)
        def _sync(*a, **kw2):
            return wrapped(*a, **kw2)
        return _sync
    return decorator


def set_span_io(*, input: Any = None, output: Any = None) -> None:
    """Explicitly set input/output on the current span.

    Best practice: when using ``@observed`` on functions with sensitive
    args, disable auto-capture (``capture_input=False``) and call this to
    record only the user-visible payload.
    """
    if not _ENABLED:
        return
    from langfuse import get_client
    kwargs: dict[str, Any] = {}
    if input is not None:
        kwargs["input"] = input
    if output is not None:
        kwargs["output"] = output
    if kwargs:
        get_client().update_current_span(**kwargs)


@contextmanager
def trace_context(
    *,
    session_id: str | None = None,
    tags: list[str] | None = None,
    user_id: str | None = None,
    name: str | None = None,
    metadata: dict | None = None,
):
    """Context manager to propagate trace attributes in Langfuse v3+.

    Wraps Langfuse's ``propagate_attributes`` to set session_id, tags, etc.
    for the current trace and all child spans. Falls back to a no-op when
    Langfuse is disabled or when ``propagate_attributes`` is not available.

    Example:
        with trace_context(session_id=trace_id, tags=["deep-research"]):
            result = graph.invoke(state)
    """
    if not _ENABLED or _propagate_attributes is None:
        yield
        return
    kwargs: dict[str, Any] = {}
    if session_id is not None:
        kwargs["session_id"] = session_id
    if tags is not None:
        kwargs["tags"] = tags
    if user_id is not None:
        kwargs["user_id"] = user_id
    if name is not None:
        kwargs["name"] = name
    if metadata is not None:
        kwargs["metadata"] = metadata
    with _propagate_attributes(**kwargs):
        yield


def update_trace(
    *,
    name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> None:
    """Deprecated. Use trace_context() context manager instead.

    Note: Langfuse SDK v3+ does not support updating trace metadata after
    creation. This is a no-op; trace context should be set at trace creation
    time or via the API.
    """
    if not _ENABLED:
        return
    # Trace metadata (session_id, tags, name, user_id) cannot be updated
    # after trace creation in Langfuse v3+. They must be set when the trace
    # is initially created or via the REST API.


def score_current(
    name: str,
    value: float,
    *,
    data_type: str = "NUMERIC",
    comment: str | None = None,
) -> None:
    """Attach a score (RAGAS metric, guardrail verdict, user feedback)
    to the current trace."""
    if not _ENABLED:
        return
    from langfuse import get_client
    kwargs: dict[str, Any] = {"name": name, "value": value, "data_type": data_type}
    if comment is not None:
        kwargs["comment"] = comment
    get_client().score_current_trace(**kwargs)


def flush() -> None:
    """Force-send buffered spans. MUST be called before scripts exit —
    without it, short-lived processes (evals, CLI) lose traces.
    """
    if not _ENABLED:
        return
    from langfuse import get_client
    get_client().flush()
