from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

import httpx
import trafilatura

from ..obs import observed


_UA = "deep-research-agent/0.1 (+github.com/enldias)"
_INJECT_PATTERNS = [
    re.compile(r"ignore\s+(the\s+)?previous(\s+instructions?)?", re.I),
    re.compile(r"ignore\s+the\s+above", re.I),
    re.compile(r"(?m)^\s*system\s*:", re.I),
    re.compile(r"<\|im_start\|>", re.I),
    re.compile(r"<system>.*?</system>", re.I | re.S),
]


def is_url_allowed(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").lower()
    if not host or host == "localhost":
        return False
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return True  # not an IP literal → OK at this layer
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
        return False
    return True


def scrub_injection(text: str) -> str:
    out = text
    for pat in _INJECT_PATTERNS:
        out = pat.sub("[filtered]", out)
    return out


@observed(name="tool.fetch", as_type="tool")
def fetch(url: str, *, timeout: float = 10.0, client: httpx.Client | None = None) -> str:
    if not is_url_allowed(url):
        raise ValueError(f"url not allowed: {url}")
    owns = client is None
    c = client or httpx.Client(
        headers={"User-Agent": _UA}, timeout=timeout, follow_redirects=True
    )
    try:
        r = c.get(url)
        r.raise_for_status()
        extracted = trafilatura.extract(r.text) or ""
    finally:
        if owns:
            c.close()
    return scrub_injection(extracted)
