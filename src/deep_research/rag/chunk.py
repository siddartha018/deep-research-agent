from __future__ import annotations

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def _tok(text: str) -> list[int]:
    return _ENC.encode(text, disallowed_special=())


def _detok(tokens: list[int]) -> str:
    return _ENC.decode(tokens)


def chunk_text(
    text: str,
    *,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
    min_chars: int = 200,
) -> list[str]:
    if not text or not text.strip():
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    tokens: list[int] = []
    for p in paragraphs:
        if tokens:
            tokens.extend(_tok("\n\n"))
        tokens.extend(_tok(p))

    if not tokens:
        return []

    chunks: list[str] = []
    step = max_tokens - overlap_tokens
    if step <= 0:
        raise ValueError("overlap_tokens must be less than max_tokens")

    i = 0
    while i < len(tokens):
        window = tokens[i : i + max_tokens]
        chunk = _detok(window).strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if i + max_tokens >= len(tokens):
            break
        i += step
    return chunks
