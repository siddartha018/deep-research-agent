from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from groq import Groq
from google import genai
from google.genai import types as genai_types

from .config import get_settings
from . import obs


_MODEL_MAP = {
    "primary":  ("groq",   "llama-3.3-70b-versatile"),
    "fast":     ("groq",   "llama-3.1-8b-instant"),
    "fallback": ("gemini", "gemini-flash-latest"),
}

_EMBED_MODEL = "gemini-embedding-001"
_EMBED_DIM = 768
_EMBED_BATCH = 100

_LOCAL_EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # 768-dim, matches Qdrant config


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    model: str = ""
    usage: dict = field(default_factory=dict)


def _redact_messages(messages: list[dict]) -> list[dict]:
    """Best-effort scrub: strip anything that looks like a bearer token
    or long secret before it reaches Langfuse."""
    out = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str) and ("api_key" in c.lower() or "bearer " in c.lower()):
            c = "[redacted]"
        out.append({"role": m.get("role", "user"), "content": c})
    return out


class LLM:
    def __init__(self) -> None:
        s = get_settings()
        self._groq = Groq(api_key=s.groq_api_key)
        self._gemini = genai.Client(api_key=s.gemini_api_key)
        self._local_embed = None  # lazy — loaded on first Gemini failure

    def _resolve(self, alias: str) -> tuple[str, str]:
        if alias not in _MODEL_MAP:
            raise ValueError(f"unknown model alias: {alias}")
        return _MODEL_MAP[alias]

    def chat(
        self,
        messages: list[dict],
        *,
        model: str = "primary",
        tools: list[dict] | None = None,
        stream: bool = False,
        temperature: float = 0.2,
    ) -> ChatResult:
        """Chat with primary/fast Groq model, retry on 429 (Groq's free
        tier resets per-minute), fall back to Gemini only after exhausting
        Groq retries. Gemini's free tier (5 req/min on flash-latest) is
        tighter than Groq's, so retrying Groq is almost always cheaper
        than switching.
        """
        provider, name = self._resolve(model)
        if provider == "gemini" or model == "fallback":
            return self._gemini_chat(messages, name, tools, temperature)

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                return self._groq_chat(messages, name, tools, stream, temperature)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                is_ratelimit = "429" in msg or "rate limit" in msg or "quota" in msg
                if not is_ratelimit and attempt >= 1:
                    break
                time.sleep(2.0 * (attempt + 1))
        try:
            _, gname = self._resolve("fallback")
            return self._gemini_chat(messages, gname, tools, temperature)
        except Exception:
            raise last_err if last_err else RuntimeError("chat failed")

    def _emit_generation(
        self,
        *,
        name: str,
        model: str,
        messages: list[dict],
        temperature: float,
        result: ChatResult,
    ) -> None:
        """Emit a `generation` observation for one LLM call with model,
        prompt, output, and token usage.

        Following Langfuse baseline requirements (references/
        instrumentation.md §2): every LLM call needs model name +
        input/output tokens captured to enable cost calculation and
        model comparison.
        """
        if not obs.is_enabled():
            return
        try:
            from langfuse import get_client
            client = get_client()
            usage = result.usage or {}
            usage_details: dict[str, int] = {}
            if "prompt_tokens" in usage:
                usage_details["input"] = int(usage.get("prompt_tokens") or 0)
            if "completion_tokens" in usage:
                usage_details["output"] = int(usage.get("completion_tokens") or 0)
            if "total_tokens" in usage:
                usage_details["total"] = int(usage.get("total_tokens") or 0)
            span = client.start_observation(
                as_type="generation",
                name=name,
                model=model,
                input=_redact_messages(messages),
                output=result.content,
                model_parameters={"temperature": temperature},
                usage_details=usage_details or None,
            )
            span.end()
        except Exception:
            pass

    def _groq_chat(self, messages, name, tools, stream, temperature) -> ChatResult:
        kwargs: dict[str, Any] = {"messages": messages, "model": name, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self._groq.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        result = ChatResult(
            content=msg.content or "",
            tool_calls=[tc.model_dump() for tc in (msg.tool_calls or [])],
            model=name,
            usage=resp.usage.model_dump() if resp.usage else {},
        )
        self._emit_generation(
            name=f"groq:{name}", model=name, messages=messages,
            temperature=temperature, result=result,
        )
        return result

    def _gemini_chat(self, messages, name, tools, temperature) -> ChatResult:
        contents = _messages_to_gemini(messages)
        cfg = genai_types.GenerateContentConfig(temperature=temperature)
        resp = self._gemini.models.generate_content(
            model=name, contents=contents, config=cfg
        )
        usage_meta = getattr(resp, "usage_metadata", None)
        usage: dict[str, int] = {}
        if usage_meta is not None:
            usage = {
                "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(usage_meta, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(usage_meta, "total_token_count", 0) or 0,
            }
        result = ChatResult(
            content=resp.text or "", tool_calls=[], model=name, usage=usage,
        )
        self._emit_generation(
            name=f"gemini:{name}", model=name, messages=messages,
            temperature=temperature, result=result,
        )
        return result

    def _embed_gemini(self, batch: list[str]) -> list[list[float]]:
        resp = self._gemini.models.embed_content(
            model=_EMBED_MODEL,
            contents=batch,
            config=genai_types.EmbedContentConfig(output_dimensionality=_EMBED_DIM),
        )
        return [e.values for e in resp.embeddings]

    def _embed_local(self, batch: list[str]) -> list[list[float]]:
        if self._local_embed is None:
            from fastembed import TextEmbedding
            self._local_embed = TextEmbedding(model_name=_LOCAL_EMBED_MODEL)
        return [v.tolist() for v in self._local_embed.embed(batch)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, falling back to local BGE-base when Gemini free
        tier is exhausted or the network is unavailable. Local model is
        also 768-dim, so downstream Qdrant collections work unchanged.
        """
        out: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            batch = texts[i : i + _EMBED_BATCH]
            provider = "gemini"
            try:
                values = self._embed_gemini(batch)
            except Exception:
                values = self._embed_local(batch)
                provider = "local"
                # On first fallback stick to local for the rest of the
                # call — retrying Gemini per-batch would burn latency.
                for j in range(i + _EMBED_BATCH, len(texts), _EMBED_BATCH):
                    b2 = texts[j : j + _EMBED_BATCH]
                    out.extend(self._embed_local(b2))
                    self._emit_embedding_span("local", _LOCAL_EMBED_MODEL, b2, _EMBED_DIM)
                out.extend(values)
                self._emit_embedding_span(provider, _LOCAL_EMBED_MODEL, batch, _EMBED_DIM)
                return out
            out.extend(values)
            self._emit_embedding_span(provider, _EMBED_MODEL, batch, _EMBED_DIM)
        return out

    def _emit_embedding_span(
        self, provider: str, model: str, batch: list[str], dim: int
    ) -> None:
        if not obs.is_enabled():
            return
        try:
            from langfuse import get_client
            span = get_client().start_observation(
                as_type="embedding",
                name=f"{provider}:{model}",
                model=model,
                input={"n_texts": len(batch)},
                output={"dim": dim, "count": len(batch)},
            )
            span.end()
        except Exception:
            pass


def _messages_to_gemini(messages: list[dict]) -> list[dict]:
    parts: list[dict] = []
    for m in messages:
        role = "user" if m["role"] in ("user", "system") else "model"
        parts.append({"role": role, "parts": [{"text": m["content"]}]})
    return parts


@lru_cache
def get_llm() -> LLM:
    return LLM()
