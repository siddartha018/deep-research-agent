# Deep Research Agent

A multi-agent research pipeline that turns a research question into a
citation-grounded report streamed over SSE. Built end-to-end on free-tier
APIs (Groq, Gemini, Qdrant Cloud, Langfuse) using Python.

**One-line pitch:** planner decomposes → executors run hybrid RAG per
sub-question → critic verifies coverage → writer synthesizes a cited
report → everything traced in Langfuse and evaluated with RAGAS.

---

## Table of Contents

1. [Why this project](#why-this-project)
2. [Architecture](#architecture)
3. [Stack](#stack)
4. [Quickstart](#quickstart)
5. [How to test](#how-to-test)


---

## Why this project



| Signal | Where you see it |
|---|---|
| Multi-agent planning + tool use | `nodes/planner.py`, `nodes/executor.py`, `nodes/critic.py`, `nodes/writer.py` |
| Hybrid RAG (dense + sparse, RRF fusion) | `rag/qdrant.py`, `tools/retrieve.py` |
| Streaming serving | `api.py` — FastAPI + SSE |
| Observability | `obs.py` — Langfuse `@observed` on every node/tool/generation |
| Guardrails | `tools/fetch.py` (SSRF + injection scrub), writer citation regex |
| Automated evals | `evals/` — RAGAS + red-team + latency |
| LLM fallback + cost control | `llm.py` — Groq→Gemini→local BGE fallback chain |



---

## Architecture

```
                ┌──────────┐
   user query → │ planner  │  decomposes into 3–7 sub-questions
                └────┬─────┘
                     │
              ┌──────▼──────┐
              │ dispatcher  │  fan out one executor per sub-question
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  executor   │ × N   web search → scrape → chunk → embed →
              └──────┬──────┘         upsert to Qdrant → hybrid retrieve
                     │                → sub-answer with citations
              ┌──────▼──────┐
              │   critic    │  scores coverage per sub-question;
              └──────┬──────┘  loops back to dispatcher if any gap (cap = 2)
                     │
              ┌──────▼──────┐
              │   writer    │  synthesizes cited report;
              └──────┬──────┘  regex-validates ≥ 1 citation per sentence
                     ▼
                  report
```

The whole thing is a `langgraph.StateGraph`. State carries
`plan`, `evidence`, `answers`, `gaps`, `iterations`, `report`, `trace_id`.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Agent framework | **LangGraph 1.0.x** | State-graph shape, per-node observability, first-class conditional edges |
| Primary LLM | **Groq** — `llama-3.3-70b-versatile` (planner/critic/writer), `llama-3.1-8b-instant` (executor tool routing) | 14k req/day free tier, fastest inference |
| Fallback LLM | **Gemini `gemini-flash-latest`** | Kicks in only after 3× Groq retries fail |
| Embeddings | **Gemini `gemini-embedding-001`** (768-dim) with **local FastEmbed BGE-base-en-v1.5** fallback | Same 768-dim → Qdrant config never changes |
| Vector store | **Qdrant Cloud** — 1 GB free forever | Per-run collection, dense + sparse (BM25), RRF fusion via `query_points` |
| Web tools | **`ddgs`** (DuckDuckGo successor) + `httpx` + `trafilatura` | No API keys |
| Serving | **FastAPI + `sse-starlette`** | SSE stream: `plan` → `evidence` → `subanswer` → `token` → `done` |
| Tracing | **Langfuse Cloud** — `@observe` decorators | Per-node/per-tool spans, session_id = trace_id, token usage captured |
| Evals | **RAGAS** (faithfulness / context precision / answer relevancy) + red-team injection set + latency | RAGAS judge LLM = Groq via `langchain-groq` |
| Run history | **SQLite** via SQLAlchemy Core | Zero setup, per-project file |

Everything above has a free tier and no card required.

---

## Quickstart

```bash
cd deep-research-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env with your keys (see below)
```

### Required environment variables

Paste into `.env`:

```
GROQ_API_KEY=              # https://console.groq.com/keys
GEMINI_API_KEY=            # https://aistudio.google.com/apikey
QDRANT_URL=                # https://cloud.qdrant.io
QDRANT_API_KEY=
LANGFUSE_PUBLIC_KEY=       # https://cloud.langfuse.com (optional)
LANGFUSE_SECRET_KEY=       # optional — Langfuse degrades to no-op if unset
LANGFUSE_HOST=https://cloud.langfuse.com
DEEP_RESEARCH_MAX_ITERATIONS=2
DEEP_RESEARCH_DB_PATH=./deep_research.sqlite
```

### One-shot demo

```bash
python -c "
from deep_research.graph import run
state = run('What is retrieval-augmented generation?', trace_id='demo-1')
print(state['report'][:800])
"
```



### Streaming demo

```bash
uvicorn deep_research.api:app --reload &
curl -N http://localhost:8000/research \
  -H 'content-type: application/json' \
  -d '{"query": "What is retrieval-augmented generation?"}'
```

You'll see SSE events stream in real time.

---

## How to test

### 1 — Unit tests (no keys, no cost)

```bash
source .venv/bin/activate
pytest -v
```

Expected: **55 passed** in ~3–5 s. Covers every module: chunker,
guardrails, Qdrant lifecycle, hybrid retrieval, planner/critic/writer,
graph smoke, SSE endpoint, store, red-team scoring logic.

### 2 — Live sanity checks

```bash
# Qdrant Cloud
python -c "from deep_research.rag.qdrant import get_client; print(get_client().get_collections())"

# Langfuse
python -c "from deep_research.obs import init_langfuse, is_enabled; init_langfuse(); print('langfuse:', is_enabled())"

# LLM router (fast model, cheapest sanity ping)
python -c "from deep_research.llm import get_llm; print(get_llm().chat([{'role':'user','content':'say hi'}], model='fast').content)"

# Embeddings (Gemini with automatic local fallback)
python -c "from deep_research.llm import get_llm; v = get_llm().embed(['hello']); print('embed_dim:', len(v[0]))"
```

### 3 — Full eval batch

Only when Groq's daily quota is fresh (free tier resets midnight UTC).

```bash
python -m evals.latency_cost --queries evals/datasets/queries.jsonl --out evals/results/latency_baseline.json
python -m evals.red_team     --payloads evals/datasets/red_team.jsonl --out evals/results/red_team_baseline.json
python -m evals.ragas_eval   --queries evals/datasets/queries.jsonl --out evals/results/ragas_baseline.json
```

Each script is **resume-safe**: writes results incrementally, so a
mid-batch API failure leaves you with partial data rather than nothing.

---


