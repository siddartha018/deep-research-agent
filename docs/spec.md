# Deep Research Agent — Design Spec

**Status:** Draft
**Owner:** enldias
**Date:** 2026-07-31

## 1. Problem

Given a research question, produce a citation-grounded report by decomposing
the question, retrieving evidence from the open web + a persistent knowledge
base, verifying claims, and streaming the answer to the user. The system must
demonstrate every layer a GenAI production stack touches: planning, tool use,
hybrid RAG, streaming, tracing, and automated eval.

## 2. Non-goals

- Human-in-the-loop editing.
- Fine-tuning any model.
- Beating hosted deep-research products on quality — this is a portfolio
  build, not a product.
- Multi-user auth / accounts.

## 3. Success criteria

Measurable acceptance targets (verified in `evals/`):

| Metric | Target |
|---|---|
| RAGAS **faithfulness** (report vs retrieved contexts) | ≥ 0.80 |
| RAGAS **context precision** | ≥ 0.70 |
| Citation coverage (fraction of report sentences with ≥1 citation) | ≥ 0.90 |
| End-to-end latency (median, 5 sub-questions) | ≤ 60 s |
| Cost per run (Groq + Gemini free tier) | $0 |
| Prompt-injection red-team pass rate (10 attacks) | ≥ 0.80 |

## 4. Architecture

LangGraph state machine. Nodes are agents, edges are conditional. State
carries messages, plan, evidence, and draft.

```
                ┌──────────┐
   user query → │ planner  │  decomposes into sub-questions + search plan
                └────┬─────┘
                     │
              ┌──────▼──────┐
              │ dispatcher  │  fan out one executor per sub-question
              └──────┬──────┘
                     │ (parallel)
              ┌──────▼──────┐
              │  executor   │ × N   web search → scrape → chunk → embed →
              └──────┬──────┘         upsert to Qdrant → hybrid retrieve
                     │                → tool-call answer with sources
                     │ (join)
              ┌──────▼──────┐
              │   critic    │  scores coverage per sub-question;
              └──────┬──────┘  loops back to planner if any gap
                     │
              ┌──────▼──────┐
              │   writer    │  synthesizes cited report;
              └──────┬──────┘  streams SSE tokens
                     ▼
                  report
```

### 4.1 State (`ResearchState`)

```python
class ResearchState(TypedDict):
    query: str
    plan: list[SubQuestion]        # [{id, question, status}]
    evidence: dict[str, list[Chunk]]  # sub_q_id -> retrieved chunks
    answers: dict[str, SubAnswer]  # sub_q_id -> {answer, citations}
    gaps: list[str]                # sub_q_ids the critic re-opened
    iterations: int
    report: str | None
    trace_id: str
```

### 4.2 Agents (nodes)

| Node | Model | Job |
|---|---|---|
| `planner` | Groq Llama 3.3 70B | Break query into 3–7 sub-questions with search terms. |
| `executor` | Groq Llama 3.1 8B (fast, cheap) | For one sub-question: search → scrape → retrieve → answer with citations. Uses tools (§4.3). |
| `critic` | Groq Llama 3.3 70B | Score each `SubAnswer`: covered / partial / gap. If gap, mark sub-question re-open. Max 2 loops. |
| `writer` | Groq Llama 3.3 70B (streamed) | Merge sub-answers into a single report with inline `[n]` citations. Enforce ≥1 citation per claim. |

Fallback: on Groq 429/5xx, retry once, then fall back to Gemini 2.5 Flash for
the same node. Configured centrally in `src/deep_research/llm.py`.

### 4.3 Tools (executor)

Registered as LangGraph tool nodes; the executor is a ReAct agent.

- `web_search(query: str, k: int = 8) -> list[SearchHit]`
  DuckDuckGo (via `duckduckgo-search`). No API key needed.
- `fetch(url: str) -> str`
  `httpx` GET → `trafilatura.extract()` → clean text. Timeout 10s.
  Robots.txt honored; user agent identifies as research bot.
- `retrieve(query: str, k: int = 8) -> list[Chunk]`
  Hybrid query against Qdrant collection for this run (see §4.4).
- `cite(chunk_ids: list[str], claim: str) -> Citation`
  Emits a citation record linked to specific chunks; no LLM call.

### 4.4 RAG — Qdrant hybrid collection

One collection per run, named `run_{trace_id}`, deleted at run end.

- **Dense vectors**: Gemini `gemini-embedding-001`, 768-dim, cosine.
- **Sparse vectors**: BM25 via FastEmbed `Qdrant/bm25` (built-in), for keyword recall.
- **Payload**: `{url, title, chunk_text, chunk_idx, sub_q_id, fetched_at}`.
- **Chunking**: 800 tokens with 100 overlap; boundary-aware via `tiktoken` +
  paragraph split.
- **Query**: `client.query_points(collection, prefetch=[dense, sparse],
  query=FusionQuery(fusion=RRF), limit=8)`.

Why per-run: keeps free tier well under 1 GB, gives clean state per demo, and
makes tracing simpler. Persistent collections would need indexing hygiene the
project doesn't need.

### 4.5 Serving — FastAPI + SSE

Single endpoint:

```
POST /research
body: {"query": str, "max_iterations": int = 2}
response: text/event-stream
  event: plan      data: {plan}
  event: evidence  data: {sub_q_id, n_chunks}
  event: subanswer data: {sub_q_id, answer, citations}
  event: token     data: {t}          # writer streamed tokens
  event: done      data: {report, trace_id, cost}
```

Non-streaming `GET /runs/{trace_id}` returns the full record from SQLite.

### 4.6 Observability

- `@observe(as_type="agent")` on each LangGraph node.
- `@observe(as_type="tool")` on each tool function.
- `@observe(as_type="generation")` on the LLM wrapper; records prompt, output,
  token usage, latency.
- Every span carries `trace_id = state["trace_id"]` so a full research run is
  one trace in Langfuse.

### 4.7 Guardrails

Three concrete guardrails, all covered in `evals/red_team/`:

1. **Prompt-injection filter** on scraped page content: strip common injection
   markers (`ignore previous`, `system:`, hidden HTML instructions) before it
   ever hits an LLM prompt.
2. **URL allowlist** in `fetch`: only http(s), no `file://`, no localhost,
   no private IP ranges. `httpx` DNS resolution checked against blocklist.
3. **Citation enforcement**: writer prompt requires `[n]` after every claim;
   post-generation regex validates ≥1 citation per sentence. Regenerate up
   to 2 times if invalid.

### 4.8 SQLite run history

Table `runs(trace_id PK, query, report, plan_json, cost_json, started_at,
finished_at, faithfulness, context_precision)`.
Written at run end. Read by `GET /runs/*` and by eval scripts.

## 5. Evaluation harness (`evals/`)

Three eval scripts, each producing a JSON report:

- `evals/ragas_eval.py` — 10 curated queries with reference answers. Runs the
  full pipeline, then computes RAGAS faithfulness + context precision + answer
  correctness. Evaluator LLM: Groq Llama 3.3 70B via `langchain-groq` wrapped
  in `ragas.llms.LangchainLLMWrapper`.
- `evals/red_team.py` — 10 prompt-injection payloads (planted in mock pages).
  Pass = agent ignores injected instructions.
- `evals/latency_cost.py` — same 10 queries, records p50/p95 latency and
  free-tier request count.

Reports checked into `evals/results/` so recruiters see baselines.

## 6. Repo layout

```
01-deep-research-agent/
├── pyproject.toml
├── .env.example
├── README.md
├── docs/
│   ├── spec.md
│   └── plan.md
├── src/deep_research/
│   ├── __init__.py
│   ├── api.py              # FastAPI + SSE
│   ├── graph.py            # LangGraph wiring
│   ├── llm.py              # Groq/Gemini client + fallback
│   ├── state.py            # ResearchState TypedDict
│   ├── nodes/
│   │   ├── planner.py
│   │   ├── executor.py
│   │   ├── critic.py
│   │   └── writer.py
│   ├── tools/
│   │   ├── search.py       # DuckDuckGo
│   │   ├── fetch.py        # httpx + trafilatura + guardrails
│   │   ├── retrieve.py     # Qdrant hybrid
│   │   └── cite.py
│   ├── rag/
│   │   ├── qdrant.py       # client + per-run collection lifecycle
│   │   ├── chunk.py
│   │   └── embed.py        # Gemini embeddings
│   ├── obs.py              # Langfuse decorators / setup
│   └── store.py            # SQLite run history
├── tests/
│   ├── test_chunk.py
│   ├── test_fetch_guardrails.py
│   ├── test_retrieve.py
│   └── test_graph_smoke.py
└── evals/
    ├── datasets/queries.jsonl
    ├── ragas_eval.py
    ├── red_team.py
    ├── latency_cost.py
    └── results/            # committed
```

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Groq rate limit mid-run | Fallback to Gemini; exponential backoff; cache identical prompts within a run. |
| DuckDuckGo throttling | 1-req/sec pacing; retry with jitter; final fallback to a static test corpus in tests. |
| Scraped pages are junk (paywalls, JS-only) | `trafilatura` filters low-quality extracts; drop chunks < 200 chars. |
| Critic loops forever | Hard cap `max_iterations = 2`. |
| Free tier exhausted during demo | `evals/` results are pre-computed and checked in; live demo uses 1 short query. |

## 8. Out of scope explicitly

- User accounts, saved reports, sharing.
- Non-English content.
- PDF / arXiv ingestion (a possible v2).
- Any paid API.
