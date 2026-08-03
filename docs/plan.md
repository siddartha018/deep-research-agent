# Deep Research Agent — Implementation Plan

Sequential milestones. Each ends with a test or eval that must pass before
moving on. Estimated total effort: 4–6 focused days.

## M0 — Scaffold & keys (0.5 day)

- [ ] `pyproject.toml` with pinned deps: `langgraph>=1.0.6`, `groq>=0.11`,
      `google-genai>=1.0`, `qdrant-client[fastembed]>=1.12`, `langfuse>=3.0`,
      `ragas`, `langchain-groq`, `fastapi`, `uvicorn`, `httpx`,
      `trafilatura`, `duckduckgo-search`, `tiktoken`, `pydantic-settings`.
- [ ] `.env.example` with all keys documented.
- [ ] `src/deep_research/config.py` — `pydantic-settings` loader.
- [ ] `src/deep_research/llm.py` — Groq client + Gemini fallback wrapper
      (`chat(messages, tools=None, stream=False)` interface).
- [ ] Smoke test: `python -m deep_research.llm --ping` prints one token from
      Groq and one from Gemini.

## M1 — RAG plumbing (1 day)

- [ ] `rag/qdrant.py` — `create_run_collection(trace_id)`,
      `delete_run_collection(trace_id)`; dense (768d) + sparse (BM25) config.
- [ ] `rag/chunk.py` — token-aware splitter (800/100, paragraph-bounded).
- [ ] `rag/embed.py` — Gemini `embed_content` with batching (100 per call).
- [ ] `tools/retrieve.py` — hybrid `query_points` with RRF fusion.
- [ ] Tests:
  - `test_chunk.py` — chunks respect boundaries, overlap correct.
  - `test_retrieve.py` — round-trip: embed 20 fixture chunks, retrieve top-5,
    assert recall on 3 known-relevant chunks.

## M2 — Tools + guardrails (0.5 day)

- [ ] `tools/search.py` — DuckDuckGo, 1 req/s rate limit.
- [ ] `tools/fetch.py` — httpx GET, URL allowlist (no localhost / private IPs
      / non-http schemes), trafilatura extract, injection-marker scrub.
- [ ] `tools/cite.py` — pure function, no LLM.
- [ ] Test `test_fetch_guardrails.py` — `file://`, `http://127.0.0.1`,
      `http://192.168.1.1`, and a page containing `Ignore previous
      instructions` are all rejected or scrubbed.

## M3 — Agent nodes (1.5 days)

- [ ] `state.py` — `ResearchState` TypedDict, `SubQuestion`, `SubAnswer`,
      `Chunk`, `Citation` dataclasses.
- [ ] `nodes/planner.py` — prompt + JSON-mode parse → `plan`.
- [ ] `nodes/executor.py` — ReAct agent bound to `web_search`, `fetch`,
      `retrieve`, `cite`. Uses `create_react_agent` from `langgraph.prebuilt`.
- [ ] `nodes/critic.py` — per-sub-question coverage scorer; emits `gaps`.
- [ ] `nodes/writer.py` — streamed generation with citation-regex retry.
- [ ] `graph.py` — wire nodes; conditional edge critic → (writer | planner);
      cap iterations.
- [ ] `test_graph_smoke.py` — with mocked tools (fixture pages), full graph
      returns a report with ≥1 citation.

## M4 — Serving (0.5 day)

- [ ] `api.py` — `POST /research` SSE endpoint emitting `plan`, `evidence`,
      `subanswer`, `token`, `done`.
- [ ] `GET /runs/{trace_id}` — SQLite lookup.
- [ ] `store.py` — SQLAlchemy Core (no ORM), one `runs` table.
- [ ] Manual smoke: `curl -N ... /research` prints streamed events.

## M5 — Observability (0.5 day)

- [ ] `obs.py` — Langfuse init + shared `@observe` decorators.
- [ ] Decorate every node, tool, and LLM call.
- [ ] Verify in Langfuse UI: one trace per run, spans nested correctly, token
      usage populated.

## M6 — Evals (1 day)

- [ ] `evals/datasets/queries.jsonl` — 10 questions with reference answers
      and expected topics (curated by hand).
- [ ] `evals/ragas_eval.py` — run pipeline for each query, compute metrics
      with `langchain-groq` evaluator, write `results/ragas_<ts>.json`.
- [ ] `evals/red_team.py` — 10 injection payloads in mock pages, run agent
      offline against a stub fetcher.
- [ ] `evals/latency_cost.py` — record wall-clock and Groq request counts.
- [ ] Commit a baseline result set to `evals/results/`.

## M7 — Polish (0.5 day)

- [ ] Top-level `README` demo: one working curl + a screenshot of the
      Langfuse trace + a link to `evals/results/`.
- [ ] Short "what I'd do next" section in `README` (persistent KB, PDF/arXiv
      ingestion, reranker).

## Acceptance gate (spec §3)

Only mark project complete when the latest `evals/results/ragas_*.json`
shows faithfulness ≥ 0.80, context precision ≥ 0.70, citation coverage ≥
0.90, red-team pass rate ≥ 0.80, and median latency ≤ 60s.
