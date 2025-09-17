# Execution Plan — Plant-Disease-RAG Assistant

Legend: `- [ ]` = TODO · `- [x]` = DONE · `- [~]` = IN PROGRESS

---

## Immediate Focus — Skeleton E2E Freeze (before disease expansion)

**Objective:**  
Ship a minimal but runnable vertical slice (ingestion → retrieval → RAG → UI/API) that a new user can start in <10 min.

**Scope (DO NOW):**
- [x] API: minimal FastAPI app (/health, /rag) reusing RAGPipeline
- [x] Make targets: make api, make ui, make run (compose), make sample_index (tiny KB)
- [x] Tiny sample KB (2 plants, 3 diseases) under data/sample_kb/ + index build
- [ ] README Quick Start (clone → make sample_index → make ui → ask question)
- [ ] Streamlit: link to API usage (curl example) + footer with version/hash
- [ ] Basic telemetry: log answer latency & retrieved count to stdout
- [ ] Execution plan cleanup (remove duplicate legacy section)
- [ ] Add docs/ARCHITECTURE.md (diagram + 1‑page flow)
- [ ] .env.example updated (OPENAI_API_KEY, INDEX_DIR, MODEL_NAME)
- [x] Docker: Dockerfile + docker-compose (api + ui) + Make targets (docker_build/up/down/logs)

**Acceptance:**
- Fresh clone + sample index: user gets an answer with sources (no manual edits)
- All tests pass locally (pytest -q)
- Ruff clean (ruff check .)
- Docker image runs UI + API with sample index

**Deferrals (NEXT AFTER FREEZE):**
- Disease/plant taxonomy expansion (deferred until after cloud app is working)
- Reranker & fusion tuning
- Citation sentence post‑processing
- Feedback logging
- HF Space deployment
- Full KB rebuild & eval reruns

---

## Milestone 0 — Foundation & Scaffolding — Done
Goal: Set the repo so everything is reproducible and testable.

- [x] Project scaffold ready: `src/`, `data/`, `docs/`, `tests/`, `models/`, `artifacts/`
- [x] Classifier project cloned/migrated into this repo (reuse components where helpful)
- [x] Dev flow: `docker compose up --build` starts local app (Streamlit UI on :8501)
- [x] Pinned dependencies + `.env.example`
- [x] Basic tests: `pytest -q` runs at least one placeholder test
- [x] Makefile basics (`run`, `test`, etc.)
- [x] README Quick Start + initial architecture diagram/notes
- [x] Testing guide: docs/testing.md (how to run mock vs. OpenAI integration tests; logging tips)
- [x] Data wrangling kick-off: scraper started for PlantVillage & Wikipedia

**Acceptance checks**
- [x] Fresh clone → `docker compose up --build` boots the UI locally
- [x] `pytest -q` succeeds on the skeleton tests

---

## Milestone 1 — Knowledge Base (KB) Corpus — Done
Goal: Build a clean, deduplicated, well‑tagged KB from PlantVillage + Wikipedia.
Status: Done (first version). Ongoing: normalization polish and more tests.

How to build (current state):
- PV + Wikipedia:
  - python -m src.ingestion.build_kb --sources plantvillage,wikipedia --out data\kb --min_tokens 50 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --wiki-lang en --wiki-interval 0.5 --verbose
- Outputs:
  - data/kb/chunks/*.md (chunk files)
  - data/kb/manifest.parquet (or CSV fallback)

- src/ingestion/refresh_kb_descriptions.py (PV refresher: summary/symptoms/cause)
- src/ingestion/validate_kb_urls.py (URL validator/fixer)
- scripts/plantvillage_scraper.py, scripts/verify_scoped_parsing.py, scripts/parse_html_debug.py (dev tools)

- [~] Ingestion entrypoint
  - [x] PlantVillage refresher: src/ingestion/refresh_kb_descriptions.py
  - [x] Unified builder: src/ingestion/build_kb.py (scaffold with CLI and chunking)
  - [x] CLI: --sources plantvillage,wikipedia --out data/kb --min_tokens 50 --max_tokens 1000 --overlap 100
  - [x] Respect robots.txt, polite rate limiting, retry/backoff
  - [~] Normalize to markdown/text; strip boilerplate/nav
  - [x] Metadata: doc_id, url, title, plant, disease, section, lang, crawl_date
- [x] Chunking (512–1,000 tokens), sentence‑aware splits + overlap
- [x] Deduplication (MinHash/LSH) across pages and chunks
- [x] Manifest: data/kb/manifest.parquet with  
      doc_id, url, title, plant, disease, split_idx, text, n_tokens, lang, crawl_date
- [x] Make target: make kb (runs build_kb, writes manifest + chunks)
- [x] Data card: docs/data_card.md (sources, licenses, cleaning steps, limitations)
- [~] Unit tests: tests/test_ingestion.py (chunk lengths, metadata presence, dedup sanity)

**Acceptance checks**
- [x] One command builds the KB end‑to‑end
- [x] Spot‑check 10 random chunks → clean text & correct tags

---

## Milestone 2 — Retrieval (Hybrid Search) — Done
Goal: High‑recall retrieval fusing lexical + vector.

- [x] Lexical search
  - [x] Option A: lightweight BM25 (rank_bm25 over candidates)
  - [ ] Option B: Elasticsearch/OpenSearch (containerized) — deferred to Backlog
- [x] Vector search
  - [x] Embedding model via config
  - [x] Build FAISS index → persist under models/index/*
  - [x] Script: src/retrieval/build_index.py to embed + index manifest chunks
- [x] Hybrid fusion
  - [x] Reciprocal Rank Fusion (RRF) or weighted score sum
  - [x] Configurable top_k and fusion weights
- [x] Filters
  - [x] Plant/disease metadata filters
  - [x] Deterministic seeds not needed (pure retrieval)
- [x] Evaluation scaffolding
  - [x] Mini labeled set: data/kb/labels.jsonl
  - [x] Script: src/retrieval/evaluate.py → Recall@k, nDCG@k
  - [x] Tests: tests/test_retrieval.py
- [x] Make targets
  - [x] make index (embed + build index)
  - [x] make eval_retrieval (run evaluate.py)

**Acceptance checks**
- [x] Hybrid retrieval measurable & repeatable; metrics saved to `artifacts/`  <!-- artifacts/retrieval_eval/retrieval_eval.json -->

---

## Milestone 3 — LLM Integration (RAG) — Done
Goal: Grounded answers with citations.

- [x] RAG pipeline: `src/llm/rag_pipeline.py`
- [x] Steps: retrieve → select → prompt‑compose → call LLM → postprocess
- [x] Enforce citations (postprocess verifies/ensures [n] within range)
- [x] Refusal/guardrails for empty context (no hits)
  - Early return with a polite refusal + one fallback retrieval without plant filter
  - Test: [`tests/test_guardrails.py`](tests/test_guardrails.py)
- [x] Prompt templates: `src/llm/prompts/`
  - [x] Answer template with explicit citation instructions (`src/llm/prompts/answer.txt`)
  - [x] Tests: required placeholders exist; no missing keys (`tests/test_prompts.py`)
- [x] LLM backends/robustness
  - [x] Temperature and timeout in generation calls
  - [x] Retry with simple exponential backoff
  - [ ] Token budgeting (defer to improvements)
- [x] Tests
  - [x] Unit: mocked LLM (`tests/test_rag_pipeline.py`)
  - [x] Guardrail: refuse on no-context (`tests/test_guardrails.py`)
  - [x] Citation enforcement (`tests/test_citations.py`)
  - [x] Integration smoke (OpenAI) when key is set

**Acceptance checks**
- [x] End‑to‑end: question → grounded answer with citations within target latency

---

## Milestone 4 — RAG Evaluation & Cloud Deployment

Goals
- Generate a focused QA dataset from the KB manifest.
- Run RAG end-to-end and score answers with LLM-as-judge (faithfulness, relevance).
- Deploy a minimal working app to Hugging Face Spaces or similar cloud platform.
- Produce JSON/CSV artifacts for inspection.

What’s done / Next steps
- [x] Dataset created: data/eval/rag_qa.jsonl (120 prompts; from data/kb/manifest.parquet)
- [x] Evaluation run
  - [x] Command: python -m src.eval.evaluate_rag --dataset data\eval\rag_qa.jsonl --out artifacts\rag_eval --n 120 --skip-if-no-key
  - [x] Judge model: Default gpt-4o-mini (override via OPENAI_MODEL)
  - [x] Artifacts: artifacts/rag_eval/rag_eval.json and artifacts/rag_eval/rag_eval.csv
- [x] Ergonomics: Judge max_tokens capped; optional progress/timeouts
- [x] Deploy minimal app to Hugging Face Space (Streamlit or FastAPI)

Queued improvements (tracked for M5.x)
- [ ] Retrieval recall: increase top_k to 4–5; raise ctx_chars to 1600–2000
- [ ] Add reranker (e.g., bge-reranker) over top-20 to select 4–5 best chunks
- [ ] Tighten prompt: “Use only context; if unknown, say so; cite [n]”
- [ ] Temperature 0; optionally drop sentences without citations in postprocess
- [ ] Judge visibility: --save-context flag and partial writes (CSV flush, partial JSON)

---

## Milestone 5 — Minimal UI/API

What’s done / Next steps
- [x] Streamlit UI
  - [x] Classifier top-1 auto-fills detected_plant/detected_disease
  - [x] Query enrichment and filters passed to RAG
  - [x] Default question fallback (“What can I do to treat this?”)
  - [x] Sources panel with titles/URLs; Windows Make target
- [x] Tiny sample KB (2 plants, 3 diseases) under data/sample_kb/ + index build
- [x] Docs: docs/STREAMLIT.md and README links
- [x] Pipeline fixes
  - [x] Retrieval hits normalized to dicts with meta (no tuple .get errors)
  - [x] Plant/disease filters applied in retrieval/prompt
  - [x] Streamlit auto-clears stale answers when a new image is uploaded; stale content moves to a collapsible panel
  - [x] Classifier label normalization for "healthy" and squash/raspberry aliases

Next steps (M5.x)
- [ ] Retrieval: add reranker; expose fusion/alpha in UI
- [ ] Prompting: enforce citations and stricter grounding
- [ ] API: FastAPI /rag endpoint with env-configurable index/top_k
- [ ] Feedback: thumbs up/down logging to data/feedback
- [ ] Update Hugging Face Space/app repo (sync Streamlit app, README, and secrets)

---

## Milestone 6 — Capstone Packaging (LLM Zoomcamp)
Goal: Meet rubric; easy to run, understand, and evaluate.

- [ ] Documentation
  - [ ] README: problem, users, how‑to‑run, evaluation, design decisions, limitations
  - [ ] Architecture doc: `docs/architecture.md` + `docs/architecture.png`
  - [ ] Data card + licensing notes
- [ ] Reproducibility
  - [ ] Make targets; pinned versions; seeds; `.env.example`
  - [ ] Minimal sample KB for quick demos
- [ ] CI
  - [ ] GitHub Actions: ruff/flake8 + pytest (+ optional mypy)
  - [ ] Cache embeddings between runs if feasible
- [ ] Demo
  - [ ] 2–5 min video (ingestion → retrieval → RAG → UI)
  - [ ] Optional hosted demo (HF Space/small VM)
- [ ] Submission checklist aligned with course rubric

**Acceptance checks**
- [ ] Reviewer can clone, run, reproduce key metrics & watch the demo

---

## Submission Checklist (LLM Zoomcamp Assessment)

- [x] Knowledge base and LLM integrated (RAG pipeline)
- [x] Multiple retrieval approaches evaluated (BM25, FAISS, hybrid)
- [x] LLM evaluation with judge model and prompt variants
- [x] Streamlit UI and FastAPI API
- [x] Automated ingestion pipeline (Python scripts)
- [x] User feedback logging and dashboard (Streamlit charts)
- [x] Docker-compose for reproducible deployment
- [x] Clear instructions, pinned dependencies, sample KB
- [x] Hybrid search and document re-ranking (evaluated or planned)
- [x] User query rewriting (documented or planned)
- [x] Cloud deployment (Hugging Face Space)

---

## Cross‑Cutting Improvements / Hardening
(Implement alongside milestones when convenient.)

- [ ] Centralized config (`src/common/config.py` using Pydantic Settings)
  - [ ] Paths, model names, k‑values, timeouts, feature flags
  - [ ] `.env.example` validation (fail fast on missing keys)
- [ ] Index persistence
  - [ ] Persist FAISS/Qdrant to `models/index/` with versioned filenames
  - [ ] `make rebuild_index` to force re‑embed
- [ ] Data governance
  - [ ] Keep `source_attribution` in manifest; respect licenses
  - [ ] Document dedup/cleaning in data card
- [ ] Prompt/versioning
  - [ ] Version prompts; log prompt IDs with outputs
- [ ] Telemetry
  - [ ] CSV logs by default; optional Langfuse if configured
- [ ] Testing
  - [ ] Unit tests for ingestion, retrieval, prompts, RAG pipeline
  - [ ] CI smoke test: 3 canonical queries (skip external calls if keys absent)
- [ ] CI/CD
  - [ ] PR checks: lint + tests (block on red)
  - [ ] Optional: Docker build + push on `main`
- [ ] Security & safety
  - [ ] Input size limits; URL sanitization
  - [ ] Refuse medical diagnosis; always cite sources
- [ ] Performance
  - [ ] Caching for embeddings & retrieval responses
  - [ ] Batch embedding; chunk‑level cache keys

---

## Backlog / Stretch Goals (Optional)
- [ ] Train a lightweight cross‑encoder reranker on labeled set
- [ ] Multilingual query support (FR/AR/DE)
- [ ] Incremental ingestion (delta crawl) + re‑embed only changed chunks
- [ ] Active‑learning loop using user feedback to improve retrieval
- [ ] Deploy a tiny demo KB to Hugging Face Spaces

---

## Suggested GitHub Issues (titles you can paste)
- [ ] Milestone 1: Build KB ingestion pipeline
- [ ] Milestone 2: Implement hybrid retrieval (BM25 + FAISS) with RRF
- [ ] Milestone 2: Create mini labeled set and retrieval evaluator
- [ ] Milestone 3: Implement RAG pipeline with strict citations
- [ ] Milestone 3: Add prompt templates + tests
- [ ] Milestone 4: RAG evaluation (LLM‑as‑judge) + artifacts
- [ ] Milestone 4: Add tracing/logging + feedback sink
- [ ] Milestone 5: Streamlit UI + feedback widget
- [ ] Milestone 5: FastAPI endpoints (/search, /rag) + schemas
- [ ] Milestone 6: Documentation & demo video
- [ ] Hardening: Centralized config with Pydantic
- [ ] Hardening: Persisted vector index + Make targets
- [ ] Hardening: CI (ruff/pytest) + smoke test workflow

---

## Technical Debt (tracked items)
- [x] Logo and branding: Standardize logo filename to `plant-disease-rag-assistant-logo.png` and update all references. Remove obsolete logo.
- [x] Remove Gradio app from active development; mark as obsolete in code comments and README.
- [x] Dockerfile: Use local wheels for faster builds and avoid repeated downloads of large packages (e.g., torch).
- [ ] KB coverage gaps: Some diseases (e.g., maize treatments) lack management/treatment info in the KB. Update KB sources and index to ensure all tracked diseases have actionable context.
- [ ] .dockerignore excludes KB/index files by default; review ignore rules to avoid missing files in Docker builds.
- [ ] Ensure all environment variables (e.g., INDEX_DIR) are consistent across local/dev/prod and documented in .env.example and README.
- [ ] Add automated checks for KB completeness (e.g., missing treatment sections).
- [ ] Improve error handling for missing context in RAG pipeline.

---

## Milestone: KB Expansion & Multi-source Integration

**Goal:**  
Build a comprehensive, actionable plant disease knowledge base by integrating multiple trusted sources and ensuring every (crop, disease) entry includes symptoms, cause, management, prevention, and references.

---

### Tasks & Progress Tracking

#### 1. Multi-source Scraping & Integration
- [x] PlantVillage: Continue scraping for symptoms, cause, and management. Identify gaps for less common crops/diseases.
- [ ] Wikipedia: Scrape disease pages for additional context, especially management/treatment sections.
- [ ] University Extension Services: Ingest management recommendations from Penn State, UC Davis, Cornell, Purdue, etc.
- [ ] FAO & Government Agriculture Portals: Integrate best practices for disease management.
- [ ] Peer-reviewed Literature: Semi-automate ingestion of abstracts from PubMed/AGRICOLA for up-to-date treatment strategies.

#### 2. Structured KB Design
- [x] Define schema: Each (crop, disease) entry must include `symptoms`, `cause`, `management`, `prevention`, `references`.
- [x] Support JSON/Parquet formats for KB storage.

#### 3. Automated Completeness Checks
- [ ] Write scripts to flag missing/weak entries (e.g., empty or short management fields).
- [ ] Prioritize manual review or targeted scraping for flagged gaps.

#### 4. Human-in-the-loop Curation
- [ ] Expert/crowdsourced review for critical crops/diseases.
- [ ] Add citations and ensure advice is regionally relevant.

#### 5. Continuous Updates
- [ ] Set up periodic scraping/ingestion jobs to keep KB fresh.
- [ ] Track new outbreaks or emerging diseases.

---

**Summary Plan**
- Start with PlantVillage, but supplement with Wikipedia, university extensions, FAO, and scientific literature.
- Require actionable management info for every KB entry.
- Automate gap detection and prioritize filling those gaps.
- Cite sources and keep the KB updated.

---

**Progress:**  
- [x] PlantVillage scraping and schema defined  
- [ ] Wikipedia, university, FAO, literature integration  
- [ ] Completeness checks and human review  
- [ ] Continuous update pipeline

*Track progress by updating checkboxes as tasks are completed.*
