# Quiz Generator — Multilingual RAG for Tunisian Curriculum

A Retrieval-Augmented Generation (RAG) system that produces educational quiz
questions in **English, French, and Arabic**, grounded in an existing corpus
of Tunisian-curriculum questions. Designed to be called by a school
platform's backend so teachers can generate fresh, on-topic quizzes on
demand.

## What it does

Given a topic (free text), a target language, and optional filters
(subject, level, question type), the system:

1. Embeds the topic with **BGE-M3** (a multilingual embedder).
2. Retrieves similar existing questions from a **Chroma** vector store.
3. Reranks them with a **cross-encoder** (BGE-reranker-v2-m3) for higher
   precision.
4. Feeds the top examples to **Qwen 2.5** (running locally via Ollama)
   as few-shot prompts.
5. Validates and parses the LLM output into typed Pydantic models, with
   automatic retry on validation failure.
6. Returns a structured quiz: a list of questions with choices, correct
   answers, and explanations.

## Architecture

```
   ┌──────────────────┐         ┌──────────────────┐
   │  Stage 1–2       │         │  Stage 3         │
   │  Cleaning &      │ ───────►│  Indexing        │
   │  Normalization   │         │  (BGE-M3 →       │
   │                  │         │   Chroma)        │
   └──────────────────┘         └────────┬─────────┘
                                         │
                                         ▼
                                ┌──────────────────┐
                                │  Stage 4         │
                                │  Retrieval +     │
                                │  Reranker        │
                                └────────┬─────────┘
                                         │
                                         ▼
                                ┌──────────────────┐
                                │  Stage 5         │
                                │  Generation      │
                                │  (Qwen via       │
                                │   Ollama)        │
                                └────────┬─────────┘
                                         │
                                         ▼
                                ┌──────────────────┐
                                │  Stage 6         │
                                │  QuizPipeline    │
                                │  (orchestrator)  │
                                └────────┬─────────┘
                                         │
                                         ▼
                                ┌──────────────────┐
                                │  HTTP API        │
                                │  (FastAPI)       │
                                └──────────────────┘
```

## Quick start

### Option A — Docker (recommended for evaluators)

Single command starts the API + Ollama. See **[`docker/README.md`](docker/README.md)**
for detailed instructions. Short version:

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:3b      # one-time, ~2 GB
docker compose run --rm api python -m src.indexing.build  # one-time, ~10 min

# API now live at http://localhost:8000
open http://localhost:8000/docs                         # interactive Swagger UI
```

### Option B — Native (for active development)

Requires Python 3.11+, [Ollama](https://ollama.ai), and ~10 GB free disk.

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Install LLM via Ollama (in a separate terminal)
ollama serve &
ollama pull qwen2.5:3b

# 3. Build cleaned data + vector index (one time)
python -m src.data.ingest
python -m src.data.normalize
python -m src.indexing.build

# 4. Start the API
python -m src.api
```

## Using the API

Once the server is running on `http://localhost:8000`:

```bash
# Health check
curl http://localhost:8000/health

# List supported subjects / levels / languages (for UI dropdowns)
curl http://localhost:8000/taxonomy | jq

# Retrieval only (no LLM call) — useful for debugging
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"primitives","language":"fr","subject":"MATHEMATICS","top_k":5}' | jq

# Full generation
curl -X POST http://localhost:8000/quiz/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topic":"primitives des fonctions",
    "language":"fr",
    "subject":"MATHEMATICS",
    "count":3,
    "include_retrieval":true
  }' | jq
```

The interactive Swagger UI at `http://localhost:8000/docs` exposes every
endpoint with a "Try it out" button — no curl needed for exploration.

### What the API exposes

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness probe |
| `/taxonomy` | GET | Legal subjects/levels/languages |
| `/retrieve` | POST | Retrieval only, no LLM |
| `/quiz/generate` | POST | Full retrieve → generate → validate |

Tuning parameters (`temperature`, `few_shot_count`, `max_attempts`) are
**not** exposed in the API — they live in `configs/models.yaml` and are set
once by the AI engineer.

## CLI usage (for engineering work)

The same pipeline is also available as command-line tools:

```bash
# Generate a quiz directly (without the HTTP layer)
python -m src.pipeline "primitives des fonctions" --language fr \
    --subject MATHEMATICS --count 3 --show-retrieval --save-run

# Inspect what retrieval pulls (no LLM)
python -m src.retrieval.query "primitives" --language fr --top-k 5

# A/B test the reranker
python -m src.retrieval.compare_rerank "primitives" --language fr --top-k 5
```

`--save-run` writes a debugging artifact to `./last_run.json` containing
both the retrieved chunks and the generated quiz — useful for inspection.

## Project layout

```
quiz-generator/
├── configs/
│   ├── models.yaml          # embedder, reranker, LLM, indexing config
│   ├── pipeline.yaml        # search_text composition recipes
│   └── subject_aliases.yaml # subject canonicalization rules
├── data/
│   ├── raw/                 # source data (read-only)
│   ├── interim/             # cleaning outputs
│   ├── processed/           # ready.jsonl (final cleaned corpus)
│   └── vector_store/        # Chroma DB (gitignored)
├── docker/                  # Docker docs
├── docs/                    # design notes
├── notebooks/               # Stage 1–3 exploration notebooks
├── scripts/                 # one-off diagnostics
├── src/
│   ├── api/                 # FastAPI HTTP surface
│   ├── data/                # Stage 1–2: ingestion + cleaning
│   ├── generation/          # Stage 5: prompts + LLM client + retry
│   ├── indexing/            # Stage 3: build vector store
│   ├── pipeline/            # Stage 6: end-to-end orchestrator + CLI
│   ├── retrieval/           # Stage 4: retriever + reranker
│   └── shared/              # cross-cutting schemas + utilities
├── tests/                   # pytest test suite (uses mocks; no Ollama needed)
├── Dockerfile               # API container image
├── docker-compose.yml       # API + Ollama stack
├── requirements.txt
└── README.md
```

## Configuration

All runtime knobs live in `configs/models.yaml`:

```yaml
model:
  name: BAAI/bge-m3                   # embedder
  embedding_dim: 1024

vector_store:
  type: chroma
  persist_directory: data/vector_store/chroma_db

reranker:
  enabled: true                       # filter-then-rerank pipeline
  model_name: BAAI/bge-reranker-v2-m3
  candidate_pool_size: 50

llm:
  provider: ollama
  model: qwen2.5:3b                   # or qwen2.5:7b for production quality
  default_temperature: 0.75
  max_attempts: 3                     # retries on validation failure
  default_few_shot_count: 5
```

Change a value, restart the API, and every endpoint and CLI invocation
picks up the new default.

## Tests

The test suite uses mocked LLMs and fake retrievers, so it runs without
Ollama or real model downloads:

```bash
python tests/test_api.py            # API endpoints
python tests/test_pipeline.py       # Stage 6 orchestrator
python tests/test_generation.py     # Stage 5 LLM + retry + validation
python tests/test_retriever.py      # Stage 4 retrieval
python tests/test_reranker.py       # Stage 4 cross-encoder
python tests/test_indexing.py       # Stage 3 Chroma + config
python tests/test_normalize.py      # Stage 2 cleaning
python tests/test_ingest.py         # Stage 2 raw load
python tests/test_latex.py          # LaTeX normalization
python tests/test_domain_rules.py   # Tunisian curriculum rules
```

Or all at once:

```bash
for t in tests/test_*.py; do python "$t" || break; done
```

## Status

**Working end-to-end:**
- Stage 1: Data scoping (`docs/01_scope.md`)
- Stage 2: Cleaning, deduplication, LaTeX normalization, language detection,
  domain rules
- Stage 3: BGE-M3 indexing with Chroma + boolean-per-level metadata for
  native pre-filtering
- Stage 4: Bi-encoder retrieval + cross-encoder reranking
  (`compare_rerank` CLI for evaluation)
- Stage 5: Qwen-based generation with multilingual prompts (en/fr/ar) +
  retry-on-validation-failure
- Stage 6: `QuizPipeline` orchestrator + CLI + HTTP API + Docker stack

**Not yet done (planned):**
- Authentication on the HTTP API (will be required before any external
  exposure beyond localhost)
- Stage 7: evaluation framework (gold set + retrieval/generation metrics)
- Regurgitation detection (cosine similarity check between generated and
  retrieved questions)
- Containerized deployment to a real server

## License

(Internal — to be defined)
