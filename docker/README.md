# Running the Quiz Generator with Docker

This is a packaged version of the quiz generator — you don't need to install
Python, Ollama, or any of the libraries on your own machine. Docker handles
everything.

## What you need on your machine

- Docker Desktop (Mac/Windows) or Docker Engine + docker-compose plugin (Linux)
- ~10 GB of free disk space (mostly for the Qwen 2.5 7B model)
- ~8 GB of RAM available to Docker (Qwen needs it to run inference)

That's it. No Python, no Ollama, nothing else.

## First-time setup (about 15 minutes, mostly downloads)

```bash
# 1. Get the code (your team will tell you the GitLab URL)
git clone <repo-url>
cd quiz-generator

# 2. Start both containers in the background
docker compose up -d

# 3. Pull the LLM model into the Ollama container (one time only, ~5 GB)
docker compose exec ollama ollama pull qwen2.5:7b

# 4. Build the vector index (one time only, ~10 minutes)
docker compose run --rm api python -m src.indexing.build
```

After step 4, the system is fully ready.

## Daily usage

```bash
docker compose up -d          # start (fast — everything is already cached)
docker compose down           # stop (data and models are kept)
docker compose logs -f api    # tail the API logs
```

## What's running and where

| Service | What it does | Port |
|---------|--------------|------|
| `api`   | FastAPI app — accepts HTTP requests | `8000` |
| `ollama`| Runs Qwen 2.5 7B for the LLM calls   | `11434`|

The API is reachable at **`http://localhost:8000`**.

Useful URLs:
- `http://localhost:8000/docs` — interactive Swagger UI (try requests in browser)
- `http://localhost:8000/health` — liveness check
- `http://localhost:8000/taxonomy` — list of supported subjects/levels/languages

## Smoke test (~30 seconds)

After `docker compose up -d` completes, wait ~60 seconds for the API to
finish loading the embedder + reranker, then:

```bash
curl http://localhost:8000/health
# → {"status":"ok","pipeline_loaded":true}

curl -X POST http://localhost:8000/quiz/generate \
  -H "Content-Type: application/json" \
  -d '{"topic":"primitives des fonctions","language":"fr","subject":"MATHEMATICS","count":3}'
```

The first call takes ~10–15 seconds (Qwen is generating). Subsequent calls
to similar topics are similar.

## Where data lives

These folders on your machine are mounted into the API container:

| Host path                 | Container path           | What it is              |
|---------------------------|--------------------------|-------------------------|
| `./data/vector_store/`    | `/app/data/vector_store` | Chroma vector database  |
| `./data/processed/`       | `/app/data/processed`    | Cleaned quiz JSONL      |
| `./configs/`              | `/app/configs`           | YAML config             |

So if you change `configs/models.yaml`, restart the api container with
`docker compose restart api` and the new value takes effect.

Ollama models and the BGE-M3 / reranker downloads live in **Docker
named volumes** (`ollama_data` and `hf_cache`). They survive
`docker compose down` and stop bloating your project folder.

## Troubleshooting

**`api` container keeps restarting**
- Check logs: `docker compose logs api`
- Most common: Qwen wasn't pulled. Run step 3 of first-time setup.

**Slow first request (60–90s)**
- Normal. The API is downloading BGE-M3 + the reranker on first use.
  After that they're cached in the `hf_cache` volume.

**`ConnectionError` calling /quiz/generate**
- Likely Ollama isn't ready yet. Wait a minute, or check
  `docker compose ps` — both services should show `healthy`.

**Out of memory**
- Qwen 2.5 7B needs ~5 GB RAM during inference. If your machine is tight,
  bump Docker Desktop's RAM limit to 8 GB+ in settings.

## Stopping vs removing

```bash
docker compose down            # stop containers, keep data + models
docker compose down -v         # ALSO delete named volumes (Qwen, HF cache)
                               # — only do this when you really want to start over
```

`docker compose down` is the normal "I'm done for the day." Don't use `-v`
unless you want to redownload ~6 GB of models on the next start.
