# Runbook

Pipeline stages, container commands and the evaluation harness.

## Standard path

```bash
make setup        # virtualenv + pinned dependencies + dev tools
make test         # 389 tests, no models, no keys, no network
make run          # builds anything missing, then serves on :8000/ui
```

`make run` wraps `run_local.sh`, which is idempotent: a stage is skipped if its
output already exists. `./run_local.sh --check` runs the preflight and changes
nothing; `--rebuild` forces the whole pipeline; `--no-serve` builds without
starting the API.

## Running the stages individually

Against the synthetic sample corpus in `data/sample/`:

```bash
git clone <this-repo> && cd quiz-generator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

# 1. ingest — scope filter + structural filters
python -m quiz_generator.ingestion.ingest --input data/sample/quizzes-sample-raw.json \
                                     --scope configs/scope.yaml \
                                     --output data/sample/interim/flat.jsonl \
                                     --stats  data/sample/interim/flat_stats.json

# 2. normalize — language resolution, HTML strip, aliases, curriculum rules
python -m quiz_generator.ingestion.normalize --input  data/sample/interim/flat.jsonl \
                                        --output data/sample/interim/normalized.jsonl \
                                        --stats  data/sample/interim/normalized_stats.json

# 3. build_index_text — compose the embedded text per configs/pipeline.yaml
python -m quiz_generator.ingestion.build_index_text --input  data/sample/interim/normalized.jsonl \
                                               --output data/sample/processed/payload.jsonl \
                                               --stats  data/sample/processed/payload_stats.json

# 4. index — BGE-M3 embeddings into Chroma (~1 min after the model downloads)
python -m quiz_generator.indexing.build --input data/sample/processed/payload.jsonl
```

Each stage writes a `*_stats.json` beside its output recording what it dropped
and why. The sample corpus contains intentionally malformed rows — duplicates,
a question with no correct answer, an image-only question, colliding `order`
values and a curriculum violation — so each stage has rows to reject.

The same four stages are `make ingest`, `make normalize`, `make build-text`,
`make build-index`, or `make build` for all four.

## Querying

Retrieval requires no provider key:

```bash
python -m quiz_generator.retrieval.query "past tense" --language en --top-k 3
```

Generation requires one:

```bash
echo "GEMINI_API_KEY=..." > .env
set -a; source .env; set +a
python -m quiz_generator.api        # then open http://localhost:8000/ui
```

`.env.example` documents all 14 variables with the defaults the code applies.
Each is a field on `src/quiz_generator/shared/settings.py`; a malformed value
stops the process at startup instead of falling back to a default.

## Quality gates

```bash
make test          # pytest
make lint          # ruff + mypy --strict
make ci            # everything CI runs, including gitleaks and pip-audit
```

Tests require no models, keys or network: every heavy adapter is mocked, so the
suite runs without torch or a GPU.

## Containers

See [`docker/README.md`](../docker/README.md). In short:

```bash
docker compose up --build            # API on :8000
docker compose -f docker-compose.gpu.yml up    # with GPU
docker compose -f docker-compose.etl.yml run --rm api \
  python -m quiz_generator.ingestion.ingest --scope configs/scope.yaml
```

The image builds from `requirements.lock.txt`, so a later rebuild resolves to
the same dependency versions.

## The eval harness

The inputs are derived from the private corpus and are not in the repository,
so these commands need `eval/*.json` and `eval/topics_*.csv` present.

```bash
make eval-validate     # check the answer key against the index — do this first
make eval              # the English template set
make eval-realistic    # the 50 realistic teacher questions
make refresh-topics    # rebuild answer keys from the index (dry run; WRITE=1 writes)
```

`make eval` writes a timestamped directory under `eval/results/` containing:

| File | What it holds |
|---|---|
| `summary.json` | The aggregated metrics |
| `per_query.jsonl` | One row per test case, for failure analysis |
| `config_snapshot.yaml` | The `models.yaml` that produced the numbers |
| `pipeline_snapshot.yaml` | The `search_text` recipe the index was built with |
| `provenance.json` | Git SHA, config hashes, and the index searched: payload hash, model, row count. Warnings list anything the run could not identify |

A run whose payload no longer matches the index it searched reports both
hashes. Measured results and their limits: [`eval/RESULTS.md`](../eval/RESULTS.md).
