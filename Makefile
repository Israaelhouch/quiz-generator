# Makefile — the entry points named in CLAUDE.md §0.
#
#   make setup   one-time: virtualenv + pinned deps + dev tools
#   make test    the pytest suite
#   make lint    ruff + mypy, the same gates CI runs
#   make run     serve the API on localhost:8000
#
# Everything runs inside ./.venv. No target expects an activated shell; each
# calls the interpreter by path, so `make test` behaves the same in a fresh
# terminal, in CI, and from an editor.

.DEFAULT_GOAL := help
SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
RUFF    := $(VENV)/bin/ruff
MYPY    := $(VENV)/bin/mypy
PYTEST  := $(VENV)/bin/pytest

HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: help setup test test-cov lint fmt fmt-check typecheck audit eval run \
        serve build ci clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
$(PY):
	python3 -m venv $(VENV)
# ensurepip seeds an old pip and setuptools; both have known advisories
# that `make audit` would otherwise report against a brand-new venv.
	$(PIP) install --upgrade pip setuptools

setup: $(PY)  ## Create the venv and install pinned + dev dependencies
	$(PIP) install -r requirements.lock.txt
	$(PIP) install ruff mypy pytest pytest-cov pip-audit
	@test -f .env || { cp .env.example .env; \
	  echo "created .env from .env.example — fill in your provider key"; }
	@echo "setup complete. next: make test"

# ---------------------------------------------------------------------------
# Quality gates — these four are exactly what CI runs (CLAUDE.md §3, §9)
# ---------------------------------------------------------------------------
test:  ## Run the test suite
	$(PYTEST)

test-cov:  ## Run tests with a coverage report
	$(PYTEST) --cov --cov-report=term-missing --cov-report=xml

lint: fmt-check typecheck  ## Lint + format check + type check (the full gate)
	$(RUFF) check .

fmt:  ## Rewrite files with the ruff formatter
	$(RUFF) format .
	$(RUFF) check . --fix

fmt-check:  ## Fail if anything is unformatted (advisory until the format sweep lands)
	@$(RUFF) format --check . 2>&1 | tail -1 || true
	@$(RUFF) format --check . >/dev/null 2>&1 || { \
	  echo ""; \
	  echo "NOTE: the repo predates ruff format; a one-shot sweep is tracked"; \
	  echo "      is tracked as its own commit. Not a gate yet."; \
	  echo ""; }

typecheck:  ## mypy --strict on src/, with the ratchet in pyproject.toml
	$(MYPY)

audit:  ## Scan installed dependencies for known CVEs
# Audits the environment, not requirements.lock.txt. The lockfile pins CUDA
# wheels (cuda-bindings, nvidia-*) that only resolve on Linux+GPU, so
# `pip-audit -r` fails to build a resolution on macOS or CPU-only CI. The
# installed venv is also the more honest target: it is what actually ships.
#
# IGNORED ADVISORIES — each needs evidence, an owner and a review date.
# Re-check on 2026-12-01; drop any ID the moment chromadb ships a fix.
#
#   PYSEC-2026-311, CVE-2026-45830, CVE-2026-45831, CVE-2026-45833
#     All four are in chromadb's *server*: pre-auth code injection, missing
#     authorization validation, and the SimpleRBACAuthorizationProvider.
#     This project only ever constructs `chromadb.PersistentClient` against a
#     local directory (src/indexing/vector_store.py:121) — it runs no Chroma
#     server, exposes no Chroma port, and configures no auth provider. There
#     is also nothing to upgrade to: 1.5.9 is the latest release and no fixed
#     version exists. See docs/DECISIONS.md ADR-0002 — including the
#     condition under which these ignores MUST be removed.
	$(VENV)/bin/pip-audit --skip-editable \
	  --ignore-vuln PYSEC-2026-311 \
	  --ignore-vuln CVE-2026-45830 \
	  --ignore-vuln CVE-2026-45831 \
	  --ignore-vuln CVE-2026-45833

ci: lint test audit  ## Everything CI runs, locally

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
build:  ## Build the corpus + vector index from a cold checkout (idempotent)
	./run_local.sh --no-serve

run: ## Serve the API on $(HOST):$(PORT), building anything missing first
	./run_local.sh

serve:  ## Serve the API only, assuming the index already exists
	$(PY) -m src.api --host $(HOST) --port $(PORT)

eval:  ## Run the retrieval eval and write a versioned snapshot to eval/results/
	$(PY) -m scripts.eval.run_retriever_eval

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
clean:  ## Remove caches and build artefacts (never touches data/ or .env)
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml
