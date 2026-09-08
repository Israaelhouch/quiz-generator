# CLAUDE.md — Engineering standards for this repository

> Drop this file at the repo root. Fill the `<...>` fields in §0. Everything else applies unchanged to a fresh project or a refactor. When an instruction here conflicts with a request, follow this file and say so.

---

## 0. Project context (fill in)

- **Name / purpose:** `A retrieval-augmented generation service that writes new exam questions in
**English, French and Arabic** across four subjects, grounded in a corpus of
real curriculum questions. Built as a production service: FastAPI, a Chroma
vector store, a cross-encoder reranker, and a measured retrieval eval.`
- **Status:** `<greenfield | refactor of existing code | maintenance>`
- **Stack:** Python 3.12 · FastAPI · Pydantic v2 · `<vector DB / DB>` · `<ML libs>` · Docker Compose · pytest · GitLab CI (adapt if different)
- **Entry points:** `make setup` · `make test` · `make run` · `make lint`
- **Key metrics this project must report:** `<e.g. Recall@10, MRR, p95 latency, cost/1k requests>`
- **Out of scope:** `<what this repo deliberately does not do>`

---

## 1. How to work in this repo

1. **Understand before changing.** Read the relevant module, its tests, and `docs/DECISIONS.md` first. Summarise the current behaviour in two sentences before proposing edits.
2. **Plan, then act.** For anything beyond a one-line fix: state the plan (files touched, approach, risks, how it will be verified), get agreement, then implement. Small, reviewable diffs over large rewrites.
3. **Never guess an API or a value.** If a library signature, config key, or data shape is uncertain, look it up in the code or docs, or ask. Say "I'm not sure" rather than inventing.
4. **Every change ships with its test and its docs.** No exceptions for "small" changes.
5. **Leave the codebase better than found**, but do not refactor beyond the task's scope without saying so first. Opportunistic cleanups go in a separate commit or PR.
6. **Do not delete, move, or rename data, migrations, or environment files** without explicit confirmation.
7. **Secrets never enter the repo, logs, or chat.** Use `settings` (Pydantic Settings) and `.env.example`.

---

## 2. Architecture

**Layout (src layout, one package):**
```
src/<package>/
  api/          # FastAPI routers, request/response schemas only — no business logic
  core/         # config (Settings), logging, errors, constants
  domain/       # business logic, pure Python, no framework imports
  services/     # orchestration: combine domain + adapters
  adapters/     # I/O: DB, vector store, LLM clients, HTTP, filesystem
  pipelines/    # ingestion / training / evaluation jobs (CLI entry points)
  eval/         # metrics, test sets, evaluation harness
tests/
  unit/  integration/  e2e/  fixtures/
docs/
  ARCHITECTURE.md  DECISIONS.md  RUNBOOK.md
notebooks/        # exploration only; never imported by src
```

**Rules**
- Dependencies point inward: `api → services → domain ← adapters`. `domain` imports nothing from `api`, `adapters`, or third-party frameworks.
- Every external system (DB, vector store, LLM, HTTP) is behind an interface in `adapters/` with a fake implementation for tests.
- Configuration is typed (`Settings` class), validated at startup, and injected — never read `os.environ` inside business code.
- One module = one responsibility; a file over ~400 lines or a function over ~40 lines is a signal to split.
- Public functions have type hints and a one-line docstring stating intent, not implementation.
- Errors: raise domain-specific exceptions in `domain/`; translate to HTTP errors only in `api/`. Never swallow exceptions; never `except Exception: pass`.
- Logging: structured (JSON), one logger per module, correlation ID on every request. Log events, not prose. Never log PII, tokens, or raw prompts containing user data.

**Record every non-trivial choice in `docs/DECISIONS.md`** as an ADR: context → options → decision → consequences. Examples: chunking strategy, embedder choice, reranker pool size, DB, retry policy.

---

## 3. Code style

- Formatter/linter: `ruff` (format + lint), `mypy --strict` on `src/`. CI fails on any violation.
- Naming: descriptive, no abbreviations except domain-standard ones (`mrr`, `topk`). Booleans read as predicates (`is_ready`, `has_index`).
- Prefer pure functions and explicit data classes / Pydantic models over dicts passed around.
- No clever one-liners, no commented-out code, no `TODO` without an issue reference.
- Idempotent pipelines: re-running a job with the same inputs yields the same outputs and no duplicates.
- Concurrency: no shared mutable state across requests; anything cached per-request lives in request scope. Add a concurrency regression test for any code path that touches shared resources.
- Async only where I/O-bound and measured; do not mix sync DB clients into async handlers.

---

## 4. Testing

- **Pyramid:** unit (fast, no I/O, fakes for adapters) → integration (real containers via Docker Compose or testcontainers) → e2e (one happy path per endpoint).
- **Coverage is a floor, not a goal:** ≥ 80% on `domain/` and `services/`; every bug fix adds a test that fails before the fix.
- **Property and edge cases:** empty inputs, unicode/RTL text (Arabic), very long inputs, duplicate keys, timeouts, partial failures.
- **ML/RAG specifics:** a frozen evaluation set with ground-truth IDs lives in `eval/`; `make eval` reports the project's key metrics and writes a JSON snapshot with config hash and git SHA. CI fails if a metric regresses beyond a stated tolerance.
- **Determinism:** seed everything; pin model versions; record them in the eval snapshot.
- Tests are named `test_<unit>_<condition>_<expected>` and read as specifications.

---

## 5. Debugging protocol

When something fails, do not patch symptoms. Follow this order and write the result in the PR description:

1. **Reproduce** minimally (a failing test or a one-command script). If it can't be reproduced, instrument first.
2. **Localise** with logs, correlation IDs, and bisecting (git bisect, feature flags, binary-search the data).
3. **Hypothesise → verify**: state the hypothesis, the experiment, the expected observation; run it; record what actually happened.
4. **Fix the cause**, not the site. If the fix is a workaround, label it and open an issue for the real fix.
5. **Regression test** that fails on the old code.
6. **Post-mortem line** in `docs/DECISIONS.md` or the PR: what broke, why it wasn't caught, what now catches it.

Never claim a fix works without showing the passing test or the reproduced-then-resolved output.

---

## 6. Git workflow

**Branches:** `main` is always deployable. Work on `feat/<scope>-<short>`, `fix/<scope>-<short>`, `chore/…`, `docs/…`. One branch = one logical change.

**Commits (Conventional Commits):**
```
<type>(<scope>): <imperative summary ≤ 72 chars>

<why this change, not what — the diff shows what>
<metric or behaviour impact if any>

Refs: #<issue>
```
Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`. Scope = module (`retrieval`, `api`, `eval`, `ingest`).
- Atomic: each commit builds and passes tests on its own. Squash WIP before pushing.
- Never commit secrets, data, models, notebooks with outputs, or generated files. Use `.gitignore` and `.gitattributes` (LFS only if agreed).
- Rebase on `main` before opening a PR; no merge commits from `main` into feature branches.

**Pull requests** — template in `.gitlab/merge_request_templates/default.md` (or `.github/`):
```
## What
## Why
## How verified   (commands run + output summary; eval metrics before/after if relevant)
## Risk / rollback
## Checklist
- [ ] tests added/updated   - [ ] docs/ADR updated   - [ ] lint + mypy clean
- [ ] no secrets / PII      - [ ] CHANGELOG entry     - [ ] screenshots/metrics attached
```
PRs are small (< ~400 lines changed); larger work is split into a stack.

**Releases:** SemVer tags (`v1.2.0`), `CHANGELOG.md` (Keep a Changelog format), a release note that states the metric impact.

**Repo hygiene:** `README.md` (problem → architecture diagram → metrics table → 3-command quickstart → design decisions → limitations), `LICENSE`, `CONTRIBUTING.md`, issue templates, CI badge, protected `main`.

---

## 7. Data, ML and LLM-specific rules

- Data contracts: every ingestion stage validates its input schema (Pydantic / Pandera / Great Expectations) and rejects rather than coerces silently. Count and log rejects.
- Chunking, embedding, and reranking parameters are config, not literals, and are recorded in the eval snapshot.
- Prompts are versioned files under `src/<package>/prompts/`, never inline strings; changes to a prompt are a `feat` or `fix` with an eval run attached.
- LLM calls: timeouts, retries with backoff, cost and token accounting, tracing (Langfuse/OpenTelemetry) on by default in non-test environments.
- Model artefacts and datasets live outside git with a manifest (`name`, `version`, `sha256`, `source`, `licence`).
- Evaluation is a first-class deliverable: retrieval metrics (Recall@k, MRR, P@k) and generation quality (faithfulness, relevance) reported per run, per language when multilingual.
- Guardrails: input validation, output schema enforcement, PII redaction in logs, rate limits on public endpoints, API-key auth by default.

---

## 8. Security and privacy

- Threat-model each public endpoint in one paragraph in `docs/ARCHITECTURE.md`.
- Dependencies pinned (`uv`/`pip-tools` lock), scanned in CI (`pip-audit`), Docker images non-root with pinned base tags.
- Secrets via environment or a secrets manager; `gitleaks` runs in pre-commit and CI.
- Personal data (names, emails, student IDs) never appears in logs, fixtures, or examples — use synthetic data.

---

## 9. Definition of Done

A task is done only when all of these are true:
- Code merged to `main` through a reviewed PR; CI green (lint, types, tests, eval gate, security scan).
- Tests cover the change; a regression test exists for any bug.
- `README`/`docs`/ADR/`CHANGELOG` updated; metrics table refreshed if behaviour changed.
- `make setup && make test && make run` works from a clean clone.
- No secrets, PII, or generated artefacts committed.
- The PR description explains why, how it was verified, and how to roll back.

---

## 10. Refactoring an existing codebase (use when §0 says "refactor")

Work in this order; each step is its own PR:
1. **Characterisation tests** for current behaviour before touching anything.
2. **Safety net:** CI, lint, type-check (allow-list existing violations, ratchet down over time), secrets scan, `.env.example`.
3. **Structure:** move to src layout and the §2 folders without changing behaviour.
4. **Boundaries:** extract adapters behind interfaces; inject settings.
5. **Quality:** raise type coverage and test coverage module by module; delete dead code with evidence it's dead.
6. **Docs:** README template, architecture diagram, ADRs for the decisions you discovered.
7. **Release:** tag `v1.0.0` when Definition of Done holds for the whole repo.

Never mix a behaviour change with a structural change in the same commit.

---

## 11. Communication

- Be explicit about uncertainty and trade-offs; offer at most two options with a recommendation.
- When declining or deviating from a request because of this file, cite the section.
- Reports end with: what changed, how it was verified, what remains, and any risk.
