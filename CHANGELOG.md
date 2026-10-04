# Changelog

Human-readable summary of what shipped per release. The git history has
the per-commit detail; this file has the per-release story.

---

## Unreleased

Four streams of work since v1.1.0: production hardening, an engineering UI with
feedback capture, a tooling safety net, and the repair of the English eval
answer key. The reasoning behind the non-obvious choices is in
`docs/DECISIONS.md`; the measured retrieval numbers are in `eval/RESULTS.md`.

### Added

- **API-key authentication** (`src/api/security.py`) — `X-API-Key` or
  `Authorization: Bearer`, constant-time comparison, keys from `API_KEYS`.
  Unset means auth is disabled, with a loud startup warning: fail-open so local
  development and the test suite keep working. `/health` and `/ready` stay open
  for container healthchecks.
- **Rate limiting** on `/retrieve` and `/quiz/generate` —
  `RATE_LIMIT_PER_MINUTE` (default 30) per API key or client IP over a rolling
  60s window, returning 429 + `Retry-After`. Counted per process.
- **Correlation IDs** — every response carries `X-Request-ID`, an inbound one is
  honoured, and the ID is written into `runs.jsonl`.
- **`GET /ready`** probes Chroma, payload load and the LLM client, 503 when
  degraded. `/health` only ever proved the process was alive.
- **`GET /metrics`** in Prometheus text format with no client library
  (`src/api/observability.py`): request counts by method/path/status, latency
  sum+count, counters for rate_limited / unauthorized / generation_failed.
- **`CORS_ALLOW_ORIGINS`**, unset by default — correct for a server-to-server
  caller; warns on `*`.
- **`/ui`** — one self-contained HTML console served from `src/api/static/`, no
  build step and no second container. The dropdowns encode the curriculum
  rules, so combinations the corpus cannot satisfy are unselectable instead of
  failing with a 400 fifteen seconds later. Arabic renders RTL, MathJax
  typesets LaTeX, and a print stylesheet shows every answer. Unauthenticated by
  design — a browser navigating to a URL cannot attach a header — and inert: no
  key is baked in, a 401 prompts and holds the key in memory for the session
  only, which a test asserts.
- **Retrieval visibility** — a debug panel showing the chunks the LLM actually
  saw, distance-coloured against the 0.60 floor, plus `timings` in the generate
  response when `include_retrieval=true`. This is what separates "the quiz is
  bad" from "the retriever fed it love poetry".
- **`POST /feedback`** — one human judgement per generated question (up/down
  plus an optional note) appended to `logs/feedback.jsonl`, carrying
  `request_id` rather than a copy of the retrieval. Authenticated but
  deliberately not rate-limited. `scripts/analyze_feedback.py` joins it back to
  `runs.jsonl` and compares mean worst-chunk distance for upvoted against
  downvoted questions; it refuses to interpret fewer than 5 judgements per
  verdict.
- **Tooling safety net** — `Makefile`, `pyproject.toml` (ruff, `mypy --strict`
  on `src/`, pytest, coverage), `.env.example` covering all 14 environment
  variables the code reads, `requirements-dev.txt`, GitHub Actions CI (ruff,
  mypy, pytest + coverage, gitleaks, pip-audit), `.pre-commit-config.yaml`, an
  MIT `LICENSE`, and `docs/DECISIONS.md`.
- **Eval harness** — `validate_test_cases` checks every answer key against the
  index (unknown ids, ids outside their cell, questions missing from their
  topic) and `run_retriever_eval` stops before loading the model if it fails;
  `scripts/eval/refresh_topics.py` rebuilds answer keys from the index, dry run
  by default, writing only rows whose ids changed and only if re-rendering the
  file unchanged reproduces it byte for byte; title aliases, with reviewed
  merges in a file that also records the rejected candidates and why; test
  cases may list several correct quizzes; a 50-question realistic English set,
  labelled without the search model; `make eval-realistic`,
  `make eval-validate`, `make refresh-topics`.
- `pandas` pinned in `requirements-dev.txt` — the eval scripts import it, so
  their tests failed in CI without it.

### Changed

- **The environment is read once, through a typed `Settings` object**
  (`src/shared/settings.py`). The 13 scattered `os.environ` reads across five
  modules are gone; every variable has a declared type, a declared default and
  startup validation, and the provider keys are `SecretStr` so they cannot
  reach a log line. A malformed value now stops the process instead of falling
  back to a default — `RATE_LIMIT_PER_MINUTE=abc` used to warn and apply 30. A
  test also fails if `.env.example` and the settings fields drift apart, which
  they had: two documented defaults did not match the code. `pydantic-settings`
  is now a declared dependency (ADR-0006).
- **Author PII removed from API responses and the run log.** `author_name` and
  `author_email` identify the real teachers who wrote the source corpus and
  were shipping on every `include_retrieval=true` response and every logged
  run. `INCLUDE_AUTHOR_METADATA=1` restores them.
- **Opaque error bodies.** 500s returned the exception string and 502s embedded
  corpus diagnostics; both now return a generic message plus the request ID,
  with full detail logged server-side. 400s still pass their message through —
  those are caller-caused and actionable.
- **LLM clients reuse their SDK client** and every call carries a timeout
  (`LLM_TIMEOUT_SECONDS`, default 90). Without one, a hung provider pinned a
  worker thread per retry attempt.
- **`logs/runs.jsonl` rotates** past `RUNS_LOG_MAX_BYTES` (default 50 MB),
  keeping 3 generations. It was unbounded.
- **Operational signals moved from `warnings.warn` to `logging`.** The warnings
  module dedupes per code location, so empty-retrieval, low-pool, multi-level
  and taxonomy signals fired once per process and were then silent forever.
- **Dependencies pinned** — `requirements.lock.txt` generated from the running
  image, and the Dockerfile builds from it. `requirements.txt` remains the
  statement of intent.
- `generate_detailed` declares the same `Literal` aliases as the Pydantic model
  it feeds instead of bare `str`; the type baseline shrank from 93 errors in 23
  files to 89 in 20.
- Applied ruff's 62 safe autofixes; every file is formatted and the check is a
  real gate in CI, `make lint` and pre-commit.
- `configs/phase1_scope.yaml` is now `configs/scope.yaml`: it has included
  MATHEMATICS since v1.1.0, and the old name described a project phase rather
  than the file. `configs/phase2_math_audit.yaml` is deleted — it scoped a
  one-off audit for a notebook that is not part of this repository.
- The README and `eval/RESULTS.md` report the repaired English numbers, add a
  Limits section and the realistic-question results, and revise the maths
  explanation: most of that gap is the test set's own ceiling.

### Fixed

- **Cross-request data leak.** `QuizPipeline.generate()` stashed the retrieval
  on `self.last_retrieval` and the endpoint read it after the call returned.
  One pipeline instance serves every request from the thread pool, so a
  concurrent call could overwrite it in between: teacher A's response, and A's
  `runs.jsonl` entry, could carry teacher B's source questions.
  `generate_detailed()` now returns a frozen result the API reads from, and a
  regression test drives 8 threads through a deliberately slow retriever.
- **The ML layer is serialised.** `Retriever.retrieve()` holds an `RLock` across
  embed → Chroma → rerank. Neither SentenceTransformer nor CrossEncoder
  documents thread-safety and one instance is shared by every request. The LLM
  call is deliberately outside the lock, so generation stays concurrent.
- **Silent data loss in the reranker.** `rerank()` paired candidates with model
  scores using `zip`, which stops at the shorter input — a cross-encoder
  returning fewer scores than pairs discarded the unscored tail with no error
  and no log line (five candidates in, two out). `score()` now raises
  `RerankerError` naming both counts.
- **`/taxonomy` advertised 12 phantom levels** (`LICENCE_*`, `PREPARATORY_*`)
  backed by 4 Arabic rows, because scope only inspects `levels[0]`. A teacher
  picking one got an empty retrieval and a 502. `Taxonomy` now takes
  `level_prefixes`, applied at build *and* load time, so existing indexes are
  corrected on read: 38 levels → 26, no reindex.
- **The English eval scored correct retrievals as wrong.** `topics_english.csv`
  predated the doc_id collision fix and was never rebuilt: against the current
  index it omitted 205 questions belonging to its own topics, listed one id that
  no longer exists, and carried 120 duplicates. Repaired in three separately
  measured steps — P@1 0.735 → 0.806, and 0.854 → 0.938 on well-specified
  queries — with each step's prediction matching its measured run.
- Re-applying an alias file from an earlier run marked its topics changed with
  +0 −0 and rewrote their rows. Caught by a dry run before any write.
- `make eval` called the eval script without the test file it requires, so it
  exited with a usage error.
- `tests/test_api.py` called `json.loads` without importing `json`, so two
  feedback-log tests raised `NameError` instead of running. The `/feedback`
  behaviour they assert was correct — the bug was in the test file.
- Two annotations contradicted their own null guards
  (`curriculum_rules.check_compliance`, `domain_rules.apply_subject_language_rule`).
  Both are reached from `normalize_row`, which is handed an unvalidated
  `json.loads` result, so the guards were right and the types were wrong.

### Security

- `pip-audit` found 7 advisories in `pip`, fixed by the upgrade `make setup`
  now performs, and 4 in `chromadb==1.5.9`, which has no fixed release. Those
  four are accepted with evidence and a 2026-12-01 review date: all target the
  Chroma *server*, and this project only constructs `PersistentClient` against
  a local directory (ADR-0002, which also states when the ignores must go).
- `.gitignore` hardened — `quizzes-raw-data.json` is matched unanchored. A
  stray 271 MB copy of the private corpus, carrying real teachers' names and
  emails, was sitting untracked in `notebooks/`, one `git add .` away from
  entering history.
- Eval test cases, topics CSVs, alias files and results are ignored by git:
  they are derived from the private corpus.
- Verified that no provider key appears anywhere in the repository's history
  and that no secret-shaped strings exist in tracked files.

### Known limitations

- Rate limits and metrics are per process. Scaling out needs Redis-backed
  counters, or enforcement at the proxy.
- Latency is exported as sum+count, not a histogram — true percentiles still
  come from `scripts/analyze_runs.py` offline.
- Secrets live in a plaintext `.env` on the host.
- The UI is an engineering tool, not a product surface: French copy, no i18n,
  little responsive testing, and only `MULTIPLE_CHOICE` is exposed.
- Feedback is single-rater and unblinded — useful for spotting patterns, not a
  substitute for a labelled set.
- Lint and type debt is an explicit, annotated allow-list that only shrinks
  (ADR-0001).
- The eval has ceilings of its own: duplicate queries with different correct
  quizzes cap P@1, the template queries contain their target title, and HNSW
  search returns a different candidate pool for about half of all queries
  between runs (aggregates stay within ±0.0005). See `eval/RESULTS.md`.
- 4 Arabic rows still carry phantom level keys in Chroma metadata; a caller who
  hardcodes one can still filter on it. Cleaning them needs a reindex.

---

## v1.1.0 — Phase 2 (math)

**Tag:** `v1.1.0` on `main`
**Merged via:** `feature/math-subject` → `dev` → `main`

### Added — Mathematics subject

- **Corpus:** ~1,371 math questions added to the index (1,003 fr at
  high-school level, 368 ar at middle/primary). Total index size grew
  from ~4,400 to ~5,780 documents.
- **Scope:** `configs/phase1_scope.yaml` now includes MATHEMATICS
  alongside ENGLISH, ARABIC, FRENCH.
- **Curriculum rule** (`src/data/curriculum_rules.py`): drops rows that
  violate the Tunisian curriculum mapping (primary/middle math = Arabic,
  high-school math = French). 36 mistagged rows automatically removed at
  normalize time.
- **Topics ground truth:** `eval/topics_math_fr.csv`,
  `eval/topics_math_ar.csv` — eval-ready CSVs with sample questions and
  per-quiz-title doc_id sets.
- **Test cases:** 720 FR + 360 AR LLM-generated retrieval test cases at
  `eval/math_*_retriever_test_cases_topic_specific.json`.
- **Eval baseline** recorded for math retrieval:
  - FR (720 cases): precision@1 0.615, hit@10 0.735, MRR 0.649
  - AR (360 cases): precision@1 0.492, hit@10 0.692, MRR 0.547
- **Audit notebook:** `notebooks/math_data_audit.ipynb` documents the
  Phase-2 discovery process — data quality issues found, curriculum
  decisions, LaTeX patterns in the corpus.

### Added — Quality safety nets

- **LaTeX validity check** (`src/generation/latex_validity.py`):
  rejects LLM output containing broken LaTeX (missing closing brace,
  unclosed inline math, over-escaped delimiters like `\\)` instead of
  `\)`) before it reaches the frontend renderer. Wired into the existing
  retry loop, so the LLM gets specific feedback and tries again. ~30 lines
  of pure-stdlib code, 19 unit tests.
- **Generalized eval scripts** (`scripts/eval/validate_test_cases.py`,
  `scripts/eval/run_retriever_eval.py`): now handle multiple subjects
  per language (math sits alongside the language-subject in the same
  language code).

### Known issues / limitations

- **Math retrieval is ~10pp weaker than language retrieval** across
  precision@1, hit@10, MRR. The dominant failure mode is sibling-topic
  confusion (test demands "Fonction Logarithme 2", retriever returns
  "Fonctions affines"). The retrieved content is semantically relevant
  but not the exact title.
- **Production mitigation:** `school_phase` filter (already exposed)
  narrows the candidate pool and removes most sibling-topic noise.
- **LLM still occasionally produces a stray `\)` at the end of a math
  expression.** The validator now treats this as cosmetic (frontend
  shows it as literal text) rather than fatal — better than killing a
  10-question quiz over a typo.
- **No SymPy-based correctness verification.** The LLM's math is
  reviewed manually; if production usage surfaces wrong answers we'd
  add this. Deferred because algebra-only corpus + small known error
  rate didn't justify the complexity yet.

---

## v1.0.0 — Phase 1 (en/ar/fr literature + grammar)

**Tag:** `v1.0.0` on `main` (`c73791b`)

### Shipped

- **Subjects:** ENGLISH, ARABIC, FRENCH (literature + grammar
  curricula).
- **Languages:** en, ar, fr.
- **Levels:** PRIMARY, MIDDLE, HIGH school.
- **Corpus size:** 4,411 indexed questions (3,079 en + 1,317 ar + 15
  fr).
- **Architecture:**
  - Bi-encoder (BGE-M3) + cross-encoder reranker (BGE-reranker-v2-m3)
  - Chroma vector store with metadata pre-filter
  - LLM provider switchable: Gemini 2.5 Flash (default), Groq, Ollama
- **API:** FastAPI with `/health`, `/taxonomy`, `/retrieve`,
  `/quiz/generate`. Validates input, retries on LLM failure, logs every
  call to `logs/runs.jsonl`.
- **`school_phase` filter** on both retrieve and generate endpoints.
  Production passes the user's grade phase (PRIMARY/MIDDLE/HIGH); the
  retriever metadata-filters before scoring.
- **doc_id integrity fix:** previously, ~6% of rows were silently
  overwritten in Chroma because two questions in the same quiz could
  collide on `doc_id`. Fixed at ingest with collision-aware suffixes
  (`__q5`, `__q5_2`, `__q5_3`). Backward-compatible with existing eval
  ground truth.
- **Eval framework:**
  - Per-language topics CSVs (`eval/topics_*.csv`) — ground truth per
    quiz_title.
  - LLM-generated retriever test cases per language.
  - `scripts/eval/run_retriever_eval.py` computes precision@k,
    recall@k, hit@k, MRR. Recorded baselines:
    - EN (2,761 cases): precision@1 0.735, hit@10 0.871, MRR 0.784
      (later found to be scored against a stale answer key — see Unreleased)
    - AR (400 cases): precision@1 0.585, hit@10 0.778, MRR 0.655
    - FR (46 cases): precision@1 0.870, hit@10 1.000, MRR 0.914
      (small sample, French literature is the smallest corpus slice).
- **Per-stage timing + structured logging** across retriever and
  pipeline, exposed via `last_timings` and `runs.jsonl`.

### Known issues at v1.0.0

- Math subject deferred — see Phase 2.
- French corpus small (15 literature docs) — eval numbers correspondingly
  noisy.
- No automated LLM-output quality eval (manual review only).
- No API authentication.

---

## How releases are made

1. Develop on a feature branch (`feature/<topic>`).
2. Manual end-to-end validation on real `/quiz/generate` calls.
3. Atomic commits with substantive messages ("why" not just "what").
4. Merge into `dev`, then `dev` → `main`.
5. Tag the merge commit on `main` with semantic version (`v1.0.0`,
   `v1.1.0`, etc.). Move the tag forward only when shipping forward;
   never edit a tagged commit.
6. Update this CHANGELOG with the release notes.
