# Changelog

Human-readable summary of what shipped per release. The git history has
the per-commit detail; this file has the per-release story.

---

## Unreleased — Engineering UI + generation feedback

**On branch:** `feature/ui` (branched off `dev`)

> Note: three commits on this branch carry messages that don't match their
> contents (they were committed between edits, so each `git add -A` swept up
> the previous change). The branch was already pushed, so the history stands.
> This section is the accurate record.

### Added — `/ui`, a single-page console

- One self-contained HTML file (`src/api/static/index.html`) served at `/ui`,
  with `/` redirecting to it. No build step, no node, no second container —
  it lives under `src/` so the existing `COPY src/` picks it up, and being
  same-origin means CORS stays off.
- **The dropdowns encode the curriculum.** Subject → language → school phase
  narrow each other according to `src/data/curriculum_rules.py`: maths is
  Arabic at primary/collège and French at lycée; ENGLISH/ARABIC/FRENCH each
  fix their language. Combinations the corpus physically cannot satisfy are
  unselectable rather than a 400 fifteen seconds later, with a line of text
  explaining why the choice narrowed.
- **Thin-cell warning.** Selecting Français warns that the entire subject is
  15 questions across two quizzes (`l'amour`, `Enfants de tous les pays`),
  both literature, and steers the topic field at them. Discovered the hard
  way: `future simple` returned 400 because no French grammar exists.
- Arabic results render RTL; MathJax typesets after each render so LaTeX in
  maths questions displays; answers hide behind a reveal toggle; a print
  stylesheet drops the form and shows every answer.
- `/ui` is unauthenticated **by design** — a browser navigating to a URL
  cannot attach `X-API-Key`, so requiring one would make the page
  unreachable. The page is inert: no key baked in. When an API call returns
  401 it prompts and retries, holding the key in memory for the session only.
  Never `localStorage`; a test asserts that.

### Added — retrieval visibility

- The debug panel shows the chunks the LLM actually saw: cosine distance
  colour-coded against the 0.60 floor, quiz title, question text, levels and
  `doc_id`, plus per-stage timings. This is what separates "the quiz is bad"
  from "the retriever fed it love poetry".
- `POST /quiz/generate` now returns `timings` alongside `retrieval` when
  `include_retrieval=true`. Opt-in, so the default response shape the
  platform sees is unchanged.

### Added — `POST /feedback`, the missing measurement

- One human judgement per generated question (`up`/`down` + optional note),
  appended to `logs/feedback.jsonl`. `eval/RESULTS.md` has real retrieval
  baselines but states outright that generation quality is judged manually
  and never written down — which makes every prompt or threshold change
  unfalsifiable. This turns normal use into a labelled set.
- Rows carry `request_id` and **not** the retrieval: the join back to
  `runs.jsonl` already has the filters, chunks, distances and timings.
  Duplicating them would double the storage and let the copies drift.
- Authenticated but deliberately **not** rate-limited — throttling the one
  signal we want more of would be perverse.
- `scripts/analyze_feedback.py` performs the join and reports up/down by
  cell, the downvote notes, and the comparison that matters: mean worst-chunk
  distance for upvoted vs downvoted questions. If downvoted questions were
  built from farther chunks, `llm.default_max_distance` is too loose — which
  is the per-language split `configs/models.yaml` has had a TODO for since
  May. Refuses to interpret below 5 judgements per verdict.

### Known issues / limitations

- The UI is an engineering tool, not a product surface: French copy, no
  i18n, no responsive testing beyond a narrow breakpoint.
- Only `MULTIPLE_CHOICE` is exposed. `FILL_IN_THE_BLANKS` works in the API
  but has 99 rows in the corpus, so retrieval is thin.
- Feedback is single-rater and unblinded — useful for spotting patterns and
  tuning thresholds, not a substitute for a proper eval set.

---

## Unreleased — Production hardening

**On branch:** `feature/production-hardening`

Everything here is about making the service safe to expose. No change to
retrieval or generation behaviour; no reindex required.

### Added — API security boundary

- **API-key authentication** (`src/api/security.py`). `X-API-Key` header or
  `Authorization: Bearer`. Keys come from the `API_KEYS` env var
  (comma-separated); constant-time comparison. **Unset means auth is
  DISABLED** and the server logs a loud warning at startup — fail-open by
  design so local development and the test suite keep working.
  `/health` and `/ready` stay open so container healthchecks work.
- **Rate limiting** on `/retrieve` and `/quiz/generate` —
  `RATE_LIMIT_PER_MINUTE` (default 30) per caller over a rolling 60s window,
  keyed on the API key (hashed) or client IP. Returns 429 + `Retry-After`.
  Counted **per process**: with N workers the real ceiling is N × the limit.
- **Correlation IDs** — every response carries `X-Request-ID`, an inbound one
  is honoured, and the ID is written into `runs.jsonl`.
- **Opaque error bodies.** 500 responses returned `f"{type(exc).__name__}:
  {exc}"`; 502s embedded `diagnose_empty()` output (corpus size, per-subject
  language counts). Both now return a generic message plus the request ID,
  with full detail logged server-side. 400s still pass their message through
  — those are caller-caused and actionable.
- **CORS is now an explicit decision** — `CORS_ALLOW_ORIGINS`, unset by
  default (correct for a server-to-server caller). Warns on `*`.

### Fixed — cross-request data leak (concurrency)

- `QuizPipeline.generate()` stashed the retrieval on `self.last_retrieval`
  and the endpoint read it after the call returned. One pipeline instance
  serves every request from FastAPI's thread pool, so a concurrent call could
  overwrite it in between: teacher A's response — and A's `runs.jsonl` entry —
  could carry teacher B's source questions.
  New `generate_detailed()` returns a frozen `GenerationResult(quiz,
  retrieval, timings)`; the API reads only from it. `generate()` remains for
  the CLI and still mirrors `last_*`. Regression test drives 8 threads through
  a deliberately slow retriever and asserts each gets only its own data.
- **The ML layer is serialised.** `Retriever.retrieve()` now holds an `RLock`
  across embed → Chroma → rerank. Neither SentenceTransformer nor CrossEncoder
  documents thread-safety, and one instance is shared by every request. The
  LLM call is deliberately outside the lock, so generation stays concurrent.

### Fixed — out-of-scope levels in `/taxonomy`

- `scope.decide_in_scope()` only inspects `levels[0]`, so a kept row could
  carry out-of-scope SECONDARY tags. The taxonomy harvested all of them, and
  `/taxonomy` advertised 12 phantom levels (`LICENCE_*`, `PREPARATORY_*`)
  backed by 4 Arabic rows. A teacher picking one from a dropdown got an empty
  retrieval and a 502. `Taxonomy` now takes `level_prefixes`, applied at build
  **and load** time — so existing indexes are corrected on read, no reindex.
  38 levels → 26.

### Changed — resource safety

- **LLM clients reuse their SDK client** instead of constructing one per call,
  and every call carries a timeout (`LLM_TIMEOUT_SECONDS`, default 90).
  Without one a hung provider pinned a worker thread, up to `max_attempts` of
  them per request. Timeout wiring falls back gracefully when an SDK version
  rejects the argument.
- **`logs/runs.jsonl` rotates** past `RUNS_LOG_MAX_BYTES` (default 50 MB),
  keeping 3 generations. It was unbounded.
- **Author PII removed from API responses and the run log.**
  `author_name` / `author_email` identify real teachers who wrote the source
  corpus and were shipping on every `include_retrieval=true` response and
  every logged run. Set `INCLUDE_AUTHOR_METADATA=1` to restore.
- **Operational signals moved from `warnings.warn` to `logging`.**
  The warnings module dedupes per code location, so empty-retrieval,
  low-pool, multi-level and taxonomy-validation signals fired **once per
  process** and were then silent forever. A test now asserts the empty-store
  signal repeats on every call.
- **Dependencies pinned.** `requirements.lock.txt` generated from the running
  image; the Dockerfile builds from it. `requirements.txt` remains the
  statement of intent.
- **`.gitignore` hardened** — `quizzes-raw-data.json` is now matched
  unanchored. A stray 271 MB copy of the private corpus (real teachers' names
  and emails) was sitting untracked in `notebooks/`, one `git add .` away
  from entering history.

### Added — operations

- **`GET /ready`** — probes Chroma document count, payload load and LLM
  client; 503 when degraded. `/health` only ever proved the process was
  alive, which cannot distinguish "serving" from "will 502 on every request".
  The compose healthcheck now uses `/ready`.
- **`GET /metrics`** — Prometheus text format, no client library
  (`src/api/observability.py`). Request counts by method/path/status, latency
  sum+count per path, event counters for rate_limited / unauthorized /
  generation_failed. Behind the API key. `/health` and `/metrics` excluded
  from their own metrics. Per-process, like the rate limiter.

### Known issues / limitations

- Rate limits and metrics are per process. Scaling out needs Redis-backed
  counters, or enforcement at Nginx.
- Secrets live in a plaintext `.env` on the host.
- Latency is exported as sum+count, not a histogram — true percentiles still
  come from `scripts/analyze_runs.py` offline.
- The 4 Arabic rows carrying phantom level tags still have
  `levels_LICENCE_*` keys in Chroma metadata. A caller who hardcodes one can
  still filter on it; `/taxonomy` no longer offers it. Cleaning the metadata
  needs a reindex.

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
