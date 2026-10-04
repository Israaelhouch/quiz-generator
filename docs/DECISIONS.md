# Decisions

Seven records, kept because a future reader — including future you — would
otherwise have no way to tell a deliberate choice from an accident: a security
exposure that is dormant today but becomes real if the deployment shape
changes, which of the checker's complaints turned out to be real bugs, and why
the eval's answer key and test questions look the way they do.

Format: **context → options → decision → consequences** (CLAUDE.md §2).

> Earlier decisions (chunking strategy, embedder choice, reranker pool size,
> the distance floor) are not written up here. They are reconstructable from
> `configs/models.yaml` and `eval/RESULTS.md`.

---

## ADR-0001 — Lint and type debt is baselined, not swept

**Status:** accepted · 2026-09-08

**Context.** Turning on ruff and `mypy --strict` against the existing codebase
produced 168 lint violations and 93 type errors across 23 files. CI cannot be
introduced red.

**Options.**
1. Fix all 261 findings in one change.
2. Weaken the tools until the codebase passes.
3. Baseline what exists, gate on what is new, shrink the baseline over time.

**Decision.** Option 3, except for ruff's 62 *safe* autofixes — import
sorting, dead imports, `datetime.UTC` — which were applied in their own
commit, verified by the suite staying at 267 passed.

**Consequences.** `make lint` is green and genuinely gates new code. The
allow-lists in `pyproject.toml` are counted and annotated, and only shrink.

**Three baselined entries were findings, not cosmetics.** All three are now
fixed and removed from the baseline; each is written up below. The baseline
went from 93 type errors in 23 files to 89 in 20.

---

## ADR-0002 — Four chromadb advisories are accepted, conditionally

**Status:** accepted · 2026-09-08 · **review 2026-12-01**

> **Read this before changing how Chroma is deployed.** The exemption below
> depends entirely on this project running Chroma as a local embedded store.
> If that ever changes, these four ignores must be removed first.

**Context.** `pip-audit` reports four vulnerabilities in `chromadb==1.5.9`:

| ID | Issue |
|---|---|
| PYSEC-2026-311 | pre-authentication code injection |
| CVE-2026-45830 | missing authorization validation |
| CVE-2026-45831 | `SimpleRBACAuthorizationProvider` flaw |
| CVE-2026-45833 | authenticated code injection |

None has a fix version, and 1.5.9 is the latest release — there is nothing to
upgrade to.

**Options.**
1. Fail CI until upstream ships a fix (red for an unknown duration).
2. Drop `pip-audit` from the gate.
3. Ignore these four IDs specifically, with justification and a review date.

**Decision.** Option 3.

**Evidence.** All four live in Chroma's *server*: its authentication, its
authorization providers, its request handling. This project never runs that
server. The only client construction in the codebase is
`chromadb.PersistentClient(path=...)` at `src/indexing/vector_store.py:121` —
an embedded, on-disk store. There is no `HttpClient` anywhere in `src/`,
`scripts/` or `docker-compose.yml`, no Chroma port is exposed, and no auth
provider is configured. The vulnerable code paths are unreachable.

**Consequences.** `make audit` and the CI security job stay green and keep
catching *new* advisories, which is the point of the gate. The exemption is
narrow — four specific IDs, not the package and not the tool — and carries a
review date.

**The condition, stated plainly:** switching Chroma to client/server mode, or
exposing its port, turns all four of these into live exposures in a service
whose CI will still be green. The ignores are in `Makefile` (the `audit`
target) and `.github/workflows/ci.yml`. Delete them there, in the same change
that moves Chroma off `PersistentClient`.

---

## ADR-0003 — The three findings, and what they turned out to be

**Status:** accepted · 2026-09-08

Each was surfaced by turning on the gates in ADR-0001, then investigated
before being touched. Only one was a live bug; saying so plainly is more
useful than three equally-weighted bullet points.

### 1. The reranker silently dropped candidates — a real bug

`Reranker.rerank()` paired candidates with model scores using
`zip(candidates, scores)`. `zip` stops at the shorter input, so if the
cross-encoder ever returned fewer scores than pairs, the unscored tail was
discarded and `rerank()` returned **fewer candidates than it was given**, with
no exception and no log line. Reproduced with a stub model: five candidates
in, two out, silence.

Whether `sentence_transformers` can actually do that is beside the point — the
failure mode is invisible, and its symptom is a slightly worse quiz rather
than an error, so nothing downstream would ever attribute it correctly.

`score()` now enforces its own docstring (`len(scores) == len(candidates)`) and
raises `RerankerError` naming both counts, and the `zip` uses `strict=True` as
a backstop. Regression test:
`test_rerank_never_silently_drops_candidates`, deliberately phrased as *"never
returns fewer candidates than it was handed"* rather than *"raises
RerankerError"*, so it is meaningful against the old code — where it fails
with `2 of 5 candidates and raised nothing`.

The other three `zip()` sites (`retriever.py`, `indexing/build.py`,
`indexing/query.py`) all zip parallel arrays from a single Chroma response.
They got `strict=True` too, but as assertions on a driver contract, not as bug
fixes.

### 2. The `None` guards — the annotations were wrong, not the guards

`curriculum_rules.check_compliance` and
`domain_rules.apply_subject_language_rule` both guard `if subj is None:
continue` while declaring `subjects: list[str]`, so mypy called the guard
unreachable. Deleting it was the obvious reading, and would have been wrong.

Traced the data: both are called from `normalize_row`, which receives
`json.loads(line)` from the interim JSONL and **never re-validates it against
`FlatQuestion`**. Through the normal pipeline a null cannot get that far —
ingest validates `RawQuiz`, whose `subjects: list[str]` rejects it (verified:
Pydantic raises `string_type`), and drops the quiz as
`quiz_validation_failed`. But normalize is a CLI stage that accepts any
`--input`, so nothing between the file and these functions guarantees element
types.

So the annotations were widened to `Sequence[str | None] | None` — `Sequence`
because `list` is invariant and `list[str]` would not satisfy
`list[str | None]`.

Worth recording, because it splits the two cases: deleting the guard in
`domain_rules` **does** change behaviour — a null stringifies to `"NONE"`,
becomes the primary subject, and masks the real subject behind it (confirmed:
the regression test fails without the guard). In `curriculum_rules` the guard
is genuinely redundant, since `"NONE"` matches no rule key and falls through
the next branch anyway. It is kept there for symmetry, and the comment says
so rather than implying it is load-bearing.

### 3. `str` where a `Literal` was declared — a real gap, no live exposure

`QuizPipeline.generate_detailed` accepted `language: str` and
`question_type: str`, then passed them to `GenerationRequest`, which declares
`Literal['en','fr','ar']` and `Literal['MULTIPLE_CHOICE','FILL_IN_THE_BLANKS']`.

Checked both shipped entry points before changing anything: the API declares
the same Literal on `GenerateRequest`, so FastAPI rejects bad values with a
422, and the CLI uses `argparse(choices=[...])`. Nothing reachable today can
pass an invalid value. The annotation was simply weaker than what every caller
already guarantees, discarding type information at the boundary.

`generate_detailed` now declares the same aliases, imported under
`TYPE_CHECKING` so the lazy runtime import inside the method — there to keep
generation's module graph out of orchestrator import time — is preserved.

**Post-mortem (CLAUDE.md §5.6).** None of these were caught earlier because
nothing ran a linter or a type checker over this repository; there was no CI
at all. All three surfaced within minutes of the gates being switched on. What
catches them now: `ruff check` and `mypy --strict` on every push and pull
request, plus 8 regression tests, of which the reranker one is verified to
fail against the pre-fix code.

---

## ADR-0004 — The eval answer key is repaired in measured steps, not regenerated

**Status:** accepted · 2026-09-10

**Context.** The English retrieval eval scores a retrieved question as correct
only if its doc_id is listed under the target topic in
`eval/topics_english.csv`. That file was built on 2026-05-13, before the
2026-05-18 doc_id collision fix, and never rebuilt. Against the current index
it omitted 205 questions belonging to its topics, listed one unknown id, and
carried 120 duplicate ids. Its 206 topics were also merged by hand from 251
raw titles, and only partly: seven topic names exist on no indexed question at
all. The
2,761 test cases target those 206 hand-merged names. Nothing compared the key
with the index, so the published English P@1 of 0.735 understated retrieval.

**Options.**
1. Regenerate the CSV by re-running the notebook that built it.
2. Refresh doc_ids, then merge title variants automatically by similarity
   (typos, subtitles, numbered parts).
3. Refresh doc_ids, then merge only by a narrow mechanical rule, and leave
   every judgement merge to a human — each step measured separately.

**Decision.** Option 3, with a validator that refuses to run the eval on a key
that disagrees with the index.

**Why not option 1.** The notebook groups by exact title and produces about
360 topics. Every one of the 2,761 test cases targets a hand-merged name that
would no longer exist.

**Why not option 2.** The evidence ran against it everywhere it was tried:

- A similarity rule proposed 151 English subtitle/fuzzy pairs, most of them
  wrong — `'The + adjective'` matched `'Compound Adjectives'` and
  `'Possessive Adjectives'`.
- In Arabic, all 29 near-identical title pairs are distinct lessons or
  sub-topics:
  `الدَّرْسُ الثَّانِي` (lesson 2) and `الدَّرْسُ الثَّانِي عَشَر` (lesson 12);
  the same verb in the indicative and in the subjunctive mood. Near-identical
  spelling routinely means different grammar.
- Of the English candidates reviewed by hand, the single largest score gain
  (+0.0025 P@1) came from a merge judged wrong — transport prepositions
  folded into dependent prepositions. An automatic rule would have kept it and
  reported an improvement.

**The narrow rule.** Titles merge automatically only when they differ in case,
spacing, punctuation, `&`/`and`, or a leading English article — and only into a
topic that is the sole owner of the group. A leading article is stripped only
when a word follows it, and punctuation is found by Unicode category, because a
word-character regex also deletes Arabic vowel marks.

**Consequences.**
- English P@1: 0.735 (stale) → 0.795 (refresh) → 0.802 (safe merges) →
  0.806 (reviewed merges). Re-scoring fixed retrieval results against each
  key reproduces each run to within ±0.0005, so each gain is attributable to
  its step.
- The key is a function of the original CSV, the index and two alias files
  (`eval/topic_aliases.safe.yaml`, `eval/topic_aliases.reviewed.yaml`), and
  rebuilds identically whether the steps are applied one by one or in one pass. Rejected merges are recorded in the
  reviewed file so they are not re-proposed without new evidence.
- The alias files, like the CSVs, name real quiz titles and stay out of git.
- What this does **not** fix: the test cases themselves. They are
  template-generated with the target title inside the query, some queries are
  labelled with several correct quizzes (capping P@1 at 0.887 for English and
  0.667 for Arabic maths), and the generator is not in the repository. Those
  need new test cases, not a better answer key.

---

## ADR-0005 — Realistic eval questions are labelled blind and may accept several quizzes

**Status:** accepted · 2026-09-10

**Context.** The template test cases contain their target quiz title word for
word, so they mostly measure title matching (ADR-0004). A realistic set was
needed: questions phrased the way teachers ask. Two choices decide whether such
a set can be trusted — who picks the correct answers, and how many a question
may have.

**Options.**
1. One correct quiz per question, as in the template set.
2. Several correct quizzes, chosen with help from the search model — for
   example by accepting what it retrieves.
3. Several correct quizzes, chosen by reading each topic's questions, without
   the search model and before any score is seen.

**Decision.** Option 3.

**Why.** Option 1 repeats the flaw that makes template queries such as
'english grammar exercises' impossible to pass: a real request is often served
equally well by several topics. Option 2 lets the system under test decide what
counts as correct, which inflates its score by construction. Scores were also
withheld until the questions had been reviewed, so failing questions could not
be quietly removed.

**Consequences.**
- Test cases gained an optional `also_correct_quiz_titles` list; existing test
  files are unaffected.
- The rule is applied to every question, not only to failures. The first
  labelling considered only topics with 8 or more questions and missed fair
  answers among smaller ones. The audit that fixed it added answers to 13
  questions, 9 of which had already passed, and P@1 moved from 0.680 to 0.760
  with identical search results.
- The check that a question avoids its titles' words ignores short function
  words, so it cannot flag a title such as "In - On - At", and it is only as
  complete as the answer list: six questions share a word with a correct title
  added in the audit.
- 50 questions give a wide range (P@1 0.64–0.88). The set is for finding
  weaknesses, not for comparing close configurations.

---

## ADR-0006 — The environment is read once, through a typed Settings object

**Date:** 2026-10-04 · **Status:** accepted

### Context

Thirteen `os.environ.get` calls were spread across five modules — API-key
auth and rate limiting in `api/security.py`, the run-log paths, rotation size
and privacy switch in `api/server.py`, the LLM timeout and provider keys in
`generation/llm_client.py`, the Ollama host in `pipeline/orchestrator.py`, the
log level in `shared/logging_setup.py` — and two more in
`scripts/evaluate_runs.py`.

Three consequences, all observed in this repository:

1. **Malformed values were swallowed.** `RATE_LIMIT_PER_MINUTE=abc` logged a
   warning and applied 30; `LLM_TIMEOUT_SECONDS=not-a-number` applied 90; a
   negative size was clamped. A typo in a deployment therefore looked as if it
   had been applied, and only an operator reading warning logs would notice.
2. **Defaults lived at the point of use.** `RUNS_LOG_PATH` was read into a
   module constant at import time, which also meant tests had to reassign a
   module global to redirect it.
3. **The documentation drifted.** `.env.example` claimed an unset
   `LLM_TIMEOUT_SECONDS` meant "the provider's own default" (the code applied
   90s) and an unset `RUNS_LOG_MAX_BYTES` meant "no rotation" (the code
   applied 50 MB). Nothing could catch that.

### Options

1. **Leave it.** Zero risk, but CLAUDE.md Part II §2 requires typed,
   validated, injected configuration, and every new variable repeats the
   problem.
2. **A plain dataclass loaded by hand.** No new dependency, but the parsing
   and validation would be hand-written per field.
3. **A `pydantic-settings` `BaseSettings` class** — already present
   transitively via FastAPI's dependency tree, matching the Pydantic v2 models
   used everywhere else in the project.

### Decision

Option 3. `src/shared/settings.py` declares every variable with its type and
its default; `get_settings()` returns a cached instance, so the environment is
read once per process and a malformed value raises at startup. Secrets are
`SecretStr`, so a stray log line or `repr` prints `**********`.

Two deliberate restrictions:

- **No `.env` file is read by the class.** Compose and `run_local.sh` inject
  the variables and the test suite scrubs them; an untracked local `.env`
  silently changing test behaviour is worse than an explicit injection.
- **Settings are cached, not re-read per access.** Request handling must not
  see the environment shift underneath it. `reset_settings()` exists for tests
  and is called by the suite's `_env` helper.

### Consequences

- A bad value now **stops the process** instead of degrading quietly. This is
  a behaviour change, and the test that asserted the old lenient fallback was
  rewritten to assert the refusal.
- `RUNS_LOG_PATH` and `FEEDBACK_LOG_PATH` became functions rather than
  import-time constants, so tests redirect them through the environment like
  every other setting instead of patching module globals.
- A test compares the fields of `Settings` against the variables documented in
  `.env.example` and fails if either side gains an entry the other lacks, so
  the drift in point 3 above cannot recur.
- `pydantic-settings` moves from a transitive pin to a declared dependency in
  `requirements.txt`.
- The LLM client now takes its key from settings, which is the seam the
  roadmap's vLLM and Langfuse work needs: a new provider adds a field, not
  another `os.environ` call.

---

## ADR-0007 — A pipeline config is validated or refused, never defaulted

**Date:** 2026-10-04 · **Status:** accepted · **Extends:** ADR-0006

### Context

`configs/models.yaml` was already parsed into a validated Pydantic model. The
other three configs were read with a bare `yaml.safe_load` and a chain of
`.get(key, default)` calls, which meant every mistake in them was silent:

- `load_recipe` copied only keys it already knew: `include_choice` (singular)
  was **dropped without a word**, so the index would be built without answer
  choices while the config said they were included.
- Selecting a recipe that does not exist — `recipe: title_only` with only
  `default` defined — fell back to the default flags. An A/B test between two
  recipes could therefore run the control twice and be reported as a
  comparison.
- A missing config file returned the defaults. The job succeeded and produced
  an index that no file on disk describes.
- `load_scope` accepted `subjects: ENGLISH` (a string, not a list) and turned
  it into `frozenset({'E','N','G','L','I','S','H'})`, dropping every row as
  out of scope. The pipeline would report an empty corpus, not an error.
- `load_subject_aliases` coerced values with `str()`, so `MECHANIC: 1` mapped
  that subject to the literal `"1"` and removed those rows from every subject
  filter downstream.

These are not crashes. They are quality regressions in a system whose
deliverable is measured quality, and each one would surface as an unexplained
move in P@1 days later.

### Decision

The three remaining configs get what `models.yaml` already had: a Pydantic
model per block with `extra="forbid"`, loaded through
`src/shared/yaml_config.read_yaml_mapping`, which requires the file to exist
and to contain a mapping. A selected recipe that is not defined raises and
names the recipes that are.

`extra="forbid"` is the point of the decision rather than an implementation
detail: a config key that nobody reads is the failure mode here, so an
unknown key has to be an error.

### Consequences

- Behaviour change, in the same direction as ADR-0006: a malformed or missing
  pipeline config now stops the run. No existing test depended on the silent
  fallbacks; the real configs load to byte-identical values (recipe
  `default`, the same six flags, threshold 100, `normalize_latex: true`).
- `src/data/scope.py` went from **0% to 100%** test coverage. It decides
  corpus membership and had no tests at all, which was the weakest point in
  the suite.
- The `DEFAULT_RECIPE_FLAGS` and `DEFAULT_SEPARATORS` constants are now
  derived from the models, so the defaults cannot drift from the schema.
- Still open: `run_retriever_eval` snapshots `configs/models.yaml` with every
  run but not `configs/pipeline.yaml`, so a run's numbers cannot be traced
  back to the recipe that built the index it searched. Worth closing before
  the next recipe A/B.
