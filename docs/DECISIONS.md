# Decisions

Three records, kept because a future reader — including future you — would
otherwise have no way to tell a deliberate choice from an accident, would not
know about a security exposure that is dormant today but becomes real if the
deployment shape changes, and would not know which of the checker's complaints
turned out to be real bugs.

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
