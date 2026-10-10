# Contributing

This repository is the public mirror of a service developed privately, so it
has one maintainer and no external roadmap. Issues and pull requests are
welcome anyway — particularly ones that point out a claim the repository does
not support.

## Getting a working checkout

```bash
make setup        # virtualenv, pinned dependencies, dev tools
make test         # the full suite; no models, keys or network required
```

The suite mocks every heavy adapter, so it runs in seconds without torch, a
GPU or an API key. `make run` additionally builds the sample corpus and serves
the API; the first run downloads about 1.2 GB of model weights.

## Before opening a pull request

```bash
make ci           # ruff, mypy --strict, pytest with coverage, gitleaks, pip-audit
```

CI runs the same gates. Four things are worth knowing about them:

- **`mypy --strict` is ratcheted.** `pyproject.toml` carries a per-module
  allow-list of pre-existing findings. It only shrinks; adding to it needs a
  reason in the pull request description.
- **Ruff's `per-file-ignores`** works the same way, and `T20` (print) is
  allowed in CLI entry points, where stdout is the output contract.
- **`pip-audit` ignores four chromadb advisories**, with evidence and a review
  date, in ADR-0002. They are re-checked on 2026-12-01.
- **Tests are named as specifications** — `test_<unit>_<condition>_<expected>`
  — and a bug fix ships with a test that fails on the old code.

## What does not belong in a commit

The corpus is real curriculum content written by named teachers. It is not in
this repository and must not enter it: `data/`, `eval/*.json`, `eval/*.csv`,
`eval/*.yaml` and `eval/results/` are gitignored, and gitleaks runs in
pre-commit and in CI. The synthetic corpus in `data/sample/` exists so that
every pipeline stage can be exercised without it.

Metrics follow the same rule in the other direction: a number in the README or
in `eval/RESULTS.md` has to come from a run in `eval/results/`, and every run
records the git commit, the config hashes and the index it searched. If a
change could move retrieval quality, the pull request says what the eval
measured before and after.

## Commits

Conventional Commits, one logical change per branch, and a message that
explains why rather than what:

```
<type>(<scope>): <imperative summary ≤ 72 chars>

Why this change. The diff already shows what.
Metric or behaviour impact, if any.
```

Structural changes and behaviour changes never share a commit.
