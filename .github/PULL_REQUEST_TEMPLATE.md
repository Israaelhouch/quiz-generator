## What

<!-- One or two sentences. The diff shows the detail. -->

## Why

<!-- The problem this solves. If it fixes a bug, how the bug was reproduced. -->

## How verified

<!-- Commands run and what they printed. If retrieval could have moved,
     the eval numbers before and after, and the run directory they came from. -->

```
make ci
```

## Risk / rollback

<!-- What breaks if this is wrong, and how to undo it. -->

## Checklist

- [ ] Tests added or updated; a bug fix has a test that fails on the old code
- [ ] `make ci` passes (ruff, mypy --strict, pytest, gitleaks, pip-audit)
- [ ] Docs, ADR and CHANGELOG updated
- [ ] No corpus data, secrets or personal data added
- [ ] Any metric quoted comes from a recorded run in `eval/results/`
- [ ] Structure and behaviour are not mixed in one commit
