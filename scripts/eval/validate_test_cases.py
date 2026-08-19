"""Validate an LLM-generated retriever test-cases JSON before running eval.

Catches three classes of problem:

  1. Schema violations (missing field, wrong type, invalid enum value).
  2. `target_quiz_title` values that don't exist in the corresponding
     `eval/topics_<lang>.csv` — typically diacritic / accent / spacing typos
     from the generator LLM. These would silently get zero ground-truth docs
     and inflate "false negatives" in the eval.
  3. Cases where `top_k` exceeds the number of available ground-truth docs
     for that title — recall@k can never reach 1.0 there. Reported as
     warnings, not failures.

Exit code: 0 if every case is valid, 1 otherwise.

Usage:
    python -m scripts.eval.validate_test_cases eval/english_retriever_test_cases.json
    python -m scripts.eval.validate_test_cases eval/arabic_retriever_test_cases.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Valid (language, subject) pairs. Used to flag generator bugs where the
# LLM emits a test case with mismatched language and subject (e.g. an
# Arabic-grammar query with subject=ENGLISH).
#
# Math adds a second subject per language (Tunisian curriculum: high-school
# math is in French, middle/primary math in Arabic), so the relationship
# is no longer 1:1.
LANG_TO_SUBJECTS: dict[str, set[str]] = {
    "en": {"ENGLISH"},
    "ar": {"ARABIC", "MATHEMATICS"},
    "fr": {"FRENCH", "MATHEMATICS"},
}


class TestCase(BaseModel):
    """One LLM-generated retriever test case.

    `query_type` is left as a free-form string so new generator templates
    don't have to re-edit this file. Validation only checks it's non-empty.

    `levels` and `levels_match` are optional retrieval filters that mirror
    the production API. They default to None (no level filter) for back-
    compat with older test-case JSONs that pre-date level-aware eval.
    When set, the runner passes them straight through to
    `retriever.retrieve()` so the eval reflects how the platform actually
    queries the retriever (with the user's grade level).
    """

    query: str = Field(min_length=1)
    language: Literal["en", "fr", "ar"]
    subject: str = Field(min_length=1)
    top_k: int = Field(gt=0, le=200)
    query_type: str = Field(min_length=1)
    target_quiz_title: str = Field(min_length=1)
    levels: list[str] | None = Field(default=None)
    levels_match: Literal["any", "all"] | None = Field(default=None)


# ---------------------------------------------------------------------------
# Topic loading (the ground-truth source)
# ---------------------------------------------------------------------------

EVAL_DIR = Path("eval")

# Topics CSV path per (language, subject) pair. One CSV per subject keeps
# ground truth cleanly scoped — math topics live in topics_math_<lang>.csv
# and language-subject topics live in topics_<lang>.csv as before.
TOPICS_FILE_BY_LANG_SUBJECT: dict[tuple[str, str], Path] = {
    ("en", "ENGLISH"):     EVAL_DIR / "topics_english.csv",
    ("ar", "ARABIC"):      EVAL_DIR / "topics_arabic.csv",
    ("fr", "FRENCH"):      EVAL_DIR / "topics_french.csv",
    ("ar", "MATHEMATICS"): EVAL_DIR / "topics_math_ar.csv",
    ("fr", "MATHEMATICS"): EVAL_DIR / "topics_math_fr.csv",
}


def load_topic_index(
    lang_subject_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], int]:
    """Return {(language, quiz_title) -> n_relevant_docs} for the given pairs.

    Only loads topics CSVs we actually need (by (language, subject) pairs
    present in the test set). Missing CSVs are reported via stderr but
    don't abort — the cross-check loop will mark every test case for that
    pair as MISSING.
    """
    index: dict[tuple[str, str], int] = {}
    for lang, subject in sorted(lang_subject_pairs):
        path = TOPICS_FILE_BY_LANG_SUBJECT.get((lang, subject))
        if path is None:
            print(
                f"  ! no topics CSV registered for (language={lang!r}, "
                f"subject={subject!r})", file=sys.stderr,
            )
            continue
        if not path.exists():
            print(f"  ! topics file not found: {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        for _, row in df.iterrows():
            doc_ids = str(row["doc_ids"]) if pd.notna(row["doc_ids"]) else ""
            n_docs = len([d for d in doc_ids.split(",") if d.strip()])
            index[(lang, row["quiz_title"])] = n_docs
    return index


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def parse_cases(raw: list[dict]) -> tuple[list[TestCase], list[tuple[int, str]]]:
    """Parse all entries, collecting (index, error_msg) for invalid ones."""
    valid: list[TestCase] = []
    errors: list[tuple[int, str]] = []
    for i, entry in enumerate(raw):
        try:
            valid.append(TestCase(**entry))
        except ValidationError as e:
            errors.append((i, str(e).replace("\n", " | ")))
    return valid, errors


def cross_check(
    cases: list[TestCase], topic_index: dict[tuple[str, str], int]
) -> tuple[list[tuple[int, TestCase]], list[tuple[int, TestCase, int]]]:
    """Return (missing_titles, recall_capped) lists.

    missing_titles: cases whose (language, target_quiz_title) doesn't match
        any row in the relevant topics CSV. These are hard failures.

    recall_capped: cases where the ground-truth set is smaller than top_k.
        Not failures — just warnings so the eval reader knows recall@k can't
        reach 1.0 there even on a perfect retriever.
    """
    missing: list[tuple[int, TestCase]] = []
    recall_capped: list[tuple[int, TestCase, int]] = []
    for i, c in enumerate(cases):
        n_docs = topic_index.get((c.language, c.target_quiz_title))
        if n_docs is None:
            missing.append((i, c))
            continue
        if n_docs < c.top_k:
            recall_capped.append((i, c, n_docs))
    return missing, recall_capped


def check_subject_consistency(cases: list[TestCase]) -> list[tuple[int, TestCase]]:
    """Flag (language, subject) pairs that aren't in LANG_TO_SUBJECTS."""
    bad: list[tuple[int, TestCase]] = []
    for i, c in enumerate(cases):
        expected_set = LANG_TO_SUBJECTS.get(c.language)
        if expected_set is not None and c.subject not in expected_set:
            bad.append((i, c))
    return bad


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _hr() -> str:
    return "-" * 70


def report(
    raw_count: int,
    cases: list[TestCase],
    schema_errors: list[tuple[int, str]],
    missing: list[tuple[int, TestCase]],
    recall_capped: list[tuple[int, TestCase, int]],
    subject_mismatches: list[tuple[int, TestCase]],
) -> None:
    print()
    print(_hr())
    print("Summary")
    print(_hr())
    print(f"  total entries:        {raw_count}")
    print(f"  schema-valid:         {len(cases)}")
    print(f"  schema errors:        {len(schema_errors)}")
    print(f"  missing target:       {len(missing)}")
    print(f"  subject mismatches:   {len(subject_mismatches)}")
    print(f"  recall-capped (warn): {len(recall_capped)}")
    print()

    if cases:
        print("By language:")
        for lang, n in sorted(Counter(c.language for c in cases).items()):
            print(f"  {lang}: {n}")
        print()
        print("By query_type:")
        for qt, n in Counter(c.query_type for c in cases).most_common():
            print(f"  {qt}: {n}")
        print()
        print("By top_k:")
        for k, n in sorted(Counter(c.top_k for c in cases).items()):
            print(f"  k={k}: {n}")
        print()

    if schema_errors:
        print(_hr())
        print(f"Schema errors ({len(schema_errors)}, showing first 10):")
        for i, msg in schema_errors[:10]:
            print(f"  [#{i}] {msg[:200]}")
        print()

    if subject_mismatches:
        print(_hr())
        print(f"Subject mismatches ({len(subject_mismatches)}, showing first 10):")
        for i, c in subject_mismatches[:10]:
            expected = sorted(LANG_TO_SUBJECTS.get(c.language, set())) or ["?"]
            print(
                f"  [#{i}] lang={c.language} subject={c.subject!r}, "
                f"expected one of {expected!r}"
            )
        print()

    if missing:
        # Group by (language, title) so the same typo is not reported many times.
        by_title: dict[tuple[str, str], int] = defaultdict(int)
        for _, c in missing:
            by_title[(c.language, c.target_quiz_title)] += 1
        print(_hr())
        print(
            f"Missing target_quiz_title ({len(missing)} cases across "
            f"{len(by_title)} unique titles, showing first 20):"
        )
        for (lang, title), n in sorted(
            by_title.items(), key=lambda kv: -kv[1]
        )[:20]:
            print(f"  [{lang}] {n:>4}x  {title!r}")
        print()
        print(
            "  Most common cause: diacritics / accents / spacing differ from"
            " the verbatim title in topics_<lang>.csv."
        )
        print()

    if recall_capped:
        print(_hr())
        print(
            f"Recall-capped warnings ({len(recall_capped)}, showing first 10):"
        )
        for i, c, n_docs in recall_capped[:10]:
            print(
                f"  [#{i}] lang={c.language} top_k={c.top_k} but only {n_docs}"
                f" relevant docs for title {c.target_quiz_title!r}"
            )
        print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def validate(path: Path) -> int:
    """Returns process exit code (0 success, 1 any failure)."""
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    print(f"Loading {path} ...")
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        print(
            "error: top-level JSON must be a list of test-case objects.",
            file=sys.stderr,
        )
        return 1

    print(f"  parsed {len(raw)} entries")

    cases, schema_errors = parse_cases(raw)

    # Load topic index only for (language, subject) pairs that appear in the file.
    # Subjects matter because a single language can host multiple subjects
    # (e.g., fr → FRENCH and MATHEMATICS), each with its own topics CSV.
    lang_subject_pairs = {(c.language, c.subject) for c in cases}
    print(f"  loading topics for: {sorted(lang_subject_pairs)}")
    topic_index = load_topic_index(lang_subject_pairs)
    print(f"  topic index: {len(topic_index)} (language, title) pairs")

    missing, recall_capped = cross_check(cases, topic_index)
    subject_mismatches = check_subject_consistency(cases)

    report(
        raw_count=len(raw),
        cases=cases,
        schema_errors=schema_errors,
        missing=missing,
        recall_capped=recall_capped,
        subject_mismatches=subject_mismatches,
    )

    failed = bool(schema_errors or missing or subject_mismatches)
    if failed:
        print("FAIL — fix the problems above before running the eval.")
        return 1
    print("PASS — all test cases are valid.")
    if recall_capped:
        print(f"      ({len(recall_capped)} recall-capped warnings, see above)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in {"-h", "--help"}:
        print(__doc__)
        return 0 if args else 2
    return validate(Path(args[0]))


if __name__ == "__main__":
    raise SystemExit(main())
