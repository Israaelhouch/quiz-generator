"""Validate an LLM-generated retriever test-cases JSON before running eval.

Catches five classes of problem:

  1. Schema violations (missing field, wrong type, invalid enum value).
  2. `target_quiz_title` values that don't exist in the corresponding
     `eval/topics_<lang>.csv` — typically diacritic / accent / spacing typos
     from the generator LLM. These would silently get zero ground-truth docs
     and inflate "false negatives" in the eval.
  3. Cases where `top_k` exceeds the number of available ground-truth docs
     for that title — recall@k can never reach 1.0 there. Reported as
     warnings, not failures.
  4. Topics-CSV doc_ids that are not in the index, or not in the test's
     (language, subject) cell — the answer key names a question the
     retriever can never return.
  5. Questions in the index that belong to a topic but are missing from its
     doc_ids — retrieving them is scored as wrong. A topic owns its own
     title plus every title its listed doc_ids carry today, so hand-merged
     canonical topics are respected without a separate alias file.

Checks 4 and 5 read the index payload the eval retrieves from. Without them a
stale answer key passes: the English topics CSV was built on 2026-05-13, five
days before the doc_id collision fix added ids like `__q3_2`, and 205 questions
that belong to its topics were missing from it.

Exit code: 0 if every case and the answer key are valid, 1 otherwise.

Usage:
    python -m scripts.eval.validate_test_cases eval/english_retriever_test_cases.json
    python -m scripts.eval.validate_test_cases eval/arabic_retriever_test_cases.json \\
        --ready-jsonl data/processed/ready_phase1.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
    ("en", "ENGLISH"): EVAL_DIR / "topics_english.csv",
    ("ar", "ARABIC"): EVAL_DIR / "topics_arabic.csv",
    ("fr", "FRENCH"): EVAL_DIR / "topics_french.csv",
    ("ar", "MATHEMATICS"): EVAL_DIR / "topics_math_ar.csv",
    ("fr", "MATHEMATICS"): EVAL_DIR / "topics_math_fr.csv",
}

# The payload the retriever serves — same default as run_retriever_eval.
DEFAULT_READY_JSONL = Path("data/processed/ready_phase1.jsonl")


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
                f"  ! no topics CSV registered for (language={lang!r}, subject={subject!r})",
                file=sys.stderr,
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
# Ground truth vs the index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexDoc:
    """The fields of one indexed question that the ground-truth checks need."""

    language: str
    subject: str  # subjects[0]: the scalar the retriever pre-filters on
    quiz_title: str


@dataclass(frozen=True)
class GroundTruthIssue:
    """One doc_id on which a topics CSV and the index disagree."""

    cell: tuple[str, str]  # (language, subject)
    topic: str  # quiz_title as written in the topics CSV
    doc_id: str
    detail: str = ""  # actual cell for wrong-cell ids, matched title for unlisted docs


@dataclass
class GroundTruthProblems:
    """Every disagreement found between the topics CSVs and the index."""

    unknown_ids: list[GroundTruthIssue] = field(default_factory=list)
    wrong_cell_ids: list[GroundTruthIssue] = field(default_factory=list)
    unlisted_docs: list[GroundTruthIssue] = field(default_factory=list)

    @property
    def has_problems(self) -> bool:
        """True when any check found a disagreement."""
        return bool(self.unknown_ids or self.wrong_cell_ids or self.unlisted_docs)


def load_index(ready_path: Path) -> dict[str, IndexDoc]:
    """Read the payload JSONL the retriever serves into {doc_id: IndexDoc}."""
    index: dict[str, IndexDoc] = {}
    with ready_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            subjects = row.get("subjects") or []
            index[row["doc_id"]] = IndexDoc(
                language=str(row.get("language") or ""),
                subject=str(subjects[0]) if subjects else "",
                quiz_title=str(row.get("quiz_title") or ""),
            )
    return index


def load_topic_doc_ids(
    lang_subject_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, set[str]]]:
    """Return {(language, subject): {quiz_title: doc_ids}} from the topics CSVs."""
    topics: dict[tuple[str, str], dict[str, set[str]]] = {}
    for cell in sorted(lang_subject_pairs):
        path = TOPICS_FILE_BY_LANG_SUBJECT.get(cell)
        if path is None or not path.exists():
            continue  # already reported by load_topic_index
        df = pd.read_csv(path, encoding="utf-8-sig")
        by_title: dict[str, set[str]] = {}
        for _, row in df.iterrows():
            raw = str(row["doc_ids"]) if pd.notna(row["doc_ids"]) else ""
            by_title[str(row["quiz_title"])] = {d.strip() for d in raw.split(",") if d.strip()}
        topics[cell] = by_title
    return topics


def check_ground_truth_against_index(
    topics: dict[tuple[str, str], dict[str, set[str]]],
    index: dict[str, IndexDoc],
) -> GroundTruthProblems:
    """Compare every topic's doc_ids with the index the eval retrieves from.

    unknown_ids     a listed doc_id is not in the index at all.
    wrong_cell_ids  a listed doc_id exists outside this (language, subject)
                    cell, so the retriever's pre-filter can never return it.
    unlisted_docs   a question in this cell carries one of the topic's titles
                    but is not listed, so retrieving it is scored as wrong.
    """
    problems = GroundTruthProblems()
    for cell, by_title in sorted(topics.items()):
        in_cell_by_title: dict[str, set[str]] = defaultdict(set)
        for doc_id, doc in index.items():
            if (doc.language, doc.subject) == cell:
                in_cell_by_title[doc.quiz_title].add(doc_id)
        for topic, listed in sorted(by_title.items()):
            titles = {topic}
            for doc_id in sorted(listed):
                doc = index.get(doc_id)
                if doc is None:
                    problems.unknown_ids.append(GroundTruthIssue(cell, topic, doc_id))
                elif (doc.language, doc.subject) != cell:
                    actual = f"{doc.language} x {doc.subject}"
                    problems.wrong_cell_ids.append(GroundTruthIssue(cell, topic, doc_id, actual))
                else:
                    titles.add(doc.quiz_title)
            for title in sorted(titles):
                for doc_id in sorted(in_cell_by_title.get(title, set()) - listed):
                    problems.unlisted_docs.append(GroundTruthIssue(cell, topic, doc_id, title))
    return problems


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
            print(f"  [#{i}] lang={c.language} subject={c.subject!r}, expected one of {expected!r}")
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
        for (lang, title), n in sorted(by_title.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  [{lang}] {n:>4}x  {title!r}")
        print()
        print(
            "  Most common cause: diacritics / accents / spacing differ from"
            " the verbatim title in topics_<lang>.csv."
        )
        print()

    if recall_capped:
        print(_hr())
        print(f"Recall-capped warnings ({len(recall_capped)}, showing first 10):")
        for i, c, n_docs in recall_capped[:10]:
            print(
                f"  [#{i}] lang={c.language} top_k={c.top_k} but only {n_docs}"
                f" relevant docs for title {c.target_quiz_title!r}"
            )
        print()


def report_ground_truth(problems: GroundTruthProblems, ready_path: Path) -> None:
    """Print where the topics CSVs disagree with the index, grouped by topic."""
    print(_hr())
    print(f"Ground truth vs index ({ready_path})")
    print(_hr())
    print(f"  unknown doc_ids:       {len(problems.unknown_ids)}")
    print(f"  doc_ids in wrong cell: {len(problems.wrong_cell_ids)}")
    print(f"  unlisted questions:    {len(problems.unlisted_docs)}")
    print()
    sections = (
        ("Listed doc_ids missing from the index", problems.unknown_ids),
        ("Listed doc_ids outside their (language, subject) cell", problems.wrong_cell_ids),
        ("Questions that belong to a topic but are not listed", problems.unlisted_docs),
    )
    for label, issues in sections:
        if not issues:
            continue
        first: dict[tuple[tuple[str, str], str], GroundTruthIssue] = {}
        per_topic: Counter[tuple[tuple[str, str], str]] = Counter()
        for issue in issues:
            key = (issue.cell, issue.topic)
            per_topic[key] += 1
            first.setdefault(key, issue)
        print(f"{label} ({len(issues)} across {len(per_topic)} topics, showing first 15):")
        for (cell, topic), n in per_topic.most_common(15):
            example = first[(cell, topic)]
            detail = f" ({example.detail})" if example.detail else ""
            print(f"  [{cell[0]} x {cell[1]}] {n:>4}x  {topic!r}  e.g. {example.doc_id}{detail}")
        print()
    if problems.unlisted_docs:
        print("  Most common cause: the topics CSV predates a rebuild of the index.")
        print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def validate(path: Path, ready_path: Path = DEFAULT_READY_JSONL) -> int:
    """Validate test cases and their answer key; return the process exit code."""
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

    index_found = ready_path.exists()
    gt_problems = GroundTruthProblems()
    if index_found:
        print(f"Loading index {ready_path} ...")
        topics = load_topic_doc_ids(lang_subject_pairs)
        gt_problems = check_ground_truth_against_index(topics, load_index(ready_path))
        report_ground_truth(gt_problems, ready_path)
    else:
        print(
            f"error: index payload not found: {ready_path} — cannot verify the answer key.",
            file=sys.stderr,
        )

    failed = bool(
        schema_errors
        or missing
        or subject_mismatches
        or not index_found
        or gt_problems.has_problems
    )
    if failed:
        print("FAIL — fix the problems above before running the eval.")
        return 1
    print("PASS — test cases and answer key are valid.")
    if recall_capped:
        print(f"      ({len(recall_capped)} recall-capped warnings, see above)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in {"-h", "--help"}:
        print(__doc__)
        return 0 if args else 2
    ready_path = DEFAULT_READY_JSONL
    if "--ready-jsonl" in args:
        i = args.index("--ready-jsonl")
        if i + 1 >= len(args):
            print("error: --ready-jsonl needs a path", file=sys.stderr)
            return 2
        ready_path = Path(args[i + 1])
        del args[i : i + 2]
    if len(args) != 1:
        print("error: expected exactly one test-cases path", file=sys.stderr)
        return 2
    return validate(Path(args[0]), ready_path)


if __name__ == "__main__":
    raise SystemExit(main())
