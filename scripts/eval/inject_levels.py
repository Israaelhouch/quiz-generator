"""One-shot: inject a `levels` filter into each existing retriever test case.

Background
----------
The original test-case JSONs only carry `(query, language, subject,
target_quiz_title)`. They don't tell the retriever which education level
to filter on. That was fine while the corpus only held high-school +
middle-school content, but once primary-school was added (notably the
Arabic side, where primary expanded the corpus 2.6×) the retriever
started returning primary-school versions of the same grammar topics
that were intended to surface high-school content. Result: a ~14-point
MRR regression on Arabic in eval, where production wouldn't see it
because the platform passes `levels` based on the user's grade.

Phase-level scoping (not grade-level)
-------------------------------------
We deliberately scope test cases to the PHASE (PRIMARY_SCHOOL,
MIDDLE_SCHOOL, HIGH_SCHOOL), not the specific grade. Reasons:

1. The same topic (e.g. الاستثناء in Arabic grammar) can be tagged with
   HIGH_SCHOOL_1ST_GRADE on one quiz and HIGH_SCHOOL_2ND_GRADE_LETTRES
   on another. Locking each test case to a single grade would
   artificially shrink the candidate pool below what production wants.
2. The retriever's level filter is exact-match (one metadata key per
   level), so to express "any high-school grade" we expand the phase
   to its full list of grades and set `levels_match: "any"`. Same
   semantic outcome as a prefix match, no retriever changes needed.

We discover the grade list per phase by scanning the corpus
(`data/processed/ready_phase1.jsonl`) rather than hardcoding, so the
mapping stays correct as new grades enter the data.

Idempotent: re-running is safe — existing `levels` are overwritten with
the latest CSV+corpus-derived value.

Usage:
    python -m scripts.eval.inject_levels eval/arabic_retriever_test_cases.json
    python -m scripts.eval.inject_levels eval/english_retriever_test_cases.json
    python -m scripts.eval.inject_levels eval/french_retriever_test_cases.json

    # Or all three:
    python -m scripts.eval.inject_levels eval/*_retriever_test_cases.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


EVAL_DIR = Path("eval")
TOPICS_FILE_BY_LANG = {
    "en": EVAL_DIR / "topics_english.csv",
    "ar": EVAL_DIR / "topics_arabic.csv",
    "fr": EVAL_DIR / "topics_french.csv",
}

DEFAULT_CORPUS_PATH = Path("data/processed/ready_phase1.jsonl")

# Phase prefixes we care about. Order matters for `phase_of` — match the
# most specific first so PREPARATORY beats anything weirdly tagged.
PHASE_PREFIXES = (
    "PREPARATORY",
    "PRIMARY_SCHOOL",
    "MIDDLE_SCHOOL",
    "HIGH_SCHOOL",
    "LICENCE",
)


def phase_of(level: str) -> str | None:
    """Return the phase prefix for a specific level, or None if unknown."""
    for prefix in PHASE_PREFIXES:
        if level.startswith(prefix):
            return prefix
    return None


def build_phase_to_grades(corpus_path: Path) -> dict[str, list[str]]:
    """Scan the corpus and return {phase -> sorted list of specific grades}.

    Done from data rather than a hardcoded list so the eval stays in sync
    when new grades are added.
    """
    grades_by_phase: dict[str, set[str]] = {p: set() for p in PHASE_PREFIXES}
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for lvl in row.get("levels") or []:
                ph = phase_of(lvl)
                if ph is not None:
                    grades_by_phase[ph].add(lvl)
    return {p: sorted(gs) for p, gs in grades_by_phase.items() if gs}


def load_levels_map(language: str) -> dict[str, list[str]]:
    """Return {quiz_title -> [level1, level2, ...]} for one topics CSV."""
    path = TOPICS_FILE_BY_LANG[language]
    if not path.exists():
        raise FileNotFoundError(
            f"Topics CSV not found for language={language!r}: {path}. "
            "Build the topics CSVs first."
        )
    csv.field_size_limit(10_000_000)  # English doc_ids column is huge
    out: dict[str, list[str]] = {}
    # utf-8-sig strips an optional BOM from the first column name. The
    # French topics CSV ships with one and would otherwise produce a
    # column named "﻿quiz_title".
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row["quiz_title"].strip()
            levels_field = (row.get("levels") or "").strip()
            levels = [lv.strip() for lv in levels_field.split(",") if lv.strip()]
            out[title] = levels
    return out


def inject(
    path: Path,
    *,
    phase_to_grades: dict[str, list[str]],
    levels_match: str = "any",
    dry_run: bool = False,
) -> dict:
    """Add `levels` + `levels_match` to each case in the JSON at `path`.

    The level list written is the FULL set of corpus grades for every
    phase the target quiz_title touches. Example: a CSV row tagged with
    `HIGH_SCHOOL_2ND_GRADE_LETTRES,MIDDLE_SCHOOL_3RD_GRADE` expands to all
    HIGH_SCHOOL grades + all MIDDLE_SCHOOL grades. Combined with
    `levels_match: "any"`, this lets the retriever surface docs from any
    grade within the right phase(s) — the production-realistic scope.

    Returns a stats dict so callers can audit what was changed.
    """
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"{path}: expected top-level JSON array, got {type(cases).__name__}")

    # Group by language so we only load the CSV we need.
    languages = {c["language"] for c in cases}
    levels_maps = {lang: load_levels_map(lang) for lang in languages}

    n_total = len(cases)
    n_injected = 0
    n_skipped_unknown_title = 0
    n_skipped_unknown_phase = 0
    unknown_titles: Counter[str] = Counter()
    phases_used: Counter[str] = Counter()
    expanded_size_dist: Counter[int] = Counter()

    for c in cases:
        lang = c["language"]
        title = c["target_quiz_title"]
        csv_levels = levels_maps[lang].get(title)
        if csv_levels is None:
            n_skipped_unknown_title += 1
            unknown_titles[title] += 1
            continue

        # CSV gives specific grades; map to phases, dedup, expand to all
        # grades per phase, sort for deterministic output.
        phases: set[str] = set()
        for lvl in csv_levels:
            ph = phase_of(lvl)
            if ph is not None:
                phases.add(ph)
        if not phases:
            n_skipped_unknown_phase += 1
            continue

        expanded: list[str] = sorted({
            grade
            for ph in phases
            for grade in phase_to_grades.get(ph, [])
        })
        c["levels"] = expanded
        c["levels_match"] = levels_match
        n_injected += 1
        expanded_size_dist[len(expanded)] += 1
        for ph in phases:
            phases_used[ph] += 1

    if not dry_run:
        path.write_text(
            json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "path": str(path),
        "n_total": n_total,
        "n_injected": n_injected,
        "n_skipped_unknown_title": n_skipped_unknown_title,
        "n_skipped_unknown_phase": n_skipped_unknown_phase,
        "unknown_titles": dict(unknown_titles),
        "phases_used": dict(phases_used),
        "expanded_size_distribution": dict(expanded_size_dist),
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject `levels` filter into existing retriever test cases."
    )
    parser.add_argument(
        "test_cases",
        type=Path,
        nargs="+",
        help="One or more test-cases JSON paths.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"Path to the post-build payload JSONL (default: {DEFAULT_CORPUS_PATH}). "
             "Used to discover which specific grades exist per phase.",
    )
    parser.add_argument(
        "--levels-match",
        choices=["any", "all"],
        default="any",
        help="How the retriever should combine multiple levels. Default 'any' "
             "matches production behaviour for cross-tagged quizzes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats but don't write the files.",
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"[ERROR] corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    phase_to_grades = build_phase_to_grades(args.corpus)
    print("Phase → grade-list discovered from corpus:")
    for ph, gs in phase_to_grades.items():
        print(f"  {ph}: {len(gs)} grade(s)")
    if not phase_to_grades:
        print("[ERROR] no phases discovered in corpus", file=sys.stderr)
        return 2

    overall_ok = True
    for path in args.test_cases:
        try:
            stats = inject(
                path,
                phase_to_grades=phase_to_grades,
                levels_match=args.levels_match,
                dry_run=args.dry_run,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {path}: {e}", file=sys.stderr)
            overall_ok = False
            continue

        print(f"\n=== {path} ===")
        print(f"  total cases                  : {stats['n_total']}")
        print(f"  levels injected              : {stats['n_injected']}")
        print(f"  skipped (unknown title)      : {stats['n_skipped_unknown_title']}")
        print(f"  skipped (no recognized phase): {stats['n_skipped_unknown_phase']}")
        print(f"  phases used (case-count)     : {stats['phases_used']}")
        print(f"  expanded grade-count dist    : {stats['expanded_size_distribution']}")
        if stats["unknown_titles"]:
            print(f"  unknown target_quiz_titles:")
            for t, c in list(stats["unknown_titles"].items())[:5]:
                print(f"    x{c}  {t}")
        if stats["dry_run"]:
            print("  (dry-run — no file written)")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
