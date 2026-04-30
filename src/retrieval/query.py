"""CLI wrapper around src.retrieval.Retriever.

Similar shape to src.indexing.query, but returns typed RetrievedQuestion
objects with full payload (choices + correct_answers). Use this when you
want to see what the LLM generator will actually receive.

Usage:
    python -m src.retrieval.query "dérivée" --language fr --top-k 5
    python -m src.retrieval.query "primitives" --language fr \\
        --subject MATHEMATICS --levels HIGH_SCHOOL_4TH_GRADE_MATHEMATICS
    python -m src.retrieval.query --list-taxonomy
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("query", nargs="?", help="Query text (quote if it has spaces)")
    p.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    p.add_argument("--ready", type=Path, default=Path("data/processed/ready.jsonl"))
    p.add_argument("--language", choices=["en", "fr", "ar"], help="REQUIRED for query")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidate-pool-size", type=int, default=50)
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument(
        "--question-type",
        choices=["MULTIPLE_CHOICE", "FILL_IN_THE_BLANKS"],
    )
    p.add_argument("--subject", help="Single subject (case-insensitive)")
    p.add_argument("--levels", help="Comma-separated levels")
    p.add_argument("--levels-match", choices=["any", "all"], default="any")
    p.add_argument("--multiple-correct-answers", choices=["true", "false"])
    p.add_argument("--author-name")
    p.add_argument("--quiz-title-contains")
    p.add_argument("--no-dedup", action="store_true", help="Disable quiz-title dedup")
    p.add_argument("--list-taxonomy", action="store_true", help="Print known taxonomy and exit")
    p.add_argument(
        "--hide-answers",
        action="store_true",
        help="Hide correct answers in the prompt block (default: shown)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Lazy import so --help works without loading the model.
    from src.retrieval.retriever import Retriever

    if args.list_taxonomy:
        # Minimal retriever init just to access taxonomy
        retriever = Retriever(config_path=args.config, ready_jsonl_path=args.ready)
        print(f"Languages ({len(retriever.list_languages())}): {retriever.list_languages()}")
        print(f"Question types ({len(retriever.list_question_types())}): {retriever.list_question_types()}")
        print(f"Subjects ({len(retriever.list_subjects())}):")
        for s in retriever.list_subjects():
            print(f"  {s}")
        print(f"Levels ({len(retriever.list_levels())}):")
        for l in retriever.list_levels():
            print(f"  {l}")
        return

    if not args.query:
        raise SystemExit("Query text is required (or use --list-taxonomy).")
    if not args.language:
        raise SystemExit("--language is required (en/fr/ar).")

    mc = None
    if args.multiple_correct_answers:
        mc = args.multiple_correct_answers == "true"
    levels = None
    if args.levels:
        levels = [l.strip() for l in args.levels.split(",") if l.strip()]

    retriever = Retriever(config_path=args.config, ready_jsonl_path=args.ready)
    results = retriever.retrieve(
        args.query,
        language=args.language,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        max_distance=args.max_distance,
        question_type=args.question_type,
        multiple_correct_answers=mc,
        subject=args.subject,
        levels=levels,
        levels_match_mode=args.levels_match,
        author_name=args.author_name,
        quiz_title_contains=args.quiz_title_contains,
        dedup_by_quiz_title=not args.no_dedup,
    )

    print(f"\nQuery    : {args.query!r}")
    print(f"Language : {args.language}")
    filters: dict = {}
    if args.subject:
        filters["subject"] = args.subject
    if args.question_type:
        filters["question_type"] = args.question_type
    if levels:
        filters["levels"] = levels
        filters["levels_mode"] = args.levels_match
    if mc is not None:
        filters["multiple_correct_answers"] = mc
    if args.author_name:
        filters["author_name"] = args.author_name
    if args.quiz_title_contains:
        filters["quiz_title_contains"] = args.quiz_title_contains
    if args.max_distance is not None:
        filters["max_distance"] = args.max_distance
    print(f"Filters  : {filters or '(none)'}")
    print(f"Results  : {len(results)}")
    print("=" * 90)
    include_answers = not args.hide_answers
    for i, r in enumerate(results, start=1):
        print(f"\n[{i}]  doc_id={r.doc_id[:40]}")
        print(f"     distance   = {r.distance:+.4f}")
        print(f"     quiz_title : {r.quiz_title}")
        print(f"     subjects   : {r.subjects}   levels[:2]: {r.levels[:2]}{'...' if len(r.levels) > 2 else ''}")
        print(f"     author     : {r.author_name}")
        print(r.to_prompt_block(include_answers=include_answers))


if __name__ == "__main__":
    main()
