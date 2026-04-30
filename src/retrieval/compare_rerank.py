"""A/B compare retrieval *with* and *without* the reranker.

Runs the same query twice — once on the bi-encoder baseline (small pool,
no rerank) and once on the production path (wider pool + cross-encoder
rerank) — and prints a side-by-side diff so you can eyeball whether the
reranker is helping.

Usage:
    python -m src.retrieval.compare_rerank "primitives des fonctions" \\
        --language fr --subject MATHEMATICS --top-k 5

    # Verbose: show full question + choices for each row
    python -m src.retrieval.compare_rerank "past tense" --language en --verbose

Notes:
- Requires `reranker.enabled: true` in configs/models.yaml — otherwise
  there's nothing to compare against.
- "BEFORE" run uses the small candidate pool (production-equivalent) so
  this answers "should I ship with reranker on?" not "what's the rerank
  effect on a fixed pool?".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("query", help="Query text (quote if it has spaces)")
    p.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    p.add_argument("--ready", type=Path, default=Path("data/processed/ready.jsonl"))
    p.add_argument("--language", required=True, choices=["en", "fr", "ar"])
    p.add_argument("--top-k", type=int, default=5)
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
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show full question text + choices per row (default: 60-char preview)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _truncate(text: str, n: int) -> str:
    text = " ".join((text or "").split())  # collapse whitespace
    return text if len(text) <= n else text[: n - 1] + "…"


def _format_row(idx: int, item: Any, marker: str, verbose: bool) -> str:
    head = (
        f"  #{idx}  {item.doc_id[:10]:<10}  d={item.distance:+.3f}  "
        f"{marker:<7}"
    )
    title = _truncate(item.quiz_title or "(no title)", 35)
    if verbose:
        block = head + f"{title}\n"
        block += f"        Q: {_truncate(item.question_text, 200)}\n"
        for c in item.choices[:6]:
            mark = "*" if c in (item.correct_answers or []) else " "
            block += f"        {mark} {_truncate(c, 100)}\n"
        return block.rstrip()
    preview = _truncate(item.question_text, 60)
    return f"{head}{title} — {preview}"


def _diff_summary(before: list[Any], after: list[Any]) -> dict:
    before_ids = [x.doc_id for x in before]
    after_ids = [x.doc_id for x in after]
    new_ids = [d for d in after_ids if d not in before_ids]
    dropped_ids = [d for d in before_ids if d not in after_ids]
    moved = 0
    for after_pos, doc_id in enumerate(after_ids):
        if doc_id in before_ids and before_ids.index(doc_id) != after_pos:
            moved += 1
    return {
        "moved": moved,
        "new": new_ids,
        "dropped": dropped_ids,
        "before_ids": before_ids,
        "after_ids": after_ids,
    }


def _movement_marker(doc_id: str, after_pos: int, before_ids: list[str]) -> str:
    if doc_id not in before_ids:
        return "NEW"
    before_pos = before_ids.index(doc_id)
    if before_pos == after_pos:
        return "="
    delta = before_pos - after_pos  # positive = moved up
    if delta > 0:
        return f"↑{delta}"
    return f"↓{abs(delta)}"


def _render(
    *,
    query: str,
    filters: dict,
    before: list[Any],
    after: list[Any],
    top_k: int,
    verbose: bool,
) -> None:
    summary = _diff_summary(before, after)
    line = "=" * 90

    print()
    print(line)
    print(f"QUERY:    {query!r}")
    print(f"FILTERS:  {filters or '(none)'}")
    print(f"TOP-K:    {top_k}")
    print(line)

    print("\nBEFORE (bi-encoder only — small pool, no rerank):")
    if not before:
        print("  (no results)")
    for i, item in enumerate(before, start=1):
        # In "before" view, marker just shows where the row ends up after rerank.
        if item.doc_id in summary["after_ids"]:
            after_pos = summary["after_ids"].index(item.doc_id)
            delta = i - 1 - after_pos
            if delta == 0:
                marker = "(=)"
            elif delta > 0:
                marker = f"(→#{after_pos + 1})"  # rerank promoted it
            else:
                marker = f"(→#{after_pos + 1})"  # rerank demoted it
        else:
            marker = "(OUT)"
        print(_format_row(i, item, marker, verbose))

    print("\nAFTER (reranked — wider pool, cross-encoder reordered):")
    if not after:
        print("  (no results)")
    for i, item in enumerate(after, start=1):
        marker = _movement_marker(item.doc_id, i - 1, summary["before_ids"])
        print(_format_row(i, item, f"({marker})", verbose))

    print()
    print("CHANGES:")
    print(f"  Position changes  : {summary['moved']}")
    print(f"  New in top-{top_k}     : {len(summary['new'])}"
          + (f"  ({', '.join(d[:10] for d in summary['new'])})" if summary["new"] else ""))
    print(f"  Dropped from top-{top_k}: {len(summary['dropped'])}"
          + (f"  ({', '.join(d[:10] for d in summary['dropped'])})" if summary["dropped"] else ""))
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_filters(args: argparse.Namespace) -> dict:
    filters: dict = {"language": args.language}
    if args.subject:
        filters["subject"] = args.subject
    if args.question_type:
        filters["question_type"] = args.question_type
    if args.levels:
        filters["levels"] = args.levels
        filters["levels_mode"] = args.levels_match
    if args.multiple_correct_answers:
        filters["multiple_correct_answers"] = args.multiple_correct_answers
    if args.author_name:
        filters["author_name"] = args.author_name
    if args.quiz_title_contains:
        filters["quiz_title_contains"] = args.quiz_title_contains
    if args.max_distance is not None:
        filters["max_distance"] = args.max_distance
    return filters


def main() -> None:
    args = _parse_args()
    from src.retrieval.retriever import Retriever

    mc = None
    if args.multiple_correct_answers:
        mc = args.multiple_correct_answers == "true"
    levels = None
    if args.levels:
        levels = [l.strip() for l in args.levels.split(",") if l.strip()]

    print("Loading retriever (this also loads the reranker model on first run)…")
    retriever = Retriever(config_path=args.config, ready_jsonl_path=args.ready)

    if retriever._reranker is None:
        raise SystemExit(
            "Reranker is not enabled. Set reranker.enabled: true in "
            f"{args.config} to use this script."
        )

    common_kwargs = dict(
        language=args.language,
        top_k=args.top_k,
        max_distance=args.max_distance,
        question_type=args.question_type,
        multiple_correct_answers=mc,
        subject=args.subject,
        levels=levels,
        levels_match_mode=args.levels_match,
        author_name=args.author_name,
        quiz_title_contains=args.quiz_title_contains,
    )

    # AFTER: reranker on (production path).
    after = retriever.retrieve(args.query, **common_kwargs)

    # BEFORE: temporarily disable reranker → bi-encoder baseline with the
    # small production pool size. This is the apples-to-apples comparison
    # for a deploy decision ("ship with vs ship without").
    saved = retriever._reranker
    retriever._reranker = None
    try:
        before = retriever.retrieve(args.query, **common_kwargs)
    finally:
        retriever._reranker = saved

    _render(
        query=args.query,
        filters=_build_filters(args),
        before=before,
        after=after,
        top_k=args.top_k,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
