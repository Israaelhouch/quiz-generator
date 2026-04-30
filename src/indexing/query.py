"""Ad-hoc query tool for the Chroma vector store.

Quick way to sanity-test retrieval after running `src.indexing.build`.
This is NOT the production retriever — that lives in `src/retrieval/`
(Stage 4) and will add typed outputs, fallbacks, and quiz-title dedup.

Usage:
    # basic query
    python -m src.indexing.query "past tense verbs"

    # filter by language
    python -m src.indexing.query "dérivée" --language fr

    # filter by language + subject
    python -m src.indexing.query "primitives" --language fr --subject MATHEMATICS

    # return more results
    python -m src.indexing.query "photosynthesis" --top-k 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def query_store(
    *,
    config_path: Path,
    query_text: str,
    top_k: int = 5,
    language: str | None = None,
    question_type: str | None = None,
    subject: str | None = None,
    levels: list[str] | None = None,
    levels_match_mode: str = "any",      # "any" (OR) or "all" (AND)
    multiple_correct_answers: bool | None = None,
) -> list[dict]:
    """Run one query against the vector store. Return list of matches."""
    from src.indexing.config import load_models_config
    from src.indexing.embedding_model import EmbeddingModel, EmbeddingModelConfig
    from src.indexing.taxonomy import Taxonomy
    from src.indexing.vector_store import VectorStore, VectorStoreConfig

    config = load_models_config(config_path)

    # Load the persisted taxonomy for query-time validation.
    # Skipped silently if build_summary.json is absent — the store may
    # have been built by an older version of this code.
    summary_path = config.vector_store.persist_directory.parent / "build_summary.json"
    taxonomy = Taxonomy.from_build_summary(summary_path)
    taxonomy.validate_language(language)
    taxonomy.validate_question_type(question_type)
    if subject:
        taxonomy.validate_subject(subject.strip().upper())
    taxonomy.validate_levels(levels or [])

    embedding_config = EmbeddingModelConfig(
        name=config.model.name,
        embedding_dim=config.model.embedding_dim,
        batch_size=config.model.batch_size,
        device=config.model.device,
        normalize_embeddings=config.model.normalize_embeddings,
        passage_prefix=config.model.passage_prefix,
        query_prefix=config.model.query_prefix,
    )
    model = EmbeddingModel(embedding_config)

    store_config = VectorStoreConfig(
        persist_directory=config.vector_store.persist_directory,
        collection_name=config.vector_store.collection_name,
        distance_metric=config.vector_store.distance_metric,
        add_batch_size=config.vector_store.add_batch_size,
        reset_on_build=False,  # IMPORTANT: never wipe on query
    )
    store = VectorStore(store_config)
    store.open()

    # Build Chroma `where` clause for scalar filters.
    # - `subject` is a scalar metadata field (native pre-filter)
    # - `levels` are expanded to one boolean key per level at index time:
    #   a row with levels=["X","Y"] has metadata {"levels_X": True, "levels_Y": True}.
    #   Filter with `where={"levels_X": True}` (any-mode OR's them together).
    clauses: list[dict[str, Any]] = []
    if language:
        clauses.append({"language": language})
    if question_type:
        clauses.append({"question_type": question_type})
    if multiple_correct_answers is not None:
        clauses.append({"multiple_correct_answers": multiple_correct_answers})
    if subject:
        clauses.append({"subject": subject.strip().upper()})

    if levels:
        clean_levels = [str(l).strip() for l in levels if str(l).strip()]
        if clean_levels:
            level_clauses = [{f"levels_{lvl}": True} for lvl in clean_levels]
            if len(level_clauses) == 1:
                clauses.append(level_clauses[0])
            else:
                operator = "$and" if levels_match_mode == "all" else "$or"
                clauses.append({operator: level_clauses})

    where: dict[str, Any] | None = None
    if len(clauses) == 1:
        where = clauses[0]
    elif len(clauses) > 1:
        where = {"$and": clauses}

    q_vec = model.encode_query(query_text)
    results = store.collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=min(top_k, store.count()),
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    ids = (results.get("ids") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]

    output: list[dict] = []
    for _id, dist, meta, doc in zip(ids, distances, metadatas, documents):
        output.append(
            {
                "id": _id,
                "distance": dist,
                "metadata": meta,
                "document": doc,
            }
        )

    return output


def _print_results(
    *,
    query: str,
    language: str | None,
    question_type: str | None,
    subject: str | None,
    levels: list[str] | None,
    levels_match_mode: str,
    multiple_correct_answers: bool | None,
    results: list[dict],
) -> None:
    print(f"\nQuery   : {query!r}")
    filters = {
        "language": language,
        "question_type": question_type,
        "subject": subject,
        "levels": levels if levels else None,
        "levels_mode": levels_match_mode if levels else None,
        "multiple_correct_answers": multiple_correct_answers,
    }
    active_filters = {k: v for k, v in filters.items() if v is not None}
    print(f"Filters : {active_filters or '(none)'}")
    print(f"Results : {len(results)}")
    print("=" * 90)
    for i, r in enumerate(results, start=1):
        meta = r["metadata"] or {}
        # Reconstruct the levels list from boolean keys
        row_levels = sorted(k.removeprefix("levels_") for k in meta if k.startswith("levels_"))
        print(f"\n[{i}]  id={r['id'][:36]}")
        print(f"     distance = {r['distance']:+.4f}")
        print(f"     lang={meta.get('language'):2s}  type={meta.get('question_type')}  mcq={meta.get('multiple_correct_answers')}")
        print(f"     subject={meta.get('subject') or '(none)':<20s}  levels={row_levels[:2]}{'...' if len(row_levels) > 2 else ''}")
        print(f"     quiz_title: {meta.get('quiz_title', '')[:80]}")
        print(f"     document  : {r['document'][:220]}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "query",
        nargs="?",
        help="Query text (quote if it has spaces). Omit when using --list-taxonomy.",
    )
    p.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    p.add_argument(
        "--list-taxonomy",
        action="store_true",
        help="Print the known taxonomy (levels, subjects, languages, question_types) and exit.",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--language", choices=["en", "fr", "ar"], help="Filter by language")
    p.add_argument(
        "--question-type",
        choices=["MULTIPLE_CHOICE", "FILL_IN_THE_BLANKS"],
        help="Filter by question type",
    )
    p.add_argument("--subject", help="Filter by subject (exact match, case-insensitive)")
    p.add_argument(
        "--levels",
        help="Comma-separated levels to filter by (e.g. HIGH_SCHOOL_4TH_GRADE_MATHEMATICS)",
    )
    p.add_argument(
        "--levels-match",
        choices=["any", "all"],
        default="any",
        help="Match any of / all of the given levels (default: any)",
    )
    p.add_argument(
        "--multiple-correct-answers",
        choices=["true", "false"],
        help="Filter by multiple_correct_answers flag",
    )
    return p.parse_args()


def _print_taxonomy(config_path: Path) -> None:
    from src.indexing.config import load_models_config
    from src.indexing.taxonomy import Taxonomy

    config = load_models_config(config_path)
    summary_path = config.vector_store.persist_directory.parent / "build_summary.json"
    taxonomy = Taxonomy.from_build_summary(summary_path)
    if taxonomy.is_empty():
        print(f"No taxonomy found at {summary_path}. Did you run `python -m src.indexing.build`?")
        return
    print(f"Taxonomy from {summary_path}:")
    print(f"\n  Languages ({len(taxonomy.languages)}):")
    for v in taxonomy.list_languages():
        print(f"    {v}")
    print(f"\n  Question types ({len(taxonomy.question_types)}):")
    for v in taxonomy.list_question_types():
        print(f"    {v}")
    print(f"\n  Subjects ({len(taxonomy.subjects)}):")
    for v in taxonomy.list_subjects():
        print(f"    {v}")
    print(f"\n  Levels ({len(taxonomy.levels)}):")
    for v in taxonomy.list_levels():
        print(f"    {v}")


def main() -> None:
    args = _parse_args()

    if args.list_taxonomy:
        _print_taxonomy(args.config)
        return

    if not args.query:
        raise SystemExit("Query text is required (or use --list-taxonomy).")

    mc: bool | None = None
    if args.multiple_correct_answers:
        mc = args.multiple_correct_answers == "true"

    levels: list[str] | None = None
    if args.levels:
        levels = [l.strip() for l in args.levels.split(",") if l.strip()]

    results = query_store(
        config_path=args.config,
        query_text=args.query,
        top_k=args.top_k,
        language=args.language,
        question_type=args.question_type,
        subject=args.subject,
        levels=levels,
        levels_match_mode=args.levels_match,
        multiple_correct_answers=mc,
    )
    _print_results(
        query=args.query,
        language=args.language,
        question_type=args.question_type,
        subject=args.subject,
        levels=levels,
        levels_match_mode=args.levels_match,
        multiple_correct_answers=mc,
        results=results,
    )


if __name__ == "__main__":
    main()
