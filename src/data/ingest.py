"""Stage 2a — Ingestion.

Flattens raw quizzes JSON into a JSONL of FlatQuestion rows.

Scope filters applied here (structural only):
  - empty_choices
  - invalid_type
  - no_correct_answer
  - image_only

Language filtering is NOT applied here; it is deferred to Stage 2b
after language normalization.

Schema validation uses Pydantic v2 models at the stage boundaries.
The core filter logic (src/data/filters.py) is plain-dict and
independently testable without Pydantic.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from src.data.filters import decide_drop, derive_multiple_correct_answers
from src.shared.schemas import (
    FlatQuestion,
    IngestStats,
    RawQuestion,
    RawQuiz,
)


def load_raw_quizzes(path: Path) -> list[dict]:
    """Load the top-level JSON array. The file is ~271MB — loads in ~2s."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in {path}, got {type(data).__name__}")
    return data


def _flat_from_validated(quiz: RawQuiz, question: RawQuestion) -> FlatQuestion:
    choices_as_dicts = [choice.model_dump() for choice in question.choices]
    # Merge TEXT_MULTIPLE_CHOICE into MULTIPLE_CHOICE — structurally identical,
    # only 11 TMC rows in source corpus (<0.1%). Simpler 2-type downstream.
    normalized_type = question.type
    if normalized_type == "TEXT_MULTIPLE_CHOICE":
        normalized_type = "MULTIPLE_CHOICE"
    return FlatQuestion(
        doc_id=f"{quiz.id}__q{question.order}",
        quiz_id=quiz.id,
        quiz_title_raw=quiz.title,
        language_raw=quiz.language,
        subjects=list(quiz.subjects),
        levels=list(quiz.levels),
        question_type=normalized_type,
        multiple_correct_answers=derive_multiple_correct_answers(choices_as_dicts),
        question_text_raw=question.description,
        choices_raw=list(question.choices),
        points=question.points,
        time=question.time,
        author_name=quiz.createdBy.name if quiz.createdBy else None,
        author_email=quiz.createdBy.email if quiz.createdBy else None,
    )


def flatten_quizzes(raw_quizzes: list[dict]) -> Iterator[tuple[FlatQuestion | None, str, dict]]:
    """Yield one triple per input question.

    Returns (flat_or_none, drop_reason, raw_question_dict).
    When drop_reason == "" the FlatQuestion is valid and should be written.
    raw_question_dict is included so the caller can record stats even for dropped rows.
    """
    for quiz_dict in raw_quizzes:
        try:
            quiz = RawQuiz.model_validate(quiz_dict)
        except ValidationError:
            # A malformed quiz — all its questions are effectively skipped.
            for question_dict in quiz_dict.get("questions") or []:
                yield None, "quiz_validation_failed", question_dict
            continue

        raw_question_dicts = quiz_dict.get("questions") or []
        for question_dict, validated_question in _zip_question_validation(raw_question_dicts, quiz):
            if validated_question is None:
                yield None, "question_validation_failed", question_dict
                continue

            drop, reason = decide_drop(question_dict)
            if drop:
                yield None, reason, question_dict
                continue

            yield _flat_from_validated(quiz, validated_question), "", question_dict


def _zip_question_validation(
    question_dicts: list[dict], quiz: RawQuiz
) -> Iterator[tuple[dict, RawQuestion | None]]:
    for question_dict in question_dicts:
        try:
            yield question_dict, RawQuestion.model_validate(question_dict)
        except ValidationError:
            yield question_dict, None


def ingest(
    *,
    input_path: Path,
    output_path: Path,
    stats_path: Path,
    limit_quizzes: int | None = None,
) -> IngestStats:
    """Run Stage 2a. Writes JSONL + stats JSON. Returns stats."""
    raw_quizzes = load_raw_quizzes(input_path)
    if limit_quizzes is not None:
        raw_quizzes = raw_quizzes[:limit_quizzes]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    input_questions = 0
    output_rows = 0
    dropped: Counter[str] = Counter()
    by_language: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    quiz_validation_errors = 0
    question_validation_errors = 0

    with output_path.open("w", encoding="utf-8") as out:
        for flat, reason, raw_q_dict in flatten_quizzes(raw_quizzes):
            input_questions += 1

            if reason == "quiz_validation_failed":
                quiz_validation_errors += 1
                dropped[reason] += 1
                continue
            if reason == "question_validation_failed":
                question_validation_errors += 1
                dropped[reason] += 1
                continue
            if reason:
                dropped[reason] += 1
                continue

            assert flat is not None  # drop_reason empty implies valid flat
            out.write(flat.model_dump_json() + "\n")
            output_rows += 1
            by_language[str(flat.language_raw if flat.language_raw is not None else "null")] += 1
            by_type[flat.question_type] += 1

    stats = IngestStats(
        input_quizzes=len(raw_quizzes),
        input_questions=input_questions,
        output_rows=output_rows,
        dropped=dict(dropped),
        kept_by_language_raw=dict(by_language),
        kept_by_type=dict(by_type),
        quiz_validation_errors=quiz_validation_errors,
        question_validation_errors=question_validation_errors,
    )
    stats_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/quizzes-raw-data.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/flat.jsonl"),
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("data/interim/flat_stats.json"),
    )
    parser.add_argument(
        "--limit-quizzes",
        type=int,
        help="Only process the first N quizzes (for debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = ingest(
        input_path=args.input,
        output_path=args.output,
        stats_path=args.stats,
        limit_quizzes=args.limit_quizzes,
    )
    print(f"Input quizzes       : {stats.input_quizzes}")
    print(f"Input questions     : {stats.input_questions}")
    print(f"Kept rows           : {stats.output_rows}")
    print(f"Dropped             : {dict(stats.dropped)}")
    print(f"Kept by language    : {dict(stats.kept_by_language_raw)}")
    print(f"Kept by type        : {dict(stats.kept_by_type)}")
    print(f"Validation errors   : quiz={stats.quiz_validation_errors} question={stats.question_validation_errors}")
    print(f"Output JSONL        : {args.output}")
    print(f"Stats JSON          : {args.stats}")


if __name__ == "__main__":
    main()
