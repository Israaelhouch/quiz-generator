"""Ingestion — flatten the raw export, and drop most of it.

Flattens raw quizzes JSON into a JSONL of FlatQuestion rows
(`data/raw/quizzes-raw-data.json` → `data/interim/flat.jsonl`).

This stage removes far more than its name suggests: 5,829 of 12,480 questions
on the measured build of 2026-09-08. Two independent filters run here, and the
larger one is the scope filter, not the structural checks.

1. Scope filter — only when --scope is passed, from configs/scope.yaml
   via src/data/scope.py. Reasons are recorded with a `scope_` prefix. This is
   the big one: 4,901 rows.
     - scope_no_subjects            2,210   the quiz carries no subject at all
     - scope_subject_out_of_scope   1,417   CHEMISTRY, PHYSICS, COMPUTER_SCIENCE…
     - scope_level_out_of_scope     1,274   no level, or one outside the three
                                            school prefixes
   These rows are not bad data. They are questions in subjects and levels this
   project does not cover, and widening `configs/scope.yaml` brings them
   back with no code change.

2. Structural filters — always applied, from src/data/filters.py. 928 rows.
     - empty_choices, invalid_type, no_correct_answer, image_only

Language filtering is NOT applied here; it is deferred to the normalize module
after language normalization. Checking a language here would act on the raw
label, which is wrong for roughly 7% of rows.

Per-reason counts are written to the stats JSON beside the output.

Schema validation uses Pydantic v2 models at the stage boundaries.
The core filter logic (src/data/filters.py) is plain-dict and
independently testable without Pydantic.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from quiz_generator.ingestion.filters import (
    decide_drop,
    derive_multiple_correct_answers,
    doc_id_suffix,
)
from quiz_generator.ingestion.scope import decide_in_scope, load_scope
from quiz_generator.shared.hashing import sha256_of
from quiz_generator.shared.schemas import (
    FlatQuestion,
    IngestStats,
    RawQuestion,
    RawQuiz,
    ValidationFailure,
)

logger = logging.getLogger(__name__)

# How many validation failures to record in the stats file. Enough to debug a
# schema drift, few enough that a systematically broken export does not write
# a hundred-megabyte stats file.
VALIDATION_SAMPLE_LIMIT = 5


def load_raw_quizzes(path: Path) -> list[dict]:
    """Load the top-level JSON array. The file is ~271MB — loads in ~2s."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in {path}, got {type(data).__name__}")
    return data


def _flat_from_validated(
    quiz: RawQuiz,
    question: RawQuestion,
    *,
    doc_id_suffix: str,
) -> FlatQuestion:
    """Build a FlatQuestion from a validated raw question.

    `doc_id_suffix` is the trailing part of the doc_id (e.g. "q5" or
    "q5_2"). The caller is responsible for disambiguating colliding
    `question.order` values within the same quiz — see `flatten_quizzes`.
    Roughly 20% of quizzes in the source corpus contain duplicate `order`
    values, so we cannot trust that field alone to be unique.
    """
    choices_as_dicts = [choice.model_dump() for choice in question.choices]
    # Merge TEXT_MULTIPLE_CHOICE into MULTIPLE_CHOICE — structurally identical,
    # only 11 TMC rows in source corpus (<0.1%). Simpler 2-type downstream.
    normalized_type = question.type
    if normalized_type == "TEXT_MULTIPLE_CHOICE":
        normalized_type = "MULTIPLE_CHOICE"
    return FlatQuestion(
        doc_id=f"{quiz.id}__{doc_id_suffix}",
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


def flatten_quizzes(
    raw_quizzes: list[dict],
) -> Iterator[tuple[FlatQuestion | None, str, dict, ValidationFailure | None]]:
    """Yield one tuple per input question.

    Returns (flat_or_none, drop_reason, raw_question_dict, validation_failure).
    When drop_reason == "" the FlatQuestion is valid and should be written.
    raw_question_dict lets the caller record stats for dropped rows too, and
    validation_failure carries the validator's message when the row was
    rejected by a schema rather than by a filter.
    """
    for quiz_dict in raw_quizzes:
        try:
            quiz = RawQuiz.model_validate(quiz_dict)
        except ValidationError as exc:
            # A malformed quiz — all its questions are effectively skipped.
            # The validator's message travels with the row so the caller can
            # sample it; the quiz payload itself never does.
            failure = ValidationFailure(
                level="quiz", quiz_id=_quiz_id_of(quiz_dict), error=_first_error(exc)
            )
            for question_dict in quiz_dict.get("questions") or []:
                yield None, "quiz_validation_failed", question_dict, failure
            continue

        raw_question_dicts = quiz_dict.get("questions") or []
        # Per-quiz counter: `order` is NOT unique in ~20% of source quizzes,
        # so we track how many times each order has been seen and append a
        # disambiguator (_2, _3, …) to subsequent occurrences. The first
        # occurrence keeps the historical "q{order}" form to preserve
        # existing eval ground-truth doc_ids.
        order_seen: dict[int, int] = defaultdict(int)
        for question_dict, validated_question, error in _zip_question_validation(
            raw_question_dicts
        ):
            if validated_question is None:
                yield (
                    None,
                    "question_validation_failed",
                    question_dict,
                    ValidationFailure(level="question", quiz_id=quiz.id, error=error or ""),
                )
                continue

            drop, reason = decide_drop(question_dict)
            if drop:
                yield None, reason, question_dict, None
                continue

            occurrence = order_seen[validated_question.order]
            order_seen[validated_question.order] += 1
            suffix = doc_id_suffix(validated_question.order, occurrence)

            yield (
                _flat_from_validated(quiz, validated_question, doc_id_suffix=suffix),
                "",
                question_dict,
                None,
            )


def _zip_question_validation(
    question_dicts: list[dict],
) -> Iterator[tuple[dict, RawQuestion | None, str | None]]:
    for question_dict in question_dicts:
        try:
            yield question_dict, RawQuestion.model_validate(question_dict), None
        except ValidationError as exc:
            yield question_dict, None, _first_error(exc)


def _first_error(exc: ValidationError) -> str:
    """One line naming the field and the rule it broke. Never the value."""
    errors = exc.errors()
    if not errors:
        return "validation failed"
    first = errors[0]
    location = ".".join(str(part) for part in first["loc"]) or "(root)"
    return f"{location}: {first['msg']}"


def _quiz_id_of(quiz_dict: dict) -> str | None:
    """Best-effort id for a quiz that failed validation."""
    value = quiz_dict.get("_id") or quiz_dict.get("id")
    return str(value) if value is not None else None


def ingest(
    *,
    input_path: Path,
    output_path: Path,
    stats_path: Path,
    limit_quizzes: int | None = None,
    scope_path: Path | None = None,
) -> IngestStats:
    """Run ingest. Writes JSONL + stats JSON. Returns stats.

    When `scope_path` is provided, an additional scope filter is applied
    after the structural filters. Out-of-scope rows are dropped with a
    reason starting with `scope_*` recorded in stats.
    """
    raw_quizzes = load_raw_quizzes(input_path)
    if limit_quizzes is not None:
        raw_quizzes = raw_quizzes[:limit_quizzes]

    # Optional scope filter
    scope_cfg = None
    if scope_path is not None:
        scope_cfg = load_scope(scope_path)
        logger.info(
            "scope filter active: name=%s subjects=%d path=%s",
            scope_cfg.name,
            len(scope_cfg.subjects),
            scope_path,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    input_questions = 0
    output_rows = 0
    dropped: Counter[str] = Counter()
    by_language: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    quiz_validation_errors = 0
    question_validation_errors = 0
    failure_samples: list[ValidationFailure] = []

    # Written to a sibling temp file and renamed on success: a crash partway
    # through used to leave a truncated JSONL that the next stage reads as a
    # complete corpus.
    temp_path = output_path.with_name(output_path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as out:
        for flat, reason, _raw_q_dict, failure in flatten_quizzes(raw_quizzes):
            input_questions += 1

            if failure is not None and len(failure_samples) < VALIDATION_SAMPLE_LIMIT:
                failure_samples.append(failure)

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

            if flat is None:  # pragma: no cover - the generator's contract
                raise RuntimeError(
                    "flatten_quizzes yielded no row and no drop reason; "
                    "this is a bug in the generator, not in the data."
                )

            # Apply scope filter if configured. decide_in_scope reads only
            # `subjects` and `levels`, so pass those rather than model_dump()
            # serialising every choice of every question.
            if scope_cfg is not None:
                in_scope, scope_reason = decide_in_scope(
                    {"subjects": flat.subjects, "levels": flat.levels}, scope_cfg
                )
                if not in_scope:
                    dropped[f"scope_{scope_reason}"] += 1
                    continue

            out.write(flat.model_dump_json() + "\n")
            output_rows += 1
            by_language[str(flat.language_raw if flat.language_raw is not None else "null")] += 1
            by_type[flat.question_type] += 1

    temp_path.replace(output_path)

    stats = IngestStats(
        input_quizzes=len(raw_quizzes),
        input_questions=input_questions,
        output_rows=output_rows,
        dropped=dict(dropped),
        kept_by_language_raw=dict(by_language),
        kept_by_type=dict(by_type),
        quiz_validation_errors=quiz_validation_errors,
        question_validation_errors=question_validation_errors,
        scope_name=scope_cfg.name if scope_cfg is not None else None,
        scope_config_path=str(scope_path) if scope_path is not None else None,
        scope_config_sha256=sha256_of(scope_path) if scope_path is not None else None,
        validation_failure_samples=failure_samples,
    )
    stats_temp = stats_path.with_name(stats_path.name + ".tmp")
    stats_temp.write_text(stats.model_dump_json(indent=2), encoding="utf-8")
    stats_temp.replace(stats_path)

    logger.info(
        "ingest complete: %d/%d questions kept from %d quizzes (%d dropped)",
        output_rows,
        input_questions,
        len(raw_quizzes),
        sum(dropped.values()),
    )
    if quiz_validation_errors or question_validation_errors:
        logger.warning(
            "validation rejected rows: quiz=%d question=%d; first messages: %s",
            quiz_validation_errors,
            question_validation_errors,
            "; ".join(f.error for f in failure_samples) or "none recorded",
        )
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
    parser.add_argument(
        "--scope",
        type=Path,
        default=None,
        help="Optional path to a scope YAML (e.g. configs/scope.yaml). "
        "When set, rows outside the scope are dropped with reason 'scope_*'.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = ingest(
        input_path=args.input,
        output_path=args.output,
        stats_path=args.stats,
        limit_quizzes=args.limit_quizzes,
        scope_path=args.scope,
    )
    print(f"Input quizzes       : {stats.input_quizzes}")
    print(f"Input questions     : {stats.input_questions}")
    print(f"Kept rows           : {stats.output_rows}")
    print(f"Dropped             : {dict(stats.dropped)}")
    print(f"Kept by language    : {dict(stats.kept_by_language_raw)}")
    print(f"Kept by type        : {dict(stats.kept_by_type)}")
    print(
        f"Validation errors   : quiz={stats.quiz_validation_errors} question={stats.question_validation_errors}"
    )
    print(f"Output JSONL        : {args.output}")
    print(f"Stats JSON          : {args.stats}")


if __name__ == "__main__":
    main()
