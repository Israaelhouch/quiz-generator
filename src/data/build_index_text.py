"""Stage 2c — Build search_text.

Consumes `data/interim/normalized.jsonl`, produces `data/processed/ready.jsonl`.

For each row, composes a `search_text` string per a recipe configured in
`configs/pipeline.yaml`. The row is otherwise passed through unchanged.

Design notes:
  - search_text is model-agnostic. Model-specific prefixes (e.g. 'passage: '
    for e5 family) are added at embed time in Stage 3, not here — so we can
    swap embedding models without re-running 2c.
  - Recipe flags let us A/B test field combinations without code changes.
  - Stats report search_text length distribution so we can spot rows that
    exceed the embedder's context window (typical limit: 128 BPE tokens).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.data.latex import normalize_latex


DEFAULT_RECIPE_FLAGS = {
    "include_subjects": True,
    "include_quiz_title": True,
    "include_question": True,
    "include_choices": True,
    "include_correct_answers": False,
    "include_levels": False,
}

DEFAULT_SEPARATORS = {
    "part": ". ",
    "subjects": ", ",
    "choices": " | ",
}


def load_recipe(
    config_path: Path,
) -> tuple[str, dict[str, bool], dict[str, str], int, bool]:
    """Return (recipe_name, flags, separators, token_threshold, normalize_latex_flag)."""
    if not config_path.exists():
        return "default", dict(DEFAULT_RECIPE_FLAGS), dict(DEFAULT_SEPARATORS), 100, False

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    section = loaded.get("search_text") or {}
    recipe_name = str(section.get("recipe") or "default")
    recipes = section.get("recipes") or {}
    flags_from_config = recipes.get(recipe_name, {}) or {}

    flags = dict(DEFAULT_RECIPE_FLAGS)
    for key, value in flags_from_config.items():
        if key in flags:
            flags[key] = bool(value)

    separators = dict(DEFAULT_SEPARATORS)
    for key, value in (section.get("separators") or {}).items():
        if key in separators and isinstance(value, str):
            separators[key] = value

    token_threshold = int(section.get("token_warning_threshold", 100))
    normalize_latex_flag = bool(section.get("normalize_latex", False))
    return recipe_name, flags, separators, token_threshold, normalize_latex_flag


def _nonempty_strings(values: list[Any] | None) -> list[str]:
    if not values:
        return []
    return [str(value).strip() for value in values if value and str(value).strip()]


def compose_search_text(
    row: dict,
    *,
    flags: dict[str, bool],
    separators: dict[str, str],
    normalize_latex_flag: bool = False,
) -> str:
    """Compose the embedding-ready string from a normalized row.

    When normalize_latex_flag is True, LaTeX in question_text and
    choices_text is converted to plain math text BEFORE composition,
    so the embedder sees readable tokens instead of backslash gibberish.
    The original payload fields are not touched — only the search_text
    output of this function is normalized.
    """
    def _maybe_norm(text: str) -> str:
        return normalize_latex(text) if normalize_latex_flag else text

    parts: list[str] = []

    if flags["include_subjects"]:
        subjects = _nonempty_strings(row.get("subjects"))
        if subjects:
            parts.append(separators["subjects"].join(subjects))

    if flags["include_quiz_title"]:
        quiz_title = _maybe_norm(str(row.get("quiz_title", "")).strip())
        if quiz_title:
            parts.append(quiz_title)

    if flags["include_question"]:
        question_text = _maybe_norm(str(row.get("question_text", "")).strip())
        if question_text:
            parts.append(question_text)

    if flags["include_choices"]:
        choices = [_maybe_norm(c) for c in _nonempty_strings(row.get("choices_text"))]
        choices = [c for c in choices if c.strip()]
        if choices:
            parts.append(separators["choices"].join(choices))

    if flags["include_correct_answers"]:
        correct = [_maybe_norm(c) for c in _nonempty_strings(row.get("correct_choices_text"))]
        correct = [c for c in correct if c.strip()]
        if correct:
            parts.append(separators["choices"].join(correct))

    if flags["include_levels"]:
        levels = _nonempty_strings(row.get("levels"))
        if levels:
            parts.append(separators["subjects"].join(levels))

    return separators["part"].join(parts)


def _token_count(text: str) -> int:
    """Approximate token count via whitespace split.

    Not BPE-accurate, but sufficient for outlier detection.
    A real BPE count is typically 1.3–1.5x this value for Latin scripts.
    """
    return len(text.split()) if text else 0


def _summarize_lengths(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    sorted_lengths = sorted(lengths)
    p95_index = max(0, int(round(0.95 * (len(sorted_lengths) - 1))))
    return {
        "min": float(min(sorted_lengths)),
        "max": float(max(sorted_lengths)),
        "mean": round(statistics.mean(sorted_lengths), 2),
        "median": float(statistics.median(sorted_lengths)),
        "p95": float(sorted_lengths[p95_index]),
    }


def build_index_text(
    *,
    input_path: Path,
    output_path: Path,
    stats_path: Path,
    config_path: Path,
):
    # Lazy imports so helper tests run without Pydantic.
    from pydantic import ValidationError
    from src.shared.schemas import BuildIndexTextStats, IndexedQuestion

    recipe_name, flags, separators, token_threshold, normalize_latex_flag = load_recipe(config_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    input_rows = 0
    output_rows = 0
    lengths: list[int] = []
    rows_over_threshold = 0
    empty_search_text_rows = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            input_rows += 1
            row = json.loads(line)
            search_text = compose_search_text(
                row,
                flags=flags,
                separators=separators,
                normalize_latex_flag=normalize_latex_flag,
            )
            if not search_text:
                empty_search_text_rows += 1
                continue

            indexed = dict(row)
            indexed["search_text"] = search_text
            try:
                validated = IndexedQuestion.model_validate(indexed)
            except ValidationError:
                continue
            dst.write(validated.model_dump_json() + "\n")
            output_rows += 1

            token_count = _token_count(search_text)
            lengths.append(token_count)
            if token_count > token_threshold:
                rows_over_threshold += 1

    stats = BuildIndexTextStats(
        input_rows=input_rows,
        output_rows=output_rows,
        recipe=recipe_name,
        recipe_flags=flags,
        search_text_length_tokens=_summarize_lengths(lengths),
        rows_over_token_threshold=rows_over_threshold,
        token_threshold=token_threshold,
        empty_search_text_rows=empty_search_text_rows,
    )
    stats_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/interim/normalized.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/ready.jsonl"))
    parser.add_argument("--stats", type=Path, default=Path("data/processed/ready_stats.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = build_index_text(
        input_path=args.input,
        output_path=args.output,
        stats_path=args.stats,
        config_path=args.config,
    )
    print(f"Recipe              : {stats.recipe}")
    print(f"Recipe flags        : {stats.recipe_flags}")
    print(f"Input rows          : {stats.input_rows}")
    print(f"Output rows         : {stats.output_rows}")
    print(f"Empty search_text   : {stats.empty_search_text_rows}")
    print(f"Token length        : {stats.search_text_length_tokens}")
    print(f"Over {stats.token_threshold} tokens     : {stats.rows_over_token_threshold} rows")
    print(f"Output JSONL        : {args.output}")
    print(f"Stats JSON          : {args.stats}")


if __name__ == "__main__":
    main()
