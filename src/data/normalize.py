"""Stage 2b — Normalization.

Transforms `data/interim/flat.jsonl` → `data/interim/normalized.jsonl`.

Operations (in order):
  1. Clean text (HTML strip, entity decode, whitespace collapse) on
     quiz_title, question_text, and each choice answer.
  2. Normalize language: map raw label through alias table; re-detect
     from content when raw is missing OR when content script contradicts
     the raw label with high confidence.
  3. Apply subject aliases from configs/subject_aliases.yaml.
  4. Drop rows with empty cleaned question_text.
  5. Drop rows whose final language is not en/fr/ar.
  6. Deduplicate by (language, question_type, question_text,
     sorted(choices_text)). For duplicate groups, union `subjects` and
     `levels`; keep first-seen for every other field.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.data.domain_rules import apply_subject_language_rule
from src.data.filters import strip_html_to_plain
from src.data.language import SUPPORTED_LANGUAGES, resolve_language
from src.data.latex import normalize_latex, strip_latex_for_detection


QUIZ_PREFIX_RE = re.compile(r"^\s*quiz\s*:\s*", re.IGNORECASE)
IMG_TAG_RE = re.compile(r"<img\b", re.IGNORECASE)


def classify_empty_text_reason(raw_text: str | None) -> str:
    """Decide why a post-strip empty question_text occurred.

    - 'description_is_image_only' when the raw description contained an <img> tag
    - 'empty_description' when the raw description had no visible content at all
    """
    if raw_text and IMG_TAG_RE.search(raw_text):
        return "description_is_image_only"
    return "empty_description"


def load_subject_aliases(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return {str(key): str(value) for key, value in data.items()}


def clean_quiz_title(raw_title: str | None) -> str:
    cleaned = strip_html_to_plain(raw_title)
    return QUIZ_PREFIX_RE.sub("", cleaned).strip()


def apply_subject_aliases(subjects: list[str], aliases: dict[str, str]) -> list[str]:
    """Map + dedup within the list, preserving order."""
    result: list[str] = []
    for subject in subjects or []:
        canonical = aliases.get(subject, subject)
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def split_choices(choices_raw: list[dict]) -> tuple[list[str], list[str], list[str | None]]:
    """Return (choices_text, correct_choices_text, choices_media).

    Deduplicates choices by cleaned text within a single question — covers the
    ~31 source rows where the same answer appeared multiple times (likely a
    copy-paste error during quiz authoring). Empty answer text is allowed
    through (we don't dedup empties; the ghost-row filter handles those).
    Correct answers are kept in sync — duplicates removed there too.
    """
    texts: list[str] = []
    correct: list[str] = []
    media: list[str | None] = []
    seen_texts: set[str] = set()
    for choice in choices_raw or []:
        answer = strip_html_to_plain(choice.get("answer"))
        media_value = choice.get("media")
        # Skip a repeat of the SAME non-empty answer (data-entry duplicate).
        if answer and answer in seen_texts:
            continue
        if answer:
            seen_texts.add(answer)
        texts.append(answer)
        media.append(media_value)
        if choice.get("isTrue") and (answer or media_value):
            if answer not in correct:    # also dedup correct list
                correct.append(answer)
    return texts, correct, media


def normalize_row(flat: dict, aliases: dict[str, str]) -> tuple[dict | None, str, dict]:
    """Return (normalized_dict, drop_reason, audit_info).

    When drop_reason == "" the normalized dict is valid.
    audit_info carries language-resolution lineage for stats.
    """
    question_text = strip_html_to_plain(flat.get("question_text_raw"))
    quiz_title = clean_quiz_title(flat.get("quiz_title_raw"))
    choices_text, correct_text, choices_media = split_choices(flat.get("choices_raw") or [])

    # Language detection sees LaTeX-free text AND combines quiz_title with
    # question_text. Two reasons:
    # 1) Math-heavy rows like "Les primitives de \(\cos^2(x)\)" have their
    #    French stopword signal drowned out by fake tokens (left, right, frac).
    #    Stripping LaTeX recovers the signal.
    # 2) The quiz_title (e.g., "Limites et comportement asymptotique") is often
    #    pure natural-language French/English and adds crucial signal when the
    #    question itself is math-heavy.
    # The STORED question_text is unaffected; normalization happens only for
    # the detection call.
    # For DETECTION we use a plain stripper (no `\to`→"to" substitutions that
    # would inject fake English stopwords). Embedding still uses the richer
    # normalize_latex elsewhere.
    text_for_detection = strip_latex_for_detection(question_text)
    title_for_detection = strip_latex_for_detection(quiz_title)
    combined_for_detection = " ".join(
        part for part in [title_for_detection.strip(), text_for_detection.strip()] if part
    )

    final_language, language_source = resolve_language(
        flat.get("language_raw"),
        combined_for_detection,
        title_for_detection,
    )

    # DOMAIN RULE: in our Tunisian corpus, only subject=ENGLISH has English
    # content. Scientific subjects (MATHEMATICS, PHYSICS, etc.) are never
    # in English. If the detector disagrees with this prior, trust the
    # subject-based rule — it's much stronger for this specific dataset.
    final_language_post, domain_rule = apply_subject_language_rule(
        subjects=flat.get("subjects") or [],
        detected_language=final_language,
        text_sample=combined_for_detection,
    )
    if domain_rule != "none":
        final_language = final_language_post
        language_source = f"{language_source}+{domain_rule}"

    audit = {
        "doc_id": flat.get("doc_id"),
        "raw_language": flat.get("language_raw"),
        "final_language": final_language,
        "language_source": language_source,
    }

    if not question_text:
        reason = classify_empty_text_reason(flat.get("question_text_raw"))
        return None, reason, audit

    if final_language not in SUPPORTED_LANGUAGES:
        return None, "unsupported_language", audit

    # Ghost-row filter: for MCQ/TMC rows, if every choice is empty after cleaning
    # (e.g. original source had choices=['','',''] — placeholder/lost-media row),
    # drop it. Stage 2a's empty_choices check catches zero-length lists; this
    # catches lists with entries that are all blank strings after HTML strip.
    question_type_val = str(flat.get("question_type") or "")
    # After Stage 2a's TMC→MCQ merge, only MCQ needs the empty-choices check.
    if question_type_val == "MULTIPLE_CHOICE":
        if not any(c and c.strip() for c in choices_text):
            return None, "all_choices_empty", audit
        # After choice dedup (in split_choices), check for "too few" — a
        # source row like choices=['2', '2'] dedups to ['2'], leaving an
        # MCQ with only one option, which isn't really an MCQ.
        non_empty_unique = [c for c in choices_text if c and c.strip()]
        if len(non_empty_unique) < 2:
            return None, "too_few_choices_after_dedup", audit

    subjects = apply_subject_aliases(flat.get("subjects") or [], aliases)

    normalized = {
        "doc_id": flat["doc_id"],
        "quiz_id": flat["quiz_id"],
        "quiz_title": quiz_title,
        "language": final_language,
        "subjects": subjects,
        "levels": list(flat.get("levels") or []),
        "question_type": flat["question_type"],
        "multiple_correct_answers": bool(flat["multiple_correct_answers"]),
        "question_text": question_text,
        "choices_text": choices_text,
        "correct_choices_text": correct_text,
        "choices_media": choices_media,
        "points": flat.get("points"),
        "time": flat.get("time"),
        "author_name": flat.get("author_name"),
        "author_email": flat.get("author_email"),
    }
    return normalized, "", audit


def dedup_key(row: dict) -> tuple[str, str, str, tuple[str, ...]]:
    language = row["language"]
    question_type = row["question_type"]
    question_text = row["question_text"].casefold().strip()
    choices_key = tuple(
        sorted(
            choice.casefold().strip()
            for choice in row["choices_text"]
            if choice and choice.strip()
        )
    )
    return (language, question_type, question_text, choices_key)


def _union_preserving_order(rows: list[dict], field: str) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for value in row.get(field, []) or []:
            if value and value not in seen:
                seen.append(value)
    return seen


def dedup_rows(rows: list[dict]) -> tuple[list[dict], int, int]:
    """Group by dedup_key; merge subjects + levels across duplicates.

    Returns (kept_rows, groups_with_duplicates, rows_dropped).
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[dedup_key(row)].append(row)

    kept: list[dict] = []
    groups_with_duplicates = 0
    rows_dropped = 0

    for group in buckets.values():
        if len(group) == 1:
            kept.append(group[0])
            continue

        groups_with_duplicates += 1
        rows_dropped += len(group) - 1

        merged = dict(group[0])  # first-seen wins for all fields
        merged["subjects"] = _union_preserving_order(group, "subjects")
        merged["levels"] = _union_preserving_order(group, "levels")
        kept.append(merged)

    return kept, groups_with_duplicates, rows_dropped


def normalize(
    *,
    input_path: Path,
    output_path: Path,
    stats_path: Path,
    aliases_path: Path,
):
    # Lazy import so unit tests on the helper functions can run without Pydantic.
    from pydantic import ValidationError
    from src.shared.schemas import NormalizedQuestion, NormalizeStats

    aliases = load_subject_aliases(aliases_path)

    dropped: Counter[str] = Counter()
    lang_corrections: Counter[str] = Counter()
    subjects_remapped_rows = 0
    normalized_rows: list[dict] = []
    input_rows_total = 0

    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            input_rows_total += 1
            flat = json.loads(line)
            normalized, reason, audit = normalize_row(flat, aliases)

            if reason:
                dropped[reason] += 1
                continue

            if any(subject in aliases for subject in (flat.get("subjects") or [])):
                subjects_remapped_rows += 1

            if audit["language_source"] != "raw":
                raw_label = audit["raw_language"] if audit["raw_language"] is not None else "null"
                final_label = audit["final_language"] or "unknown"
                lang_corrections[f"{raw_label}->{final_label}"] += 1

            normalized_rows.append(normalized)

    deduped_rows, duplicate_groups, duplicate_rows_dropped = dedup_rows(normalized_rows)
    dropped["duplicate_question"] = duplicate_rows_dropped

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for row in deduped_rows:
            try:
                validated = NormalizedQuestion.model_validate(row)
            except ValidationError as exc:
                dropped["schema_validation_failed"] += 1
                continue
            out.write(validated.model_dump_json() + "\n")

    by_language = Counter(row["language"] for row in deduped_rows)
    by_type = Counter(row["question_type"] for row in deduped_rows)

    stats = NormalizeStats(
        input_rows=input_rows_total,
        output_rows=len(deduped_rows),
        dropped=dict(dropped),
        language_corrections=dict(lang_corrections),
        by_language=dict(by_language),
        by_type=dict(by_type),
        subjects_remapped_rows=subjects_remapped_rows,
        duplicate_groups=duplicate_groups,
        duplicate_rows_dropped=duplicate_rows_dropped,
    )
    stats_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/interim/flat.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/normalized.jsonl"))
    parser.add_argument("--stats", type=Path, default=Path("data/interim/normalized_stats.json"))
    parser.add_argument(
        "--aliases",
        type=Path,
        default=Path("configs/subject_aliases.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = normalize(
        input_path=args.input,
        output_path=args.output,
        stats_path=args.stats,
        aliases_path=args.aliases,
    )
    print(f"Input rows          : {stats.input_rows}")
    print(f"Output rows         : {stats.output_rows}")
    print(f"Dropped             : {dict(stats.dropped)}")
    print(f"By language         : {dict(stats.by_language)}")
    print(f"By type             : {dict(stats.by_type)}")
    print(f"Lang corrections    : {dict(stats.language_corrections)}")
    print(f"Subjects remapped   : {stats.subjects_remapped_rows} rows")
    print(f"Dedup duplicate grps: {stats.duplicate_groups} (dropped {stats.duplicate_rows_dropped} rows)")
    print(f"Output JSONL        : {args.output}")
    print(f"Stats JSON          : {args.stats}")


if __name__ == "__main__":
    main()
