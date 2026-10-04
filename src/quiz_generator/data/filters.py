"""Structural filters — the rules that decide a question is unusable.

Scope filtering (which subjects and levels this project covers) lives in
`scope.py`; this module is about rows that cannot be used at all: no choices,
no correct answer, nothing but an image.

Pure functions over plain dicts, so they are testable without instantiating
Pydantic models. The ingest orchestrator wraps them with validated models.
"""

from __future__ import annotations

import html
import re


HTML_TAG_RE = re.compile(r"<[^>]+>")

# The question types the raw export may carry. Mirrors RAW_QUESTION_TYPES in
# shared/schemas.py, which cannot be imported here without pulling Pydantic
# into a deliberately dependency-free module.
ALLOWED_RAW_QUESTION_TYPES = frozenset(
    {"MULTIPLE_CHOICE", "FILL_IN_THE_BLANKS", "TEXT_MULTIPLE_CHOICE"}
)


def strip_html_to_plain(text: str | None) -> str:
    """Decode HTML entities, strip tags, collapse whitespace."""
    if not text:
        return ""
    decoded = html.unescape(text)
    no_tags = HTML_TAG_RE.sub(" ", decoded)
    return " ".join(no_tags.split()).strip()


def has_correct_answer(choices: list[dict]) -> bool:
    """True if at least one choice is marked isTrue and has content."""
    for choice in choices or []:
        if not choice.get("isTrue"):
            continue
        if (choice.get("answer") or "").strip() or choice.get("media"):
            return True
    return False


def is_image_only(description: str | None, image: str | None) -> bool:
    """A question is 'image-only' when it has an image and no visible text."""
    if not image:
        return False
    return not strip_html_to_plain(description)


def count_correct(choices: list[dict]) -> int:
    return sum(1 for choice in (choices or []) if choice.get("isTrue"))


def derive_multiple_correct_answers(choices: list[dict]) -> bool:
    """Our trust rule: multiple_correct_answers is derived, not trusted from the source."""
    return count_correct(choices) > 1


def decide_drop(question: dict) -> tuple[bool, str]:
    """Return (should_drop, reason). Reason is empty when keeping the row.

    Order of checks matters — we report the *first* failing rule.
    """
    choices = question.get("choices") or []

    if not choices:
        return True, "empty_choices"

    # A guard for callers that pass unvalidated dicts. Ingest never reaches it:
    # RawQuestion's Literal rejects an unknown type first, and the row is
    # counted as a validation failure instead. Kept in sync with that Literal
    # by test_question_types_match_the_schema.
    qtype = question.get("type")
    if qtype not in ALLOWED_RAW_QUESTION_TYPES:
        return True, "invalid_type"

    if not has_correct_answer(choices):
        return True, "no_correct_answer"

    if is_image_only(question.get("description"), question.get("image")):
        return True, "image_only"

    return False, ""


def doc_id_suffix(order: int, occurrence: int) -> str:
    """Build the trailing part of a question's doc_id.

    The doc_id is `{quiz_id}__{suffix}`. We can't trust `order` alone to
    be unique within a quiz — the source corpus has ~20% of quizzes where
    two or more questions share the same `order` value, which silently
    causes Chroma to overwrite rows during indexing.

    Disambiguation rule:
      - First occurrence of a given `order` in a quiz stays as `q{order}`
        (backward-compatible with every doc_id already referenced in eval
        ground truth and downstream artifacts).
      - Subsequent occurrences get `_2`, `_3`, ... appended.

    Args:
        order: the `order` field of the raw question.
        occurrence: zero-indexed count of how many times this `order` has
            already been emitted for the current quiz. 0 for first, 1 for
            the second, etc.
    """
    if occurrence == 0:
        return f"q{order}"
    return f"q{order}_{occurrence + 1}"
