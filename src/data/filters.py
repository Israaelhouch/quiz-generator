"""Stage 2a scope-filter logic.

Pure functions — operate on plain dicts so they are trivially testable
without instantiating Pydantic models. The ingest orchestrator wraps
these with validated models.
"""

from __future__ import annotations

import html
import re


HTML_TAG_RE = re.compile(r"<[^>]+>")


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

    qtype = question.get("type")
    if qtype not in {"MULTIPLE_CHOICE", "FILL_IN_THE_BLANKS", "TEXT_MULTIPLE_CHOICE"}:
        return True, "invalid_type"

    if not has_correct_answer(choices):
        return True, "no_correct_answer"

    if is_image_only(question.get("description"), question.get("image")):
        return True, "image_only"

    return False, ""
