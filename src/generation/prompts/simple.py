"""Multilingual prompt builder.

Single entry point: `build_prompt(language=..., question_type=..., ...)`.
All language-specific phrases live in `strings.py`. This module is the
template glue — interpolation, ordering, and JSON schema block.

Backward-compatible aliases kept so older code/tests calling
`build_prompt_english` and `build_mcq_prompt_english` still work.
"""

from __future__ import annotations

from src.generation.prompts.strings import STRINGS, SUPPORTED_LANGUAGES
from src.retrieval.schemas import RetrievedQuestion


# JSON schema block — same in every language (only field NAMES are universal;
# the values inside are produced in the target language by the LLM).
OUTPUT_SCHEMA_BLOCK = (
    "{\n"
    '  "questions": [\n'
    "    {\n"
    '      "question_text": "...",\n'
    '      "choices": [...],\n'
    '      "correct_answers": [...],\n'
    '      "explanation": "...",\n'
    '      "difficulty": "easy" | "medium" | "hard"\n'
    "    }\n"
    "  ]\n"
    "}"
)


SUPPORTED_QUESTION_TYPES = ("MULTIPLE_CHOICE", "FILL_IN_THE_BLANKS")


def _render_example(index: int, example: RetrievedQuestion) -> str:
    return f"Example {index}:\n{example.to_prompt_block(include_answers=True)}"


def build_prompt(
    *,
    language: str,
    question_type: str,
    topic: str,
    count: int,
    examples: list[RetrievedQuestion],
    subject: str | None = None,
    level: str | None = None,
) -> tuple[str, str]:
    """Return (system_message, user_message) in the target language.

    Dispatches by `language` (en / fr / ar) and `question_type`
    (MULTIPLE_CHOICE / FILL_IN_THE_BLANKS).
    """
    if language not in STRINGS:
        raise ValueError(
            f"Unsupported language {language!r}. Supported: {list(SUPPORTED_LANGUAGES)}"
        )
    if question_type not in SUPPORTED_QUESTION_TYPES:
        raise ValueError(
            f"Unsupported question_type {question_type!r}. "
            f"Supported: {list(SUPPORTED_QUESTION_TYPES)}"
        )

    s = STRINGS[language]
    type_display = s["type_display"][question_type]
    rules = s[f"rules_{question_type}"]
    schema_hint = s[f"schema_hint_{question_type}"]

    # Build the user message piece by piece.
    task_line = s["task_template"].format(
        count=count, type_display=type_display, topic=topic
    )
    subject_line = (
        "\n" + s["subject_line_template"].format(subject=subject) if subject else ""
    )
    level_line = (
        "\n" + s["level_line_template"].format(level=level) if level else ""
    )
    examples_header = s["examples_header_template"].format(n=len(examples))
    examples_text = "\n\n".join(
        _render_example(i + 1, ex) for i, ex in enumerate(examples)
    )
    context_filter = s["context_filter_template"].format(topic=topic)
    concept_anchor = s["concept_anchor_template"].format(topic=topic)
    final_instruction = s["final_instruction_template"].format(count=count)

    user_message = (
        f"{task_line}{subject_line}{level_line}\n\n"
        f"{examples_header}\n\n"
        f"{examples_text}\n\n"
        f"{context_filter}\n\n"
        f"{concept_anchor}\n\n"
        f"{rules}\n\n"
        f"{s['ignore_inline_warning']}\n\n"
        f"{s['output_format_header']}\n"
        f"{OUTPUT_SCHEMA_BLOCK}\n\n"
        f"NOTE: {schema_hint}\n\n"
        f"{final_instruction}"
    )

    return s["system_message"], user_message


# ---------------------------------------------------------------------------
# Backward-compatible aliases — kept so older callers/tests still work.
# ---------------------------------------------------------------------------

def build_prompt_english(
    *,
    question_type: str,
    topic: str,
    count: int,
    examples: list[RetrievedQuestion],
    subject: str | None = None,
    level: str | None = None,
) -> tuple[str, str]:
    """Legacy helper — dispatches to build_prompt with language='en'."""
    return build_prompt(
        language="en",
        question_type=question_type,
        topic=topic,
        count=count,
        examples=examples,
        subject=subject,
        level=level,
    )


def build_mcq_prompt_english(
    *,
    topic: str,
    count: int,
    examples: list[RetrievedQuestion],
    subject: str | None = None,
    level: str | None = None,
) -> tuple[str, str]:
    """Legacy helper — English MCQ shortcut."""
    return build_prompt(
        language="en",
        question_type="MULTIPLE_CHOICE",
        topic=topic,
        count=count,
        examples=examples,
        subject=subject,
        level=level,
    )
