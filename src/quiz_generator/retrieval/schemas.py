"""Output schemas for retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievedQuestion:
    doc_id: str
    quiz_id: str
    quiz_title: str
    language: str
    question_type: str
    question_text: str
    choices: list[str]
    correct_answers: list[str]
    subjects: list[str]
    levels: list[str]
    multiple_correct_answers: bool
    author_name: str | None
    author_email: str | None
    search_text: str
    metadata: dict[str, Any]
    distance: float

    def to_prompt_block(self, include_answers: bool = True) -> str:
        """Render this question as a compact prompt block for generation."""
        lines = [
            f"Type: {self.question_type}",
            f"Question: {self.question_text}",
        ]

        if self.question_type == "FILL_IN_THE_BLANKS":
            if include_answers and self.correct_answers:
                lines.append(f"Accepted answers: {' | '.join(self.correct_answers)}")
        else:
            if self.choices:
                lines.append(f"Choices: {' | '.join(self.choices)}")
            if include_answers and self.correct_answers:
                lines.append(f"Correct answer: {' | '.join(self.correct_answers)}")

        if self.subjects:
            lines.append(f"Subjects: {', '.join(self.subjects)}")
        if self.levels:
            lines.append(f"Levels: {', '.join(self.levels)}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_prompt_block()
