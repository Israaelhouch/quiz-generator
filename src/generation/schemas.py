"""Pydantic schemas for Stage 5 generation.

Three models:
  - GenerationRequest     — caller input
  - GeneratedQuestion     — one LLM-produced question (validated)
  - GeneratedQuiz         — the full response (a collection)

Validators enforce the critical invariants the prompt asks the LLM for:
  - correct_answers is a non-empty list
  - for MCQ/TMC, every correct_answer exists verbatim in choices
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SUPPORTED_LANGUAGES = Literal["en", "fr", "ar"]
SUPPORTED_QUESTION_TYPES = Literal[
    "MULTIPLE_CHOICE",
    "FILL_IN_THE_BLANKS",
]
DIFFICULTIES = Literal["easy", "medium", "hard"]


class GenerationRequest(BaseModel):
    """Input to the generator (v1 — single type, single language)."""

    model_config = ConfigDict(extra="forbid")

    topic: str                                 # free-text query for retrieval
    language: SUPPORTED_LANGUAGES
    count: int = Field(ge=1, le=20)            # how many new questions to generate
    question_type: SUPPORTED_QUESTION_TYPES = "MULTIPLE_CHOICE"
    subject: str | None = None                 # optional retrieval filter
    level: str | None = None                   # optional retrieval filter
    few_shot_count: int = Field(default=6, ge=1, le=10)
    temperature: float = Field(default=0.75, ge=0.0, le=2.0)


class GeneratedQuestion(BaseModel):
    """One LLM-produced question, validated."""

    model_config = ConfigDict(extra="forbid")

    question_type: SUPPORTED_QUESTION_TYPES
    question_text: str
    choices: list[str] = Field(default_factory=list)
    correct_answers: list[str]
    multiple_correct_answers: bool = False              # True only when >1 correct answer
    explanation: str = ""
    difficulty: DIFFICULTIES | None = None

    @model_validator(mode="after")
    def _check_answers_shape(self) -> "GeneratedQuestion":
        if not self.question_text.strip():
            raise ValueError("question_text must be non-empty")
        if not self.correct_answers:
            raise ValueError("at least one correct_answer is required")

        if self.question_type == "MULTIPLE_CHOICE":
            if not self.choices:
                raise ValueError(
                    f"choices required for {self.question_type}"
                )
            for ans in self.correct_answers:
                if ans not in self.choices:
                    raise ValueError(
                        f"correct_answer {ans!r} not found verbatim in choices"
                    )
        elif self.question_type == "FILL_IN_THE_BLANKS":
            if self.choices:
                # Many FITB rows in real data have empty choices; some LLMs add them
                # Allow it but note: not validated strictly here.
                pass

        # `multiple_correct_answers` is redundant with len(correct_answers) > 1,
        # so we DERIVE it from the answer count rather than trust the LLM.
        # This mirrors the trust rule from Stage 2a (we ignored source data's
        # `multipleChoice` and computed from `sum(isTrue) > 1`).
        self.multiple_correct_answers = len(self.correct_answers) > 1
        return self


class GeneratedQuiz(BaseModel):
    """A batch of questions from one generation run."""

    model_config = ConfigDict(extra="forbid")

    language: SUPPORTED_LANGUAGES
    subject: str | None = None
    level: str | None = None
    questions: list[GeneratedQuestion]
