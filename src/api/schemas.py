"""Pydantic request/response schemas for the API layer.

Kept separate from `src.generation.schemas` so the HTTP surface can evolve
independently from the internal data shapes (e.g. accept `levels` as a
list while the internal `GenerationRequest` still takes a single `level`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SUPPORTED_LANGUAGES = Literal["en", "fr", "ar"]
SUPPORTED_QUESTION_TYPES = Literal["MULTIPLE_CHOICE", "FILL_IN_THE_BLANKS"]


class GenerateRequest(BaseModel):
    """Body for POST /quiz/generate.

    Only fields the school platform / teacher legitimately decides per-call
    are exposed here. Tuning knobs (temperature, retry budget, few-shot
    count) live in `configs/models.yaml` — they're set once by the AI
    engineer, not by every caller.
    """

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=1, description="Free-text topic / query")
    language: SUPPORTED_LANGUAGES
    count: int = Field(default=5, ge=1, le=20,
                       description="How many new questions to generate")
    question_type: SUPPORTED_QUESTION_TYPES = "MULTIPLE_CHOICE"
    subject: str | None = Field(default=None, description="Optional retrieval filter")
    levels: list[str] | None = Field(
        default=None,
        description="Optional level filter (Tunisian curriculum tags)",
    )
    include_retrieval: bool = Field(
        default=False,
        description="If true, include the retrieved chunks in the response "
                    "(useful for the platform to debug bad outputs).",
    )


class RetrieveRequest(BaseModel):
    """Body for POST /retrieve. Retrieval-only, no LLM call."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    language: SUPPORTED_LANGUAGES
    top_k: int = Field(default=5, ge=1, le=50)
    question_type: SUPPORTED_QUESTION_TYPES | None = None
    subject: str | None = None
    levels: list[str] | None = None


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    error_code: str
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    pipeline_loaded: bool
