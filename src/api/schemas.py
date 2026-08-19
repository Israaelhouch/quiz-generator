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
SUPPORTED_SCHOOL_PHASES = Literal["PRIMARY", "MIDDLE", "HIGH"]


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
    school_phase: SUPPORTED_SCHOOL_PHASES | None = Field(
        default=None,
        description=(
            "Optional coarse education-stage filter: PRIMARY / MIDDLE / HIGH. "
            "Pre-filters in Chroma natively (each indexed doc carries a "
            "school_phase metadata field derived from its first level tag). "
            "Cheaper and more ergonomic than enumerating every specific grade "
            "in `levels`."
        ),
    )
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
    """Body for POST /retrieve. Retrieval-only, no LLM call.

    By default this endpoint returns the raw retriever output (no distance
    filter applied) so engineers can inspect the full reranked list — useful
    for debugging "what would the retriever find" before production filters.

    Pass `max_distance` to apply the same quality floor /quiz/generate uses
    (sourced from configs/models.yaml: llm.default_max_distance). This lets
    the caller mirror what the LLM actually sees in production.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    language: SUPPORTED_LANGUAGES
    top_k: int = Field(default=5, ge=1, le=50)
    question_type: SUPPORTED_QUESTION_TYPES | None = None
    subject: str | None = None
    school_phase: SUPPORTED_SCHOOL_PHASES | None = Field(
        default=None,
        description=(
            "Optional coarse education-stage filter (PRIMARY / MIDDLE / HIGH). "
            "Pre-filters natively in Chroma — cheaper than enumerating specific "
            "grades in `levels`."
        ),
    )
    levels: list[str] | None = None
    max_distance: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "Optional cosine-distance cutoff. Docs farther than this are "
            "dropped. None (default) = no filter, raw debug view. Set to "
            "mirror /quiz/generate (e.g., 0.60)."
        ),
    )


class FeedbackRequest(BaseModel):
    """Body for POST /feedback — one human judgement about one question.

    Deliberately does NOT carry the retrieved chunks. `request_id` joins this
    row to its full entry in runs.jsonl, which already has the retrieval, the
    filters and the timings. Duplicating that here would double the storage
    and let the two copies drift.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["up", "down"]
    question_text: str = Field(..., min_length=1)
    question_index: int = Field(default=0, ge=0)
    request_id: str | None = Field(
        default=None,
        description="X-Request-Id of the /quiz/generate call this question came from.",
    )
    topic: str | None = None
    language: SUPPORTED_LANGUAGES | None = None
    subject: str | None = None
    school_phase: SUPPORTED_SCHOOL_PHASES | None = None
    note: str | None = Field(default=None, max_length=2000)


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    error_code: str
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    pipeline_loaded: bool
