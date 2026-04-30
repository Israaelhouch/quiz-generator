"""Pydantic v2 schemas for the quiz RAG pipeline.

Each stage in the pipeline consumes and emits validated models.
Stage 2a (ingest) produces FlatQuestion. Stage 2b will produce CleanedQuestion.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# Raw source data has three types, but TEXT_MULTIPLE_CHOICE is structurally
# identical to MULTIPLE_CHOICE (just longer-phrase choices) and only ~0.1% of
# the corpus. We merge TMC into MCQ at ingestion time. Downstream uses the
# narrower SUPPORTED_QUESTION_TYPES.
RAW_QUESTION_TYPES = Literal[
    "MULTIPLE_CHOICE",
    "FILL_IN_THE_BLANKS",
    "TEXT_MULTIPLE_CHOICE",
]

SUPPORTED_QUESTION_TYPES = Literal[
    "MULTIPLE_CHOICE",
    "FILL_IN_THE_BLANKS",
]


class RawChoice(BaseModel):
    """One entry inside question.choices in the raw JSON."""

    model_config = ConfigDict(extra="ignore")

    answer: str | None = None
    isTrue: bool = False
    media: str | None = None


class RawCreatedBy(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    email: str | None = None


class RawQuestion(BaseModel):
    """Tolerant parser for one question inside a raw quiz."""

    model_config = ConfigDict(extra="ignore")

    order: int
    type: RAW_QUESTION_TYPES   # raw source allows all 3; TMC is merged into MCQ at ingestion
    description: str | None = None
    image: str | None = None
    points: float | None = None
    time: int | None = None
    choices: list[RawChoice] = Field(default_factory=list)


class RawQuiz(BaseModel):
    """Tolerant parser for one top-level quiz object."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(alias="_id")
    title: str = ""
    language: str | None = None
    subjects: list[str] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list)
    createdBy: RawCreatedBy | None = None
    questions: list[RawQuestion] = Field(default_factory=list)


class FlatQuestion(BaseModel):
    """Stage 2a output — one row per kept question. NOT cleaned yet.

    Text fields are suffixed `_raw` to mark that HTML stripping and
    whitespace normalization happen in Stage 2b (normalize).
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    quiz_id: str
    quiz_title_raw: str
    language_raw: str | None
    subjects: list[str]
    levels: list[str]
    question_type: SUPPORTED_QUESTION_TYPES
    multiple_correct_answers: bool
    question_text_raw: str | None
    choices_raw: list[RawChoice]
    points: float | None
    time: int | None
    author_name: str | None
    author_email: str | None


class IngestStats(BaseModel):
    """Audit record written alongside flat.jsonl after every ingest run."""

    input_quizzes: int
    input_questions: int
    output_rows: int
    dropped: dict[str, int]
    kept_by_language_raw: dict[str, int]
    kept_by_type: dict[str, int]
    quiz_validation_errors: int = 0
    question_validation_errors: int = 0


SUPPORTED_LANGUAGES = Literal["en", "fr", "ar"]


class NormalizedQuestion(BaseModel):
    """Stage 2b output — one cleaned, language-normalized, dedup'd row.

    Compared to FlatQuestion:
      - HTML stripped, whitespace collapsed
      - `language` tightened to Literal["en", "fr", "ar"]
      - `choices_text` / `correct_choices_text` split out as plain strings
      - `choices_media` preserved alongside (indices align with choices_text)
      - Subject aliases applied
      - Duplicates merged: subjects + levels are unions across duplicate rows
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    quiz_id: str
    quiz_title: str
    language: SUPPORTED_LANGUAGES
    subjects: list[str]
    levels: list[str]
    question_type: SUPPORTED_QUESTION_TYPES
    multiple_correct_answers: bool
    question_text: str
    choices_text: list[str]
    correct_choices_text: list[str]
    choices_media: list[str | None]
    points: float | None
    time: int | None
    author_name: str | None
    author_email: str | None


class NormalizeStats(BaseModel):
    """Audit record written alongside normalized.jsonl."""

    input_rows: int
    output_rows: int
    dropped: dict[str, int]
    language_corrections: dict[str, int]
    by_language: dict[str, int]
    by_type: dict[str, int]
    subjects_remapped_rows: int
    duplicate_groups: int
    duplicate_rows_dropped: int


class IndexedQuestion(BaseModel):
    """Stage 2c output — the embedding-ready row.

    Identical to NormalizedQuestion plus `search_text` — the string that
    the embedding model will actually process. Model-specific prefixes
    (like 'passage: ' for e5) are added at embed time in Stage 3, not here,
    so search_text stays model-agnostic.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    quiz_id: str
    quiz_title: str
    language: SUPPORTED_LANGUAGES
    subjects: list[str]
    levels: list[str]
    question_type: SUPPORTED_QUESTION_TYPES
    multiple_correct_answers: bool
    question_text: str
    choices_text: list[str]
    correct_choices_text: list[str]
    choices_media: list[str | None]
    points: float | None
    time: int | None
    author_name: str | None
    author_email: str | None
    search_text: str


class BuildIndexTextStats(BaseModel):
    """Audit record written alongside ready.jsonl."""

    input_rows: int
    output_rows: int
    recipe: str
    recipe_flags: dict[str, bool]
    search_text_length_tokens: dict[str, float]
    rows_over_token_threshold: int
    token_threshold: int
    empty_search_text_rows: int


class TaxonomyRecord(BaseModel):
    """Discovered-at-index-time enumeration of field values.

    Used by the retriever to validate query inputs (warn on typos) and
    to power frontend dropdowns (list available levels/subjects).
    """

    languages: list[str] = Field(default_factory=list)
    question_types: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    levels: list[str] = Field(default_factory=list)


class BuildVectorStoreStats(BaseModel):
    """Audit record for Stage 3 — written to build_summary.json."""

    rows_indexed: int
    model_name: str
    embedding_dim: int
    collection_name: str
    persist_directory: str
    distance_metric: str
    wall_clock_seconds: float
    rows_per_second: float
    by_language: dict[str, int]
    by_question_type: dict[str, int]
    rows_dropped: dict[str, int] = Field(default_factory=dict)
    taxonomy: TaxonomyRecord = Field(default_factory=TaxonomyRecord)
