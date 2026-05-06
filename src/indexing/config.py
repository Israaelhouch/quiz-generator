"""Stage 3 config loader for `configs/models.yaml`.

Pydantic-validated at load time. Pure-Python helpers elsewhere can
consume the dataclass-like result without pulling Pydantic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Defaults are set here so a partial YAML file still loads cleanly.
DEFAULT_METADATA_SCALARS = [
    "quiz_id",
    "quiz_title",
    "language",
    "question_type",
    "multiple_correct_answers",
    "author_name",
    "author_email",
    "points",
    "time",
]
DEFAULT_METADATA_LISTS = ["subjects", "levels"]


def load_models_config(config_path: Path) -> "ModelsConfig":
    """Load + validate `configs/models.yaml`. Returns a typed config object."""
    # Lazy Pydantic import so tests on the pure helpers run without it.
    from pydantic import BaseModel, ConfigDict, Field, field_validator

    class ModelConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str
        embedding_dim: int
        batch_size: int = 16
        device: str = "auto"
        normalize_embeddings: bool = True
        passage_prefix: str = ""
        query_prefix: str = ""

    class VectorStoreConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        type: str = "chroma"
        persist_directory: Path
        collection_name: str
        distance_metric: str = "cosine"
        add_batch_size: int = 128
        reset_on_build: bool = True

        @field_validator("type")
        @classmethod
        def _check_type(cls, v: str) -> str:
            if v != "chroma":
                raise ValueError(f"Only 'chroma' is supported for now, got {v!r}")
            return v

    class SmokeTestQuery(BaseModel):
        model_config = ConfigDict(extra="forbid")
        lang: str
        text: str

    class SmokeTestConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        enabled: bool = True
        queries: list[SmokeTestQuery] = Field(default_factory=list)
        top_k: int = 3

    class IndexingConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        embedding_text_column: str = "search_text"
        metadata_scalar_fields: list[str] = Field(default_factory=lambda: list(DEFAULT_METADATA_SCALARS))
        metadata_list_fields_as_json: list[str] = Field(default_factory=list)
        metadata_list_fields_as_booleans: list[str] = Field(default_factory=lambda: ["levels"])
        smoke_test: SmokeTestConfig = Field(default_factory=SmokeTestConfig)

    class RerankerConfigPyd(BaseModel):
        model_config = ConfigDict(extra="forbid")
        enabled: bool = False
        model_name: str = "BAAI/bge-reranker-v2-m3"
        candidate_pool_size: int = 50
        device: str = "auto"
        batch_size: int = 16

    class LLMConfigPyd(BaseModel):
        """Stage 6 — LLM client config.

        Supported providers:
          - 'ollama' — local Ollama server (uses `host` to connect)
          - 'groq'   — Groq hosted API (requires GROQ_API_KEY env var)
          - 'gemini' — Google Gemini (requires GEMINI_API_KEY env var)
        """
        model_config = ConfigDict(extra="forbid")
        provider: str = "ollama"
        model: str = "qwen2.5:7b"
        host: str | None = None        # only used by Ollama provider
        default_temperature: float = Field(default=0.75, ge=0.0, le=2.0)
        max_attempts: int = Field(default=3, ge=1, le=10)
        # Ceiling on examples passed to the LLM (after distance filter applies).
        default_few_shot_count: int = Field(default=5, ge=1, le=20)
        # Quality floor — chunks with distance > this are dropped. Set to None
        # to disable distance filtering and rely only on few_shot_count.
        default_max_distance: float | None = Field(default=None, ge=0.0, le=2.0)

        @field_validator("provider")
        @classmethod
        def _check_provider(cls, v: str) -> str:
            allowed = {"ollama", "groq", "gemini"}
            if v not in allowed:
                raise ValueError(
                    f"Provider must be one of {sorted(allowed)}, got {v!r}"
                )
            return v

    class ModelsConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        model: ModelConfig
        vector_store: VectorStoreConfig
        reranker: RerankerConfigPyd = Field(default_factory=RerankerConfigPyd)
        llm: LLMConfigPyd = Field(default_factory=LLMConfigPyd)
        indexing: IndexingConfig = Field(default_factory=IndexingConfig)

        @property
        def embedding_text_column(self) -> str:
            return self.indexing.embedding_text_column

    if not config_path.exists():
        raise FileNotFoundError(f"Models config not found: {config_path}")

    config_path = config_path.resolve()
    project_root = config_path.parent.parent

    with config_path.open("r", encoding="utf-8") as f:
        loaded: dict[str, Any] = yaml.safe_load(f) or {}

    # Resolve persist_directory relative to project root
    vs = loaded.get("vector_store") or {}
    persist = vs.get("persist_directory")
    if persist:
        p = Path(str(persist))
        if not p.is_absolute():
            p = project_root / p
        vs["persist_directory"] = p
        loaded["vector_store"] = vs

    return ModelsConfig.model_validate(loaded)
