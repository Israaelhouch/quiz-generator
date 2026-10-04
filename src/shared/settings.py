"""Typed runtime configuration, read once from the environment.

Every environment variable this service reads is declared here with its type
and its default, and validated the first time settings are needed — a
malformed value fails loudly instead of silently falling back to a default
that is only visible inside whichever function happened to read it.

Business code calls `get_settings()` rather than `os.environ`, so one
variable cannot be read with two different defaults in two different modules
(CLAUDE.md Part II §2).

Scope: this file is the *environment* only — secrets, paths, limits and
switches. Model, pipeline and scope parameters live in `configs/*.yaml` with
their own typed loaders, because they belong to a deployment's data rather
than its process.

The process environment is the only source: no `.env` file is read here.
Docker Compose and `run_local.sh` inject the variables, and the test suite
scrubs them, so an untracked local `.env` must not silently change behaviour.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"})


def _split_csv(raw: str) -> tuple[str, ...]:
    """Split a comma-separated environment value, dropping blanks."""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


class Settings(BaseSettings):
    """The service's environment, validated.

    Defaults match what the code applied before this class existed, so the
    behaviour of an unset variable is unchanged.
    """

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    # -- LLM providers ------------------------------------------------------
    # Secrets are SecretStr so an accidental log line or repr prints
    # "**********" instead of the key (CLAUDE.md Part II §8).
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    ollama_host: str | None = None
    judge_model: str = "gemini-2.5-flash"
    llm_timeout_seconds: float = Field(default=90.0, ge=0)

    # -- API boundary -------------------------------------------------------
    # Empty API_KEYS means authentication is disabled; the server says so
    # loudly at startup rather than pretending to be protected.
    api_keys: str = ""
    rate_limit_per_minute: int = Field(default=30, ge=0)
    cors_allow_origins: str = ""

    # -- Logging and observability -----------------------------------------
    log_level: str = "INFO"
    log_runs: bool = True
    runs_log_path: Path = Path("/app/logs/runs.jsonl")
    feedback_log_path: Path = Path("/app/logs/feedback.jsonl")
    runs_log_max_bytes: int = Field(default=50 * 1024 * 1024, ge=0)

    # -- Privacy ------------------------------------------------------------
    # The corpus carries author_name / author_email for real teachers.
    include_author_metadata: bool = False

    @field_validator("log_level")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        """Reject a level name `logging` would not understand."""
        resolved = value.strip().upper()
        if resolved not in _LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL={value!r} is not one of {sorted(_LOG_LEVELS)}")
        return resolved

    @property
    def api_key_set(self) -> frozenset[str]:
        """The configured API keys. Empty means authentication is disabled."""
        return frozenset(_split_csv(self.api_keys))

    @property
    def auth_enabled(self) -> bool:
        """Whether any API key is configured."""
        return bool(self.api_key_set)

    @property
    def gemini_key(self) -> str | None:
        """The Gemini key as a plain string, for the provider SDK."""
        return self.gemini_api_key.get_secret_value() if self.gemini_api_key else None

    @property
    def groq_key(self) -> str | None:
        """The Groq key as a plain string, for the provider SDK."""
        return self.groq_api_key.get_secret_value() if self.groq_api_key else None

    @property
    def cors_origin_list(self) -> list[str]:
        """Browser origins allowed to call this API. Empty means none."""
        return list(_split_csv(self.cors_allow_origins))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, validating them on first call.

    Cached: the environment is read once, so every module sees the same
    values and a later `os.environ` edit cannot change behaviour mid-process.
    Tests that need a different environment call `reset_settings()`.
    """
    return Settings()


def reset_settings() -> None:
    """Drop the cached settings so the next call re-reads the environment.

    For tests and for a process that deliberately reconfigures itself. Not
    for request handling: settings are meant to be stable while serving.
    """
    get_settings.cache_clear()


def describe_settings(settings: Settings | None = None) -> str:
    """One line naming the effective configuration, with no secret values."""
    current = settings or get_settings()
    return (
        f"log_level={current.log_level} "
        f"auth={'on' if current.auth_enabled else 'OFF'} "
        f"rate_limit_per_minute={current.rate_limit_per_minute} "
        f"log_runs={current.log_runs} "
        f"runs_log_max_bytes={current.runs_log_max_bytes} "
        f"include_author_metadata={current.include_author_metadata} "
        f"llm_timeout_seconds={current.llm_timeout_seconds} "
        f"gemini_api_key={'set' if current.gemini_api_key else 'unset'} "
        f"groq_api_key={'set' if current.groq_api_key else 'unset'}"
    )
