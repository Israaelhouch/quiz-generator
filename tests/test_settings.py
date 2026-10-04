"""Settings are the one place the environment enters the process.

These tests pin two things: that an unset variable still behaves the way it
did when every module read `os.environ` itself, and that a malformed value is
refused at startup instead of being quietly replaced by a default.
"""

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.shared.settings import Settings, describe_settings, get_settings, reset_settings

# Every variable Settings reads. Scrubbed before each test so the suite does
# not behave differently depending on whose shell or .env it runs under.
_MANAGED = (
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "OLLAMA_HOST",
    "JUDGE_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "API_KEYS",
    "RATE_LIMIT_PER_MINUTE",
    "CORS_ALLOW_ORIGINS",
    "LOG_LEVEL",
    "LOG_RUNS",
    "RUNS_LOG_PATH",
    "FEEDBACK_LOG_PATH",
    "RUNS_LOG_MAX_BYTES",
    "INCLUDE_AUTHOR_METADATA",
)


@contextlib.contextmanager
def _env(**overrides: str):
    """Run with a clean environment plus the given overrides."""
    previous = {k: os.environ.get(k) for k in _MANAGED}
    try:
        for key in _MANAGED:
            os.environ.pop(key, None)
        os.environ.update(overrides)
        reset_settings()
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_settings()


# ---------------------------------------------------------------------------
# Defaults — these must match what the code applied before Settings existed
# ---------------------------------------------------------------------------


def test_unset_environment_yields_the_previous_defaults() -> None:
    with _env():
        settings = get_settings()

    assert settings.rate_limit_per_minute == 30
    assert settings.llm_timeout_seconds == 90.0
    assert settings.log_level == "INFO"
    assert settings.log_runs is True
    assert settings.runs_log_path == Path("/app/logs/runs.jsonl")
    assert settings.feedback_log_path == Path("/app/logs/feedback.jsonl")
    assert settings.runs_log_max_bytes == 50 * 1024 * 1024
    assert settings.include_author_metadata is False
    assert settings.judge_model == "gemini-2.5-flash"
    assert settings.gemini_api_key is None


def test_empty_api_keys_means_auth_is_disabled() -> None:
    with _env():
        assert get_settings().auth_enabled is False
    with _env(API_KEYS=" , ,  "):
        assert get_settings().auth_enabled is False


def test_comma_separated_values_are_split_and_stripped() -> None:
    with _env(API_KEYS=" k1 , k2,,k3 ", CORS_ALLOW_ORIGINS="https://a.test, https://b.test"):
        settings = get_settings()
        assert settings.api_key_set == frozenset({"k1", "k2", "k3"})
        assert settings.auth_enabled is True
        assert settings.cors_origin_list == ["https://a.test", "https://b.test"]


def test_one_and_zero_are_read_as_booleans() -> None:
    with _env(LOG_RUNS="0", INCLUDE_AUTHOR_METADATA="1"):
        settings = get_settings()
        assert settings.log_runs is False
        assert settings.include_author_metadata is True


def test_a_lowercase_log_level_is_accepted_and_normalised() -> None:
    with _env(LOG_LEVEL="debug"):
        assert get_settings().log_level == "DEBUG"


# ---------------------------------------------------------------------------
# Refusal — the reason this class exists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("RATE_LIMIT_PER_MINUTE", "abc"),
        ("RATE_LIMIT_PER_MINUTE", "-5"),
        ("RUNS_LOG_MAX_BYTES", "fifty-megabytes"),
        ("RUNS_LOG_MAX_BYTES", "-1"),
        ("LLM_TIMEOUT_SECONDS", "soon"),
        ("LLM_TIMEOUT_SECONDS", "-30"),
        ("LOG_LEVEL", "CHATTY"),
        ("LOG_RUNS", "maybe"),
        ("INCLUDE_AUTHOR_METADATA", "sure"),
    ],
)
def test_a_value_that_cannot_be_honoured_is_refused(variable: str, value: str) -> None:
    """Every one of these used to be swallowed: the process ran on with a
    default, so a typo in a deployment looked like it had been applied."""
    with _env(**{variable: value}), pytest.raises(ValidationError) as excinfo:
        get_settings()

    assert variable.lower() in str(excinfo.value).lower()


def test_an_unknown_variable_is_ignored_rather_than_fatal() -> None:
    """Deployments carry unrelated variables (PATH, HOME, platform injections);
    only the declared ones are read."""
    with _env(SOMETHING_ELSE_ENTIRELY="1"):
        assert get_settings().rate_limit_per_minute == 30


# ---------------------------------------------------------------------------
# Secrets and caching
# ---------------------------------------------------------------------------


def test_a_secret_never_appears_in_a_repr_or_in_the_startup_line() -> None:
    """The startup log line names the configuration; CLAUDE.md Part II §8
    forbids the values of secrets reaching logs."""
    with _env(GEMINI_API_KEY="AIza-super-secret", GROQ_API_KEY="gsk-super-secret"):
        settings = get_settings()
        rendered = f"{settings!r} {settings} {describe_settings(settings)}"

        assert "super-secret" not in rendered
        assert "gemini_api_key=set" in describe_settings(settings)
        assert settings.gemini_api_key is not None
        assert settings.gemini_api_key.get_secret_value() == "AIza-super-secret"


def test_settings_are_read_once_until_explicitly_reset() -> None:
    """Request handling must not see the environment shift underneath it."""
    with _env(RATE_LIMIT_PER_MINUTE="7"):
        first = get_settings()
        assert first.rate_limit_per_minute == 7
        assert get_settings() is first  # cached, not re-read

        os.environ["RATE_LIMIT_PER_MINUTE"] = "99"
        assert get_settings().rate_limit_per_minute == 7  # still the first read

        reset_settings()
        assert get_settings().rate_limit_per_minute == 99


# ---------------------------------------------------------------------------
# Documentation drift
# ---------------------------------------------------------------------------


def test_every_variable_in_env_example_is_declared_in_settings() -> None:
    """`.env.example` is the operator-facing contract. It drifted before:
    it claimed an unset LLM_TIMEOUT_SECONDS meant the provider's own default
    when the code applied 90s."""
    documented = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", Path(".env.example").read_text(), re.M))
    declared = {name.upper() for name in Settings.model_fields}

    assert documented - declared == set(), "documented but not read by Settings"
    assert declared - documented == set(), "read by Settings but not documented"
