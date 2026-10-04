"""Shared config and parsing helpers."""

from __future__ import annotations

import ast
import json
from typing import Any

TRUTHY = {"1", "true", "yes"}
FALSY = {"0", "false", "no"}


def get_required(config: dict[str, Any], path: list[str]) -> Any:
    """Read a nested required value from a dict."""
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(path)
            raise ValueError(f"Missing required config key: {joined}")
        current = current[key]
    return current


def normalize_match_mode(value: Any, *, field_name: str) -> str:
    """Normalize list matching mode and validate supported values."""
    mode = str(value).strip().lower()
    if mode not in {"any", "all"}:
        raise ValueError(f"{field_name} must be 'any' or 'all', got: {value}")
    return mode


def parse_list_cell(raw_value: Any) -> list[str]:
    """Parse JSON/literal/list-like values into a normalized list of strings."""
    if raw_value is None:
        return []

    if isinstance(raw_value, (list, tuple, set)):
        return [str(item).strip() for item in raw_value if str(item).strip()]

    text = str(raw_value).strip()
    if not text:
        return []

    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if parsed is not None:
        value = str(parsed).strip()
        return [value] if value else []

    # Fallback for plain delimited text values.
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    return [text]


def parse_bool_cell(raw_value: Any) -> bool | None:
    """Parse a boolean-ish cell, returning None when value is empty/unknown."""
    if raw_value is None:
        return None

    text = str(raw_value).strip().lower()
    if not text:
        return None
    if text in TRUTHY:
        return True
    if text in FALSY:
        return False
    return None


def normalize_bool(raw_value: Any, *, default: bool = False) -> bool:
    """Parse boolean-ish values, falling back to `default` when unknown."""
    parsed = parse_bool_cell(raw_value)
    if parsed is None:
        return default
    return parsed


def parse_int_cell(raw_value: Any) -> int | None:
    """Parse optional integer-like values."""
    if raw_value is None:
        return None

    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float_cell(raw_value: Any) -> float | None:
    """Parse optional float-like values."""
    if raw_value is None:
        return None

    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
