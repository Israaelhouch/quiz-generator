"""Read a YAML config file, or fail with a message that names the file.

The configs under `configs/` are pipeline parameters: the `search_text`
recipe decides what text gets embedded, the scope filter decides which rows
enter the corpus at all. A missing or malformed one must not fall back to a
default, because the job would still succeed and produce an index that
silently differs from the one the published metrics were measured on
(CLAUDE.md Part I P2, Part II §7).

This module only gets the file into a mapping. Each config's own loader
validates its shape with a Pydantic model that forbids unknown keys, so a
typo is an error rather than a key nobody reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_yaml_mapping(path: Path, *, what: str) -> dict[str, Any]:
    """Return the top-level mapping in `path`.

    Raises FileNotFoundError if the file is absent, and ValueError if it is
    empty or does not hold a mapping. `what` names the config in the message.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{what} config not found: {path}. "
            "Pass an existing path; this is a pipeline parameter, so it is not "
            "defaulted silently."
        )

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raise ValueError(f"{what} config is empty: {path}")
    if not isinstance(raw, dict):
        raise ValueError(f"{what} config must be a mapping, found {type(raw).__name__}: {path}")
    return raw
