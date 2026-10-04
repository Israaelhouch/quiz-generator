"""File hashing, used to identify the inputs a pipeline stage actually read.

A stage's stats record is only useful if it says which configuration produced
it. Paths are not enough — `configs/scope.yaml` means something different
before and after an edit — so the records carry the file's digest alongside
its name.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_BYTES = 1024 * 1024


def sha256_of(path: Path) -> str | None:
    """Stream a file's SHA-256, or None when the file is not there."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
