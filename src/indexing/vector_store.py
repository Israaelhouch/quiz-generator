"""Chroma vector-store wrapper for Stage 3.

Thin abstraction so the orchestrator doesn't talk to Chroma directly.
Also keeps chromadb imports lazy so unit tests can mock this layer.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VectorStoreConfig:
    persist_directory: Path
    collection_name: str
    distance_metric: str = "cosine"
    add_batch_size: int = 128
    reset_on_build: bool = True


def row_to_metadata(
    row: dict,
    *,
    scalar_fields: list[str],
    list_fields_as_json: list[str],
    list_fields_as_booleans: list[str] | None = None,
    derive_scalar_subject: bool = True,
) -> dict[str, Any]:
    """Build a Chroma-safe metadata dict from a ready.jsonl row.

    - Scalar fields: copied as-is if the value is str/int/float/bool AND non-empty.
    - list_fields_as_json: JSON-encoded as `<name>_json`. Empty lists → "[]".
    - list_fields_as_booleans: each value in the list emits a separate metadata
      key `<name>_<value>: True`. Missing entries mean False (by omission).
      Lets Chroma pre-filter natively on essential list filters like `levels`.
    - None/empty strings are dropped (Chroma rejects None values).
    - When derive_scalar_subject is True, `subjects[0]` is copied into a scalar
      `subject` field so Chroma can pre-filter natively. Secondary subjects
      (rare: 3 rows in our data) are dropped. Set False to disable.
    """
    metadata: dict[str, Any] = {}

    for key in scalar_fields:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                metadata[key] = stripped
        elif isinstance(value, (int, float, bool)):
            metadata[key] = value

    for key in list_fields_as_json:
        value = row.get(key) or []
        if not isinstance(value, list):
            value = [value]
        metadata[f"{key}_json"] = json.dumps(value, ensure_ascii=False)

    for key in list_fields_as_booleans or []:
        values = row.get(key) or []
        if not isinstance(values, list):
            values = [values]
        for item in values:
            if item is None:
                continue
            safe = str(item).strip()
            if safe:
                metadata[f"{key}_{safe}"] = True

    if derive_scalar_subject:
        subjects = row.get("subjects") or []
        if isinstance(subjects, list) and subjects:
            first = str(subjects[0]).strip()
            if first:
                metadata["subject"] = first

    return metadata


def build_ids(rows: list[dict]) -> list[str]:
    """Stable IDs from doc_id. Guarantees uniqueness even if a doc_id repeats."""
    seen: dict[str, int] = {}
    ids: list[str] = []
    for idx, row in enumerate(rows):
        base = str(row.get("doc_id") or f"row_{idx}").strip()
        count = seen.get(base, 0)
        ids.append(base if count == 0 else f"{base}__dup{count}")
        seen[base] = count + 1
    return ids


class VectorStore:
    """Handles the Chroma collection lifecycle."""

    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self._collection: Any | None = None

    def open(self) -> None:
        """Create or re-open the collection. Call once before add_batch."""
        import chromadb  # heavy, lazy import

        if self.config.reset_on_build and self.config.persist_directory.exists():
            shutil.rmtree(self.config.persist_directory)

        self.config.persist_directory.mkdir(parents=True, exist_ok=True)

        # Clear any stale in-process Chroma client cache (common in notebooks).
        try:
            from chromadb.api import shared_system_client

            shared_system_client.SharedSystemClient.clear_system_cache()
        except Exception:  # pragma: no cover
            pass

        client = chromadb.PersistentClient(path=str(self.config.persist_directory))
        self._collection = client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric},
        )

    def add_batch(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if self._collection is None:
            raise RuntimeError("VectorStore.open() must be called before add_batch().")
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    @property
    def collection(self) -> Any:
        return self._collection
