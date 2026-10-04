"""Record what produced an eval run's numbers.

A run directory held the metrics, the test-case arguments and a copy of
`configs/models.yaml`. That is enough to know how the *retriever* was
configured, and not enough to know what it searched: the index is identified
only by a directory path, so rebuilding the index makes every earlier run
unattributable.

This module collects the rest of the chain, which already exists on disk but
was never gathered in one place:

    configs/pipeline.yaml  ──> the search_text recipe
      recorded in data/processed/<ready>_stats.json as recipe + recipe_flags
    data/processed/ready_*.jsonl  ──> the embedded payload
      hashed into data/vector_store/build_summary.json as source_sha256
    the index itself  ──> model, dimension, collection, row count, build time

The result is written as `provenance.json` next to the metrics. Anything that
cannot be found is listed under `warnings` rather than quietly omitted — a run
that cannot account for itself should say so.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str | None:
    """Stream a file's SHA-256, or None when it is not there."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo_root: Path | None = None) -> dict[str, Any]:
    """The commit the run was made from, and whether the tree was dirty.

    Best-effort: a tarball download has no git metadata, and a run from one is
    still a valid run.
    """
    root = repo_root or Path.cwd()

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "sha": sha,
        "dirty": None if status is None else bool(status.strip()),
    }


def _index_summary_path(config_path: Path) -> Path | None:
    """Where `build_summary.json` sits for the index this config points at."""
    try:
        from quiz_generator.indexing.config import load_models_config

        config = load_models_config(config_path)
    except Exception:  # a config this malformed is reported by the caller
        return None
    return Path(config.vector_store.persist_directory).parent / "build_summary.json"


def _recipe_from_payload_stats(source_path: str | None) -> tuple[dict[str, Any], str | None]:
    """The search_text recipe that produced the embedded payload.

    `build_index_text` writes it beside its output as `<stem>_stats.json`.
    """
    if not source_path:
        return {}, "index summary has no source_path, so the recipe is unknown"

    stats_path = Path(source_path).with_name(f"{Path(source_path).stem}_stats.json")
    if not stats_path.exists():
        return {}, f"payload stats not found at {stats_path}, so the recipe is unknown"

    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"payload stats at {stats_path} could not be read: {exc}"

    return {
        "stats_path": str(stats_path),
        "recipe": stats.get("recipe"),
        "recipe_flags": stats.get("recipe_flags"),
        "token_threshold": stats.get("token_threshold"),
        "rows": stats.get("output_rows"),
    }, None


def collect_provenance(
    *,
    config_path: Path,
    payload: Path,
    pipeline_config_path: Path = Path("configs/pipeline.yaml"),
    index_summary_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Gather everything needed to account for a set of eval numbers."""
    warnings: list[str] = []

    summary_path = index_summary_path or _index_summary_path(config_path)
    index: dict[str, Any] = {}
    payload_recipe: dict[str, Any] = {}

    if summary_path is None:
        warnings.append(
            f"could not read {config_path} to locate the index, so the index is unidentified"
        )
    elif not summary_path.exists():
        warnings.append(
            f"index build summary not found at {summary_path}; the index this run searched "
            "is identified only by its directory"
        )
    else:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            summary = {}
            warnings.append(f"index build summary at {summary_path} could not be read: {exc}")

        index = {
            "summary_path": str(summary_path),
            "built_at_utc": summary.get("built_at_utc"),
            "rows_indexed": summary.get("rows_indexed"),
            "model_name": summary.get("model_name"),
            "embedding_dim": summary.get("embedding_dim"),
            "collection_name": summary.get("collection_name"),
            "persist_directory": summary.get("persist_directory"),
            "distance_metric": summary.get("distance_metric"),
            "payload_path": summary.get("source_path"),
            "payload_sha256": summary.get("source_sha256"),
        }
        payload_recipe, recipe_warning = _recipe_from_payload_stats(summary.get("source_path"))
        if recipe_warning:
            warnings.append(recipe_warning)

    searched_payload_sha = sha256_of(payload)
    if searched_payload_sha is None:
        warnings.append(f"payload not found at {payload}")
    elif index.get("payload_sha256") and index["payload_sha256"] != searched_payload_sha:
        warnings.append(
            "the payload on disk is NOT the one this index was built from "
            f"({payload} hashes to {searched_payload_sha[:12]}…, the index records "
            f"{str(index['payload_sha256'])[:12]}…) — the index is stale or the payload "
            "was rebuilt after it"
        )

    return {
        "git": git_state(repo_root),
        "configs": {
            "models": {"path": str(config_path), "sha256": sha256_of(config_path)},
            "pipeline": {
                "path": str(pipeline_config_path),
                "sha256": sha256_of(pipeline_config_path),
            },
        },
        "payload_on_disk": {"path": str(payload), "sha256": searched_payload_sha},
        "index": index,
        "search_text": payload_recipe,
        "warnings": warnings,
    }


def write_provenance(
    out_dir: Path,
    record: dict[str, Any],
    *,
    pipeline_config_path: Path = Path("configs/pipeline.yaml"),
) -> Path:
    """Write `provenance.json` and snapshot the pipeline config beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "provenance.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    if pipeline_config_path.exists():
        shutil.copy(pipeline_config_path, out_dir / "pipeline_snapshot.yaml")

    return path
