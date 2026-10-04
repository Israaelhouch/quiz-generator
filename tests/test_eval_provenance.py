"""An eval run must be able to account for its own numbers.

The metrics, the retriever config and the test-case arguments were already
recorded per run. What was missing was the identity of the thing being
searched: rebuild the index and every earlier run became unattributable. These
tests pin the record, and in particular the case that matters most — numbers
measured against an index that no longer matches the payload on disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.eval.provenance import (
    collect_provenance,
    git_state,
    sha256_of,
    write_provenance,
)

CONFIG = Path("configs/models.yaml")
PIPELINE = Path("configs/pipeline.yaml")


def _payload(tmp_path: Path, text: str = '{"doc_id": "q1"}\n') -> Path:
    path = tmp_path / "ready.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def _payload_stats(payload: Path, recipe: str = "default") -> Path:
    path = payload.with_name(f"{payload.stem}_stats.json")
    path.write_text(
        json.dumps(
            {
                "recipe": recipe,
                "recipe_flags": {"include_question": True, "include_choices": True},
                "token_threshold": 100,
                "output_rows": 1,
            }
        ),
        encoding="utf-8",
    )
    return path


def _index_summary(tmp_path: Path, payload: Path, payload_sha: str) -> Path:
    path = tmp_path / "build_summary.json"
    path.write_text(
        json.dumps(
            {
                "source_path": str(payload),
                "source_sha256": payload_sha,
                "built_at_utc": "2026-09-08T10:47:42+00:00",
                "rows_indexed": 5782,
                "model_name": "BAAI/bge-m3",
                "embedding_dim": 1024,
                "collection_name": "quiz_questions",
                "persist_directory": "/tmp/chroma_db",
                "distance_metric": "cosine",
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# The complete chain
# ---------------------------------------------------------------------------


def test_a_complete_record_names_the_index_the_recipe_and_the_commit(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    _payload_stats(payload)
    summary = _index_summary(tmp_path, payload, sha256_of(payload) or "")

    record = collect_provenance(
        config_path=CONFIG,
        ready_jsonl=payload,
        pipeline_config_path=PIPELINE,
        index_summary_path=summary,
    )

    assert record["warnings"] == []
    assert record["index"]["model_name"] == "BAAI/bge-m3"
    assert record["index"]["rows_indexed"] == 5782
    assert record["search_text"]["recipe"] == "default"
    assert record["search_text"]["recipe_flags"]["include_choices"] is True
    assert record["configs"]["models"]["sha256"] == sha256_of(CONFIG)
    assert record["configs"]["pipeline"]["sha256"] == sha256_of(PIPELINE)
    assert record["payload_on_disk"]["sha256"] == sha256_of(payload)
    assert "sha" in record["git"]


PAYLOAD = Path("data/processed/ready.jsonl")
INDEX_SUMMARY = Path("data/vector_store/build_summary.json")


def _repository_is_built() -> bool:
    """True when this checkout has an index whose build record is current.

    False on CI and on a fresh clone, where the artefacts are gitignored, and
    false between a rename of the data artefacts and the rebuild that follows
    it — in that window the build record names paths that no longer exist, and
    reporting that is the feature under test, not a failure of it.
    """
    if not (PAYLOAD.exists() and INDEX_SUMMARY.exists()):
        return False
    try:
        recorded = json.loads(INDEX_SUMMARY.read_text(encoding="utf-8")).get("source_path")
    except (OSError, json.JSONDecodeError):
        return False
    return bool(recorded) and Path(recorded).exists()


@pytest.mark.skipif(not _repository_is_built(), reason="no current index build in this checkout")
def test_a_built_repository_produces_a_record_without_warnings() -> None:
    """With the artefacts present, the record should account for all of them:
    the index was built from the payload that is on disk."""
    record = collect_provenance(config_path=CONFIG, ready_jsonl=PAYLOAD)

    assert record["warnings"] == []
    assert record["index"]["payload_sha256"] == record["payload_on_disk"]["sha256"]


# ---------------------------------------------------------------------------
# The case worth catching
# ---------------------------------------------------------------------------


def test_a_payload_that_no_longer_matches_the_index_is_reported(tmp_path: Path) -> None:
    """Rebuild the payload without reindexing and the numbers describe an index
    built from something else. Silence here is how a metric moves for reasons
    nobody can reconstruct."""
    payload = _payload(tmp_path)
    _payload_stats(payload)
    stale = hashlib.sha256(b"a different payload entirely").hexdigest()
    summary = _index_summary(tmp_path, payload, stale)

    record = collect_provenance(config_path=CONFIG, ready_jsonl=payload, index_summary_path=summary)

    assert any("NOT the one this index was built from" in w for w in record["warnings"])


def test_a_missing_index_summary_is_a_warning_not_a_crash(tmp_path: Path) -> None:
    """An index built before build summaries existed should still be evaluable,
    with the run saying what it could not record."""
    payload = _payload(tmp_path)

    record = collect_provenance(
        config_path=CONFIG,
        ready_jsonl=payload,
        index_summary_path=tmp_path / "absent.json",
    )

    assert record["index"] == {}
    assert any("build summary not found" in w for w in record["warnings"])


def test_a_missing_payload_stats_file_leaves_the_recipe_unknown(tmp_path: Path) -> None:
    payload = _payload(tmp_path)  # no _stats.json beside it
    summary = _index_summary(tmp_path, payload, sha256_of(payload) or "")

    record = collect_provenance(config_path=CONFIG, ready_jsonl=payload, index_summary_path=summary)

    assert record["search_text"] == {}
    assert any("recipe is unknown" in w for w in record["warnings"])


def test_a_missing_payload_is_reported(tmp_path: Path) -> None:
    record = collect_provenance(
        config_path=CONFIG,
        ready_jsonl=tmp_path / "absent.jsonl",
        index_summary_path=tmp_path / "absent.json",
    )

    assert record["payload_on_disk"]["sha256"] is None
    assert any("payload not found" in w for w in record["warnings"])


# ---------------------------------------------------------------------------
# Writing, and git outside a repository
# ---------------------------------------------------------------------------


def test_writing_leaves_the_record_and_a_pipeline_snapshot(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"

    path = write_provenance(out_dir, {"warnings": []}, pipeline_config_path=PIPELINE)

    assert json.loads(path.read_text(encoding="utf-8")) == {"warnings": []}
    assert (out_dir / "pipeline_snapshot.yaml").read_text(encoding="utf-8") == PIPELINE.read_text(
        encoding="utf-8"
    )


def test_git_state_outside_a_repository_is_recorded_as_unknown(tmp_path: Path) -> None:
    """A run from a tarball download has no commit, and is still a valid run."""
    state = git_state(tmp_path)

    assert state == {"sha": None, "dirty": None}


def test_sha256_of_a_missing_file_is_none(tmp_path: Path) -> None:
    assert sha256_of(tmp_path / "absent") is None
