"""Tests for scripts/eval/refresh_topics.py — repairing a stale eval answer key.

The refresh must fix exactly what validate_test_cases flags and nothing else:
untouched rows stay byte-identical, hand-merged topic names survive, and new
spellings are never merged automatically. All data here is synthetic.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval import refresh_topics
from scripts.eval.refresh_topics import (
    COLUMNS,
    AliasProposal,
    CsvFormat,
    dump_aliases,
    load_aliases,
    propose_safe_aliases,
    read_topics_csv,
    refresh_cell,
    render_topics_csv,
    safe_title_key,
    topic_stats,
)
from scripts.eval.validate_test_cases import (
    TOPICS_FILE_BY_LANG_SUBJECT,
    IndexDoc,
    check_ground_truth_against_index,
)

EN = ("en", "ENGLISH")


def _row(
    doc_id: str, title: str, question: str = "Q?", level: str = "MIDDLE_SCHOOL_1ST_GRADE"
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "language": "en",
        "subjects": ["ENGLISH"],
        "quiz_title": title,
        "school_phase": level.split("_")[0],
        "levels": [level],
        "question_text": question,
        "choices_text": ["a", "b"],
    }


def _index(rows: list[dict[str, Any]]) -> tuple[dict[str, IndexDoc], dict[str, dict[str, Any]]]:
    full = {r["doc_id"]: r for r in rows}
    idx = {d: IndexDoc(r["language"], r["subjects"][0], r["quiz_title"]) for d, r in full.items()}
    return idx, full


def _csv_row(title: str, doc_ids: str, **overrides: str) -> dict[str, str]:
    row = dict.fromkeys(COLUMNS, "x")
    row.update(quiz_title=title, doc_ids=doc_ids, **overrides)
    return row


# ---------------------------------------------------------------------------
# refresh_cell
# ---------------------------------------------------------------------------


def test_refresh_cell_complete_key_leaves_rows_untouched() -> None:
    idx, full = _index([_row("q1__q0", "Pets"), _row("q1__q1", "Pets")])
    original = _csv_row("Pets", "q1__q0,q1__q1")
    result = refresh_cell(EN, [original], idx, full)
    assert result.changes == []
    assert result.rows[0] is original


def test_refresh_cell_stale_key_adds_unlisted_doc_and_recomputes_stats() -> None:
    """Regression: the English key predated the doc_id collision fix."""
    idx, full = _index(
        [_row("q1__q3", "Writing Ads", "abcd"), _row("q1__q3_2", "Writing Ads", "ab")]
    )
    result = refresh_cell(EN, [_csv_row("Writing Ads", "q1__q3")], idx, full)
    assert [(c.topic, c.added, c.removed) for c in result.changes] == [
        ("Writing Ads", ["q1__q3_2"], [])
    ]
    row = result.rows[0]
    assert row["doc_ids"] == "q1__q3,q1__q3_2"
    assert row["n_questions"] == "2"
    assert row["avg_q_len"] == "3.0"


def test_refresh_cell_keeps_existing_order_and_appends_added_ids() -> None:
    idx, full = _index([_row("q1__q0", "Pets"), _row("q1__q1", "Pets"), _row("q1__q2", "Pets")])
    result = refresh_cell(EN, [_csv_row("Pets", "q1__q2,q1__q0")], idx, full)
    assert result.rows[0]["doc_ids"] == "q1__q2,q1__q0,q1__q1"


def test_refresh_cell_keeps_sample_question_still_in_topic() -> None:
    idx, full = _index([_row("q1__q0", "Pets", "A?"), _row("q1__q1", "Pets", "B?")])
    original = _csv_row("Pets", "q1__q0", sample_question="A?", sample_choices="kept | choices")
    row = refresh_cell(EN, [original], idx, full).rows[0]
    assert (row["sample_question"], row["sample_choices"]) == ("A?", "kept | choices")


def test_refresh_cell_replaces_sample_question_that_left_topic() -> None:
    idx, full = _index([_row("q1__q0", "Pets", "A?"), _row("q1__q1", "Pets", "B?")])
    original = _csv_row("Pets", "q1__q0,gone__q0", sample_question="Gone?")
    row = refresh_cell(EN, [original], idx, full).rows[0]
    assert row["sample_question"] == "A?"


def test_refresh_cell_drops_doc_id_missing_from_index() -> None:
    idx, full = _index([_row("q1__q0", "Pets")])
    result = refresh_cell(EN, [_csv_row("Pets", "q1__q0,q1__xq6")], idx, full)
    assert result.changes[0].removed == ["q1__xq6"]
    assert result.rows[0]["doc_ids"] == "q1__q0"


def test_refresh_cell_merged_topic_keeps_its_name_and_absorbs_variant() -> None:
    idx, full = _index([_row("q1__q0", "present perfect"), _row("q1__q1", "present perfect")])
    result = refresh_cell(EN, [_csv_row("The Present Perfect", "q1__q0")], idx, full)
    assert result.rows[0]["quiz_title"] == "The Present Perfect"
    assert result.rows[0]["doc_ids"] == "q1__q0,q1__q1"


def test_refresh_cell_new_spelling_is_not_merged() -> None:
    """'simple present' is a separate reviewed decision, not a refresh."""
    idx, full = _index([_row("q1__q0", "The Simple Present"), _row("q2__q0", "simple present")])
    original = _csv_row("The Simple Present", "q1__q0")
    result = refresh_cell(EN, [original], idx, full)
    assert result.changes == []
    assert result.rows[0] is original


def test_refresh_cell_topic_left_with_no_questions_raises() -> None:
    idx, full = _index([_row("q1__q0", "Pets")])
    with pytest.raises(ValueError, match="no questions"):
        refresh_cell(EN, [_csv_row("Gone", "q9__q0")], idx, full)


def test_refresh_cell_output_passes_the_validator() -> None:
    idx, full = _index(
        [_row("q1__q3", "Writing Ads"), _row("q1__q3_2", "Writing Ads"), _row("q2__q0", "Ads")]
    )
    result = refresh_cell(EN, [_csv_row("Writing Ads", "q1__q3,gone__q1")], idx, full)
    topics = {EN: {r["quiz_title"]: set(r["doc_ids"].split(",")) for r in result.rows}}
    assert not check_ground_truth_against_index(topics, idx).has_problems


# ---------------------------------------------------------------------------
# Fidelity with the notebooks that produced the originals
# ---------------------------------------------------------------------------


def test_topic_stats_matches_notebook_aggregation() -> None:
    rows = [
        _row("q1__q0", "Pets", "What is a cat?", "PRIMARY_SCHOOL_4TH_GRADE"),
        _row("q1__q1", "Pets", 'Say "dog", please', "MIDDLE_SCHOOL_2ND_GRADE"),
        _row("q1__q2", "Pets", "Fish?", "PRIMARY_SCHOOL_4TH_GRADE"),
    ]
    df = pd.DataFrame(rows)
    df["first_level"] = df["levels"].apply(lambda L: L[0] if L else None)
    df["q_len"] = df["question_text"].apply(len)
    topics = (
        df.groupby("quiz_title")
        .agg(
            n_questions=("doc_id", "count"),
            school_phases=("school_phase", lambda x: ",".join(sorted(set(x)))),
            # Verbatim from the eval_dataset notebooks; the generator stays as written.
            levels=("first_level", lambda x: ",".join(sorted(set(str(v) for v in x if v)))),  # noqa: C401
            n_levels=("first_level", lambda x: len(set(x))),
            avg_q_len=("q_len", "mean"),
            sample_question=("question_text", "first"),
            sample_choices=("choices_text", lambda c: " | ".join(c.iloc[0]) if len(c) else ""),
            doc_ids=("doc_id", lambda ids: ",".join(ids)),
        )
        .reset_index()
    )
    topics["avg_q_len"] = topics["avg_q_len"].round(1)

    ours = render_topics_csv([{"quiz_title": "Pets", **topic_stats(rows)}], CsvFormat(False, "\n"))
    assert ours == topics.to_csv(index=False)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_render_topics_csv_round_trips_pandas_output_byte_for_byte(
    tmp_path: Path, encoding: str
) -> None:
    frame = pd.DataFrame(
        [
            {
                "quiz_title": 'Say "hi", then',
                "n_questions": 3,
                "school_phases": "HIGH,MIDDLE",
                "levels": "A,B",
                "n_levels": 2,
                "avg_q_len": 57.0,
                "sample_question": "Line one\nline two",
                "sample_choices": "a | b",
                "doc_ids": "q1__q0,q1__q1",
            },
            {
                "quiz_title": "صِيَغُ الأفعَالِ",
                "n_questions": 1,
                "school_phases": "PRIMARY",
                "levels": "",
                "n_levels": 1,
                "avg_q_len": 12.5,
                "sample_question": "?",
                "sample_choices": "",
                "doc_ids": "q2__q0",
            },
        ],
        columns=COLUMNS,
    )
    path = tmp_path / "topics.csv"
    frame.to_csv(path, index=False, encoding=encoding)
    rows, fmt, original = read_topics_csv(path)
    assert fmt.has_bom == (encoding == "utf-8-sig")
    assert render_topics_csv(rows, fmt) == original


# ---------------------------------------------------------------------------
# main — dry run by default, backup on write
# ---------------------------------------------------------------------------


def _files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    topics = tmp_path / "topics_english.csv"
    pd.DataFrame([_csv_row("Writing Ads", "q1__q3")], columns=COLUMNS).to_csv(topics, index=False)
    monkeypatch.setitem(TOPICS_FILE_BY_LANG_SUBJECT, EN, topics)
    ready = tmp_path / "ready.jsonl"
    ready.write_text(
        "".join(json.dumps(_row(d, "Writing Ads")) + "\n" for d in ("q1__q3", "q1__q3_2")),
        encoding="utf-8",
    )
    return ready


def test_main_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready = _files(tmp_path, monkeypatch)
    before = (tmp_path / "topics_english.csv").read_bytes()
    assert refresh_topics.main(["--cell", "en:ENGLISH", "--ready-jsonl", str(ready)]) == 0
    assert (tmp_path / "topics_english.csv").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ready.jsonl", "topics_english.csv"]


def test_main_write_updates_csv_and_keeps_backup_as_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = _files(tmp_path, monkeypatch)
    before = (tmp_path / "topics_english.csv").read_bytes()
    assert (
        refresh_topics.main(["--cell", "en:ENGLISH", "--ready-jsonl", str(ready), "--write"]) == 0
    )
    with (tmp_path / "topics_english.csv").open(encoding="utf-8", newline="") as f:
        assert next(csv.DictReader(f))["doc_ids"] == "q1__q3,q1__q3_2"
    backups = list(tmp_path.glob("topics_english.backup-*.csv"))
    assert len(backups) == 1, "backup must end in .csv so eval/*.csv keeps it out of git"
    assert backups[0].read_bytes() == before


# ---------------------------------------------------------------------------
# safe_title_key — what the automatic rule may and may not treat as one title
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Subject Pronouns", "Subject pronouns"),
        ("The Simple Present", "simple present"),
        ("Verb To Be", "Verb to be."),
        ("There is / There are", "There is/There are"),
        ("Addition & Opposition", "Addition and Opposition"),
    ],
)
def test_safe_title_key_superficial_differences_share_a_key(a: str, b: str) -> None:
    assert safe_title_key(a, "en") == safe_title_key(b, "en")


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Greeting", "Greetings"),
        ("At the restaurant (1)", "At the restaurant (2)"),
        ("Object Pronouns", "Subject Pronouns"),
        ("The + adjective", "Adjective"),
    ],
)
def test_safe_title_key_judgement_calls_keep_distinct_keys(a: str, b: str) -> None:
    assert safe_title_key(a, "en") != safe_title_key(b, "en")


def test_safe_title_key_keeps_arabic_vowel_marks() -> None:
    assert safe_title_key("صِيَغُ الأفعَالِ", "ar") == "صِيَغُ الأفعَالِ"


def test_safe_title_key_strips_articles_only_for_english() -> None:
    assert safe_title_key("The Plan", "fr") == "the plan"


# ---------------------------------------------------------------------------
# propose_safe_aliases and applying aliases
# ---------------------------------------------------------------------------


def test_propose_safe_aliases_merges_variant_into_its_only_owner() -> None:
    idx, _full = _index([_row("q1__q0", "The Simple Present"), _row("q2__q0", "simple present")])
    proposal = propose_safe_aliases(EN, [_csv_row("The Simple Present", "q1__q0")], idx)
    assert proposal.aliases == {"simple present": "The Simple Present"}
    assert proposal.conflicts == []


def test_propose_safe_aliases_reports_conflict_instead_of_merging() -> None:
    idx, _full = _index([_row("q1__q0", "Pets"), _row("q2__q0", "pets"), _row("q3__q0", "PETS")])
    rows = [_csv_row("Pets", "q1__q0"), _csv_row("pets", "q2__q0")]
    proposal = propose_safe_aliases(EN, rows, idx)
    assert proposal.aliases == {}
    assert proposal.conflicts == [(["Pets", "pets"], ["PETS", "Pets", "pets"])]


def test_propose_safe_aliases_ignores_groups_outside_the_eval() -> None:
    idx, _full = _index(
        [_row("q1__q0", "Pets"), _row("q2__q0", "Numbers"), _row("q3__q0", "numbers")]
    )
    assert propose_safe_aliases(EN, [_csv_row("Pets", "q1__q0")], idx).aliases == {}


def test_refresh_cell_with_alias_adds_variant_docs_and_keeps_topic_name() -> None:
    idx, full = _index([_row("q1__q0", "The Simple Present"), _row("q2__q0", "simple present")])
    result = refresh_cell(
        EN,
        [_csv_row("The Simple Present", "q1__q0")],
        idx,
        full,
        {"simple present": "The Simple Present"},
    )
    assert result.rows[0]["quiz_title"] == "The Simple Present"
    assert result.rows[0]["doc_ids"] == "q1__q0,q2__q0"
    topics = {EN: {r["quiz_title"]: set(r["doc_ids"].split(",")) for r in result.rows}}
    assert not check_ground_truth_against_index(topics, idx).has_problems


def test_refresh_cell_reapplying_an_applied_alias_changes_nothing() -> None:
    """Regression: an alias whose questions are already listed must not mark its row changed.

    Re-running a refresh with an alias file applied in an earlier run reported
    the topic as changed with +0 -0 and rewrote the row, so a second --write
    created a backup and rewrote the file for nothing.
    """
    idx, full = _index([_row("q1__q0", "The Simple Present"), _row("q2__q0", "simple present")])
    aliases = {"simple present": "The Simple Present"}
    once = refresh_cell(EN, [_csv_row("The Simple Present", "q1__q0")], idx, full, aliases).rows
    twice = refresh_cell(EN, once, idx, full, aliases)
    assert twice.changes == []
    assert twice.rows[0] is once[0]


def test_refresh_cell_alias_to_unknown_topic_raises() -> None:
    idx, full = _index([_row("q1__q0", "Pets"), _row("q2__q0", "pets")])
    with pytest.raises(ValueError, match="unknown topic"):
        refresh_cell(EN, [_csv_row("Pets", "q1__q0")], idx, full, {"pets": "Animals"})


def test_refresh_cell_alias_for_title_owned_by_another_topic_raises() -> None:
    idx, full = _index([_row("q1__q0", "Pets"), _row("q2__q0", "Animals")])
    rows = [_csv_row("Pets", "q1__q0"), _csv_row("Animals", "q2__q0")]
    with pytest.raises(ValueError, match="already own"):
        refresh_cell(EN, rows, idx, full, {"Animals": "Pets"})


def test_aliases_yaml_round_trips_including_arabic(tmp_path: Path) -> None:
    ar = ("ar", "ARABIC")
    proposals = [
        AliasProposal(EN, {"simple present": "The Simple Present"}, []),
        AliasProposal(ar, {"صيغ الأفعال": "صِيَغُ الأفعَالِ"}, []),
        AliasProposal(("fr", "FRENCH"), {}, []),
    ]
    path = tmp_path / "aliases.yaml"
    path.write_text(dump_aliases(proposals), encoding="utf-8")
    assert load_aliases(path) == {
        EN: {"simple present": "The Simple Present"},
        ar: {"صيغ الأفعال": "صِيَغُ الأفعَالِ"},
    }
