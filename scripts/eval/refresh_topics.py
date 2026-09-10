"""Refresh the doc_ids in eval/topics_<lang>.csv from the current index.

Why this exists
---------------
The retrieval eval scores a retrieved question as correct only if its doc_id
is listed under the target topic. Those lists were built once, by notebook,
from the index as it was that day. Rebuilding the index since — notably the
2026-05-18 doc_id collision fix, which added ids like `__q3_2` — left the
English list stale: 205 questions belonging to its topics were missing, and
retrieving them was scored as wrong. validate_test_cases detects that; this
script repairs it.

What it changes, and what it deliberately does not
--------------------------------------------------
Keeps every topic name, the row order and the hand-made merges. A topic owns
its own title plus every title its listed doc_ids carry today — the rule
validate_test_cases applies, reused here rather than re-implemented — so
'The Present Perfect' keeps absorbing 'present perfect', 'present pefect' and
the rest.

Per topic it adds questions in the topic's (language, subject) cell that carry
one of its titles but were not listed, and drops listed doc_ids that no longer
exist in that cell. Only rows whose doc_ids change are rewritten. Such a row
keeps its existing doc_id order with the added ids appended, keeps its sample
question if that question is still in the topic, and has its count-derived
columns recomputed as the eval_dataset notebooks compute them — so its diff is
an addition, not a reshuffle. Every other row, the byte-order mark and the line
endings are left as they were, and a file is not written unless re-rendering it
unchanged reproduces it byte for byte.

The eval reads only quiz_title and doc_ids. The other columns are descriptive,
for people browsing the CSV, and on disk they often reflect the build the CSV
was first made from; they are not refreshed on rows whose doc_ids are unchanged.

A plain refresh never merges new spellings. That only happens through a
title-alias file passed with --aliases.

Title aliases (steps 2 and 3 of the eval repair)
------------------------------------------------
--propose-safe-aliases lists titles that differ from an eval topic's titles
only in case, spacing, punctuation or a leading article, and with --write
saves them as YAML. Judgement merges (typos, numbered parts, subtitles) belong
in a separate, reviewed file. Keeping the two apart, and applying each in its
own run, lets the effect of each step on the eval be measured on its own.

Usage:
    python -m scripts.eval.refresh_topics                          # dry run, all cells
    python -m scripts.eval.refresh_topics --cell en:ENGLISH --write
    python -m scripts.eval.refresh_topics --propose-safe-aliases eval/topic_aliases.safe.yaml --write
    python -m scripts.eval.refresh_topics --cell en:ENGLISH --aliases eval/topic_aliases.safe.yaml --write
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.eval.validate_test_cases import (
    DEFAULT_READY_JSONL,
    TOPICS_FILE_BY_LANG_SUBJECT,
    IndexDoc,
    check_ground_truth_against_index,
    load_index,
)

COLUMNS = [
    "quiz_title",
    "n_questions",
    "school_phases",
    "levels",
    "n_levels",
    "avg_q_len",
    "sample_question",
    "sample_choices",
    "doc_ids",
]
BOM = "\ufeff"
COLLISION_SUFFIX = re.compile(r"__q\d+_\d+$")


@dataclass(frozen=True)
class CsvFormat:
    """The byte-level conventions of one topics CSV, preserved on write."""

    has_bom: bool
    lineterminator: str


@dataclass(frozen=True)
class TopicChange:
    """How one topic's doc_ids moved during a refresh."""

    topic: str
    added: list[str]
    removed: list[str]


@dataclass(frozen=True)
class RefreshResult:
    """A cell's refreshed CSV rows, in original order, plus what changed."""

    cell: tuple[str, str]
    rows: list[dict[str, str]]
    changes: list[TopicChange]


def read_topics_csv(path: Path) -> tuple[list[dict[str, str]], CsvFormat, str]:
    """Return (rows as raw strings, format, original text) for a topics CSV."""
    text = path.read_bytes().decode("utf-8")
    fmt = CsvFormat(has_bom=text.startswith(BOM), lineterminator="\r\n" if "\r\n" in text else "\n")
    body = text[1:] if fmt.has_bom else text
    reader = csv.DictReader(io.StringIO(body, newline=""))
    if reader.fieldnames != COLUMNS:
        raise ValueError(f"{path}: unexpected columns {reader.fieldnames}, expected {COLUMNS}")
    return list(reader), fmt, text


def render_topics_csv(rows: list[dict[str, str]], fmt: CsvFormat) -> str:
    """Serialise rows the way pandas.to_csv wrote the originals."""
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator=fmt.lineterminator)
    writer.writeheader()
    writer.writerows(rows)
    return (BOM if fmt.has_bom else "") + buf.getvalue()


def topic_stats(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Stats columns for one topic, computed as the eval_dataset notebooks do."""
    df = pd.DataFrame(rows)
    first_level = df["levels"].apply(lambda levels: levels[0] if levels else None)
    avg_q_len = pd.Series([df["question_text"].apply(len).mean()]).round(1).iloc[0]
    return {
        "n_questions": str(len(df)),
        "school_phases": ",".join(sorted(set(df["school_phase"]))),
        "levels": ",".join(sorted({str(v) for v in first_level if v})),
        "n_levels": str(len(set(first_level))),
        "avg_q_len": str(float(avg_q_len)),
        "sample_question": str(df["question_text"].iloc[0]),
        "sample_choices": " | ".join(df["choices_text"].iloc[0]),
        "doc_ids": ",".join(df["doc_id"]),
    }


def _ordered_ids(raw: str) -> list[str]:
    return list(dict.fromkeys(d.strip() for d in raw.split(",") if d.strip()))


def _rewrite_row(row: dict[str, str], docs: list[dict[str, Any]]) -> dict[str, str]:
    """Recompute a changed row's columns, keeping its sample question when still valid."""
    stats = topic_stats(docs)
    if any(doc["question_text"] == row["sample_question"] for doc in docs):
        stats["sample_question"] = row["sample_question"]
        stats["sample_choices"] = row["sample_choices"]
    return {**row, **stats}


def refresh_cell(
    cell: tuple[str, str],
    csv_rows: list[dict[str, str]],
    index: dict[str, IndexDoc],
    full_rows: dict[str, dict[str, Any]],
    aliases: dict[str, str] | None = None,
) -> RefreshResult:
    """Return the cell's CSV rows with every topic's doc_ids matching the index.

    A changed row keeps its existing doc_id order and appends added ids in index
    order, so reviewing the refreshed file shows additions rather than churn.
    `aliases` maps extra titles onto existing topics; see _alias_additions.
    """
    listed_order = {row["quiz_title"]: _ordered_ids(row["doc_ids"]) for row in csv_rows}
    listed = {topic: set(ids) for topic, ids in listed_order.items()}
    if len(listed) != len(csv_rows):
        raise ValueError(f"{cell}: duplicate quiz_title rows in the topics CSV")
    problems = check_ground_truth_against_index({cell: listed}, index)
    added: dict[str, set[str]] = defaultdict(set)
    removed: dict[str, set[str]] = defaultdict(set)
    for issue in problems.unknown_ids + problems.wrong_cell_ids:
        removed[issue.topic].add(issue.doc_id)
    for issue in problems.unlisted_docs:
        added[issue.topic].add(issue.doc_id)
    for topic, doc_ids in _alias_additions(cell, csv_rows, listed, index, aliases or {}).items():
        added[topic] |= doc_ids

    position = {doc_id: i for i, doc_id in enumerate(full_rows)}
    new_rows: list[dict[str, str]] = []
    changes: list[TopicChange] = []
    for row in csv_rows:
        topic = row["quiz_title"]
        if topic not in added and topic not in removed:
            new_rows.append(row)
            continue
        kept = [d for d in listed_order[topic] if d not in removed[topic]]
        ids = kept + sorted(added[topic], key=position.__getitem__)
        if not ids:
            raise ValueError(f"{cell}: topic {topic!r} would be left with no questions")
        new_rows.append(_rewrite_row(row, [full_rows[d] for d in ids]))
        changes.append(TopicChange(topic, sorted(added[topic]), sorted(removed[topic])))
    return RefreshResult(cell, new_rows, changes)


# ---------------------------------------------------------------------------
# Title aliases
# ---------------------------------------------------------------------------

LEADING_ARTICLE = {"en": re.compile(r"^(?:the|a|an)\s+(?=\w)")}


@dataclass(frozen=True)
class AliasProposal:
    """Safe-rule title aliases for one cell, plus the groups it refused to merge."""

    cell: tuple[str, str]
    aliases: dict[str, str]  # variant title -> existing topic
    conflicts: list[tuple[list[str], list[str]]]  # (claiming topics, titles)


def safe_title_key(title: str, language: str) -> str:
    """Key shared by titles that differ only in case, spacing, punctuation or article.

    Deliberately narrow: no typo matching, no part numbers, no subtitles, so
    'Greeting' and 'Greetings' stay apart. A leading article is removed only
    when a word follows it, so 'The + adjective' does not collapse into
    'adjective'. Punctuation is found by Unicode category rather than by a
    word-character regex, which would also delete Arabic vowel marks.
    """
    key = unicodedata.normalize("NFKC", title).casefold().strip()
    article = LEADING_ARTICLE.get(language)
    if article is not None:
        key = article.sub("", key)
    key = key.replace("&", " and ")
    key = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in key)
    return " ".join(key.split())


def _title_owners(
    cell: tuple[str, str], csv_rows: list[dict[str, str]], index: dict[str, IndexDoc]
) -> dict[str, set[str]]:
    """Map titles to the topics owning them: a topic's name and its doc_ids' titles."""
    owners: dict[str, set[str]] = defaultdict(set)
    for row in csv_rows:
        owners[row["quiz_title"]].add(row["quiz_title"])
        for doc_id in _ordered_ids(row["doc_ids"]):
            doc = index.get(doc_id)
            if doc is not None and (doc.language, doc.subject) == cell:
                owners[doc.quiz_title].add(row["quiz_title"])
    return owners


def propose_safe_aliases(
    cell: tuple[str, str], csv_rows: list[dict[str, str]], index: dict[str, IndexDoc]
) -> AliasProposal:
    """Propose aliases for titles that differ from one topic's titles only superficially.

    A group of titles sharing a safe_title_key is merged only when exactly one
    topic owns any of them. Groups owned by several topics are reported as
    conflicts; groups no topic owns are outside the eval and ignored.
    """
    owners = _title_owners(cell, csv_rows, index)
    groups: dict[str, list[str]] = defaultdict(list)
    for title in sorted({d.quiz_title for d in index.values() if (d.language, d.subject) == cell}):
        groups[safe_title_key(title, cell[0])].append(title)
    aliases: dict[str, str] = {}
    conflicts: list[tuple[list[str], list[str]]] = []
    for titles in groups.values():
        topics = set().union(*(owners.get(t, set()) for t in titles))
        if len(titles) < 2 or not topics:
            continue
        if len(topics) > 1:
            conflicts.append((sorted(topics), titles))
            continue
        (topic,) = topics
        aliases.update({t: topic for t in titles if not owners.get(t)})
    return AliasProposal(cell, aliases, conflicts)


def _alias_additions(
    cell: tuple[str, str],
    csv_rows: list[dict[str, str]],
    listed: dict[str, set[str]],
    index: dict[str, IndexDoc],
    aliases: dict[str, str],
) -> dict[str, set[str]]:
    """Doc_ids each topic gains from title aliases; refuses ambiguous merges."""
    if not aliases:
        return {}
    owners = _title_owners(cell, csv_rows, index)
    additions: dict[str, set[str]] = defaultdict(set)
    for variant, topic in sorted(aliases.items()):
        if topic not in listed:
            raise ValueError(f"{cell}: alias {variant!r} points at unknown topic {topic!r}")
        claimed = owners.get(variant, set()) - {topic}
        if claimed:
            raise ValueError(
                f"{cell}: alias {variant!r} -> {topic!r}, but {sorted(claimed)} already own that title"
            )
        for doc_id, doc in index.items():
            if (doc.language, doc.subject) == cell and doc.quiz_title == variant:
                additions[topic].add(doc_id)
        additions[topic] -= listed[topic]
    return additions


def dump_aliases(proposals: list[AliasProposal]) -> str:
    """Serialise proposals as alias YAML, with a provenance header."""
    body = {
        f"{p.cell[0]}:{p.cell[1]}": dict(sorted(p.aliases.items())) for p in proposals if p.aliases
    }
    header = (
        "# Title aliases generated by the safe rule in scripts/eval/refresh_topics.py:\n"
        "# titles that differ from an eval topic's titles only in case, spacing,\n"
        "# punctuation or a leading article. Regenerate rather than hand-edit;\n"
        "# reviewed judgement merges belong in a separate file.\n"
    )
    return header + yaml.safe_dump(body, allow_unicode=True, sort_keys=True)


def load_aliases(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Read alias YAML of the form {"lang:SUBJECT": {variant_title: topic}}."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        _parse_cell(str(cell)): {str(k): str(v) for k, v in (mapping or {}).items()}
        for cell, mapping in data.items()
    }


def _merge_alias_files(paths: list[Path]) -> dict[tuple[str, str], dict[str, str]]:
    merged: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for path in paths:
        for cell, mapping in load_aliases(path).items():
            for variant, topic in mapping.items():
                previous = merged[cell].setdefault(variant, topic)
                if previous != topic:
                    raise ValueError(
                        f"{cell}: {variant!r} aliased to both {previous!r} and {topic!r}"
                    )
    return merged


def load_full_rows(ready_path: Path) -> dict[str, dict[str, Any]]:
    """Read every payload row keyed by doc_id, in file order."""
    rows: dict[str, dict[str, Any]] = {}
    with ready_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["doc_id"]] = row
    return rows


def _report(result: RefreshResult, path: Path, is_lossless: bool, n_aliases: int) -> None:
    n_added = sum(len(c.added) for c in result.changes)
    n_removed = sum(len(c.removed) for c in result.changes)
    n_suffixed = sum(1 for c in result.changes for d in c.added if COLLISION_SUFFIX.search(d))
    print(f"[{result.cell[0]} x {result.cell[1]}] {path}")
    print(
        f"  topics: {len(result.rows)}  changed: {len(result.changes)}  aliases given: {n_aliases}"
    )
    print(f"  doc_ids: +{n_added} added ({n_suffixed} collision-suffixed)  -{n_removed} removed")
    for change in sorted(result.changes, key=lambda c: -len(c.added) - len(c.removed))[:10]:
        print(f"    +{len(change.added):<3} -{len(change.removed):<2} {change.topic!r}")
    print(f"  unchanged file re-renders byte-identical: {'yes' if is_lossless else 'NO'}")


def _refresh_file(
    cell: tuple[str, str],
    index: dict[str, IndexDoc],
    full_rows: dict[str, dict[str, Any]],
    aliases: dict[str, str],
    is_write: bool,
    stamp: str,
) -> bool:
    """Refresh one cell's CSV; return False if it could not be written safely."""
    path = TOPICS_FILE_BY_LANG_SUBJECT[cell]
    if not path.exists():
        print(f"[{cell[0]} x {cell[1]}] {path} not found, skipped")
        return True
    rows, fmt, original = read_topics_csv(path)
    result = refresh_cell(cell, rows, index, full_rows, aliases)
    is_lossless = render_topics_csv(rows, fmt) == original
    _report(result, path, is_lossless, len(aliases))
    if not result.changes:
        print("  up to date, nothing to write\n")
        return True
    if not is_write:
        print("  DRY RUN, nothing written (pass --write)\n")
        return True
    if not is_lossless:
        print("  REFUSED: re-rendering would reformat untouched rows\n")
        return False
    backup = path.with_name(f"{path.stem}.backup-{stamp}.csv")
    backup.write_bytes(original.encode("utf-8"))
    path.write_bytes(render_topics_csv(result.rows, fmt).encode("utf-8"))
    print(f"  WROTE {path}  (previous version: {backup})\n")
    return True


def _propose(
    cells: list[tuple[str, str]], index: dict[str, IndexDoc], out_path: Path, is_write: bool
) -> int:
    """Print safe-rule alias proposals per cell; with --write, save them as YAML."""
    proposals: list[AliasProposal] = []
    for cell in cells:
        path = TOPICS_FILE_BY_LANG_SUBJECT[cell]
        if not path.exists():
            continue
        rows, _fmt, _original = read_topics_csv(path)
        proposal = propose_safe_aliases(cell, rows, index)
        proposals.append(proposal)
        sizes = Counter(d.quiz_title for d in index.values() if (d.language, d.subject) == cell)
        print(
            f"[{cell[0]} x {cell[1]}] {len(proposal.aliases)} aliases, {len(proposal.conflicts)} conflicts"
        )
        for variant, topic in sorted(proposal.aliases.items(), key=lambda kv: -sizes[kv[0]]):
            print(f"    +{sizes[variant]:<3} {topic!r} <- {variant!r}")
        for topics, titles in proposal.conflicts:
            print(f"    CONFLICT, not merged: {titles} owned by {topics}")
    if not is_write:
        print(f"\nDRY RUN, {out_path} not written (pass --write)")
        return 0
    out_path.write_text(dump_aliases(proposals), encoding="utf-8")
    print(f"\nWROTE {out_path}")
    return 0


def _parse_cell(value: str) -> tuple[str, str]:
    language, _, subject = value.partition(":")
    cell = (language, subject)
    if cell not in TOPICS_FILE_BY_LANG_SUBJECT:
        raise argparse.ArgumentTypeError(
            f"unknown cell {value!r}; registered: "
            + ", ".join(f"{la}:{su}" for la, su in sorted(TOPICS_FILE_BY_LANG_SUBJECT))
        )
    return cell


def main(argv: list[str] | None = None) -> int:
    """Refresh topics CSVs from the index, or propose safe title aliases."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cell",
        action="append",
        type=_parse_cell,
        metavar="LANG:SUBJECT",
        help="Cell to work on, e.g. en:ENGLISH. Repeatable. Default: every registered cell.",
    )
    parser.add_argument("--ready-jsonl", type=Path, default=DEFAULT_READY_JSONL)
    parser.add_argument(
        "--aliases",
        action="append",
        type=Path,
        default=[],
        metavar="YAML",
        help="Title-alias file to apply while refreshing. Repeatable.",
    )
    parser.add_argument(
        "--propose-safe-aliases",
        type=Path,
        metavar="YAML",
        help="Instead of refreshing, propose safe-rule aliases; with --write, save them here.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes. Default is a dry run. A refreshed CSV keeps its previous version "
        "as <name>.backup-<UTC>.csv.",
    )
    args = parser.parse_args(argv)
    cells = args.cell or sorted(TOPICS_FILE_BY_LANG_SUBJECT)
    index = load_index(args.ready_jsonl)
    if args.propose_safe_aliases is not None:
        return _propose(cells, index, args.propose_safe_aliases, args.write)
    aliases = _merge_alias_files(args.aliases)
    full_rows = load_full_rows(args.ready_jsonl)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results = [
        _refresh_file(cell, index, full_rows, aliases.get(cell, {}), args.write, stamp)
        for cell in cells
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
