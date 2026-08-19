"""Run retriever eval against an LLM-generated test-cases JSON.

Pipeline:

  1. Validate the test-cases JSON. If anything fails (schema / missing
     target_quiz_title / subject mismatch), exit non-zero — DO NOT spend
     30+ minutes running on broken input.

  2. Load the production Retriever (configs/models.yaml). The active config
     in that file IS the configuration being evaluated. To compare runs
     (reranker on vs off, different max_distance, etc.) flip the config and
     re-run.

  3. Load ground truth from `eval/topics_<lang>.csv`: each row maps
     (language, quiz_title) -> set of doc_ids that share that title. Those
     are the relevant docs for any query about that title.

  4. For each test case, retrieve `max(test_case.top_k, K_RETRIEVE_MAX=10)`
     docs in ONE call, then compute metrics at multiple k values
     (k=1, 3, 5, 10) by truncating — same retrieval cost, more analysis.

     Compute primary metrics (at the test case's own top_k) and MRR.

  5. Write three artifacts to a versioned output folder
     `eval/results/<lang_tag>_<UTC_TIMESTAMP>/`:
       - per_query.jsonl       — one row per test case, all metrics
       - summary.json          — aggregated by language / query_type / top_k
       - config_snapshot.yaml  — copy of configs/models.yaml that produced these numbers

CRITICAL: calls retriever with dedup_by_quiz_title=False. The default
True would cap hits at 1 per quiz_title (since all relevant docs share
the title), making recall@k look artificially terrible.

Usage:
    # Quick sanity test on 50 cases
    python -m scripts.eval.run_retriever_eval eval/english_retriever_test_cases.json --limit 50

    # Full run
    python -m scripts.eval.run_retriever_eval eval/english_retriever_test_cases.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.eval.validate_test_cases import (
    TestCase,
    check_subject_consistency,
    cross_check,
    load_topic_index,
    parse_cases,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_DIR = Path("eval")
RESULTS_DIR = EVAL_DIR / "results"

# Topics CSV path per (language, subject) pair. Math adds a second subject
# per language (Tunisian curriculum: HS math → fr, middle/primary math → ar),
# so the relationship isn't 1:1.
TOPICS_FILE_BY_LANG_SUBJECT: dict[tuple[str, str], Path] = {
    ("en", "ENGLISH"):     EVAL_DIR / "topics_english.csv",
    ("ar", "ARABIC"):      EVAL_DIR / "topics_arabic.csv",
    ("fr", "FRENCH"):      EVAL_DIR / "topics_french.csv",
    ("ar", "MATHEMATICS"): EVAL_DIR / "topics_math_ar.csv",
    ("fr", "MATHEMATICS"): EVAL_DIR / "topics_math_fr.csv",
}

# Retrieve at least this many docs per call, regardless of test-case top_k —
# lets us compute precision@1/3/5/10 from the same single retrieval.
K_RETRIEVE_MAX = 10

# k values reported in the per-query metrics.
K_VALUES = (1, 3, 5, 10)

DEFAULT_CONFIG_PATH = Path("configs/models.yaml")
DEFAULT_READY_JSONL = Path("data/processed/ready_phase1.jsonl")


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def load_ground_truth(
    lang_subject_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], set[str]]:
    """Return {(language, quiz_title) -> set(doc_id)} for all relevant docs.

    Loads one topics CSV per (language, subject) pair present in the test
    set. Each language can have multiple subjects (e.g. fr → FRENCH and
    MATHEMATICS), each backed by its own topics CSV.
    """
    gt: dict[tuple[str, str], set[str]] = {}
    for lang, subject in sorted(lang_subject_pairs):
        path = TOPICS_FILE_BY_LANG_SUBJECT.get((lang, subject))
        if path is None or not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        for _, row in df.iterrows():
            doc_ids = str(row["doc_ids"]) if pd.notna(row["doc_ids"]) else ""
            ids = {d.strip() for d in doc_ids.split(",") if d.strip()}
            gt[(lang, row["quiz_title"])] = ids
    return gt


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------


def precision_recall_hit_at_k(
    retrieved_doc_ids: list[str], relevant: set[str], k: int
) -> tuple[float, float, int]:
    """Standard precision@k / recall@k / hit@k for one query.

    Hit@k is binary — 1 if at least one relevant doc appears in the top-k,
    else 0. Useful for quick "did we find anything?" rate.
    """
    if k <= 0:
        return 0.0, 0.0, 0
    top = retrieved_doc_ids[:k]
    n_hits = sum(1 for d in top if d in relevant)
    precision = n_hits / k
    recall = n_hits / len(relevant) if relevant else 0.0
    hit = 1 if n_hits > 0 else 0
    return precision, recall, hit


def mrr(retrieved_doc_ids: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank — 1 / position of first relevant doc, else 0."""
    for i, d in enumerate(retrieved_doc_ids, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


def run_eval(
    test_cases: list[TestCase],
    retriever,
    ground_truth: dict[tuple[str, str], set[str]],
) -> list[dict]:
    """Call the retriever once per test case, compute all metrics."""
    per_query: list[dict] = []
    n = len(test_cases)
    start = time.time()
    print_every = max(1, n // 20)  # ~20 progress prints per run

    for i, c in enumerate(test_cases):
        if i % print_every == 0 and i > 0:
            elapsed = time.time() - start
            rate = i / elapsed
            eta = (n - i) / rate if rate > 0 else 0
            print(
                f"  [{i}/{n}] {elapsed:.0f}s elapsed, "
                f"~{eta:.0f}s remaining ({rate:.1f} q/s)"
            )

        relevant = ground_truth.get((c.language, c.target_quiz_title), set())

        # Retrieve enough to compute metrics at all K_VALUES — cheap to over-
        # retrieve since the bottleneck is the query embedding.
        retrieve_k = max(c.top_k, K_RETRIEVE_MAX)

        diag: dict = {}
        try:
            # Forward the optional level filter from the test case. When the
            # test case doesn't set `levels`, retrieve() falls back to no
            # level filter (same as the pre-level-aware behaviour). This
            # mirrors how the production API receives `levels` from the
            # platform based on the user's grade.
            results = retriever.retrieve(
                query=c.query,
                language=c.language,
                top_k=retrieve_k,
                dedup_by_quiz_title=False,  # see module docstring
                levels=c.levels,
                levels_match_mode=(c.levels_match or "any"),
                diagnostics=diag,
            )
            retrieved_doc_ids = [r.doc_id for r in results]
            # Per-doc cosine distance from the bi-encoder — preserved through
            # the reranker (the reranker only reorders, doesn't replace).
            # Order here is FINAL (post-rerank) order, not distance-ascending.
            retrieved_distances = [float(r.distance) for r in results]
            error = None
        except Exception as e:  # noqa: BLE001 — capture the error per-query
            retrieved_doc_ids = []
            retrieved_distances = []
            error = f"{type(e).__name__}: {e}"

        row = {
            "query_id": f"{c.language}-{i:04d}",
            "language": c.language,
            "subject": c.subject,
            "query_type": c.query_type,
            "query": c.query,
            "target_quiz_title": c.target_quiz_title,
            "primary_top_k": c.top_k,
            # Record the level filter actually used so post-hoc analysis can
            # tell which queries ran scoped vs unscoped.
            "levels": list(c.levels) if c.levels else None,
            "levels_match": c.levels_match,
            "n_relevant": len(relevant),
            "n_retrieved": len(retrieved_doc_ids),
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_distances": retrieved_distances,
            # Raw bi-encoder ranking (the reranker's input pool, pre-filter,
            # pre-rerank, ascending by distance). Lets us answer:
            #   - "Would a bigger pool catch the relevant doc?"
            #   - "What threshold would drop garbage without losing recall?"
            "bi_encoder_top_doc_ids": diag.get("bi_encoder_doc_ids", []),
            "bi_encoder_top_distances": diag.get("bi_encoder_distances", []),
            "pool_size": diag.get("pool_size"),
            # Reranker scores for the FINAL returned docs (same order as
            # retrieved_doc_ids). Empty list when reranker is disabled.
            "reranker_scores": diag.get("reranker_scores", []),
            "reranker_enabled": diag.get("reranker_enabled", False),
            # Full post-rerank pool (parallel arrays). Order: score-descending
            # when reranker is on, distance-ascending when off. Lets threshold
            # analysis sweep over the entire pool, not just the visible top-k.
            "reranker_full_doc_ids": diag.get("reranker_full_doc_ids", []),
            "reranker_full_scores": diag.get("reranker_full_scores", []),
            "error": error,
        }

        # Multi-k breakdown.
        for k in K_VALUES:
            p, r, h = precision_recall_hit_at_k(retrieved_doc_ids, relevant, k)
            row[f"precision_at_{k}"] = p
            row[f"recall_at_{k}"] = r
            row[f"hit_at_{k}"] = h

        # Primary k (whatever the test case asked for).
        p, r, h = precision_recall_hit_at_k(retrieved_doc_ids, relevant, c.top_k)
        row["precision_at_primary"] = p
        row["recall_at_primary"] = r
        row["hit_at_primary"] = h

        row["mrr"] = mrr(retrieved_doc_ids, relevant)

        per_query.append(row)

    elapsed = time.time() - start
    print(f"  done in {elapsed:.0f}s ({n / elapsed:.1f} q/s)")
    return per_query


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

# Metric columns we report aggregates for.
SUMMARY_METRICS = (
    [f"precision_at_{k}" for k in K_VALUES]
    + [f"recall_at_{k}" for k in K_VALUES]
    + [f"hit_at_{k}" for k in K_VALUES]
    + ["precision_at_primary", "recall_at_primary", "hit_at_primary", "mrr"]
)


def _summarize(rows: list[dict]) -> dict:
    """Mean / median per metric, plus n and error count."""
    n = len(rows)
    n_errors = sum(1 for r in rows if r.get("error"))
    out: dict = {"n": n, "n_errors": n_errors}
    for m in SUMMARY_METRICS:
        vals = [r[m] for r in rows if r.get(m) is not None and not r.get("error")]
        if not vals:
            out[m] = {"mean": None, "median": None}
            continue
        out[m] = {
            "mean": round(sum(vals) / len(vals), 4),
            "median": round(statistics.median(vals), 4),
        }
    return out


def aggregate(per_query: list[dict]) -> dict:
    """Per-language / per-query_type / per-top_k summary blocks."""
    by_lang: dict[str, list[dict]] = defaultdict(list)
    by_query_type: dict[str, list[dict]] = defaultdict(list)
    by_top_k: dict[int, list[dict]] = defaultdict(list)

    for q in per_query:
        by_lang[q["language"]].append(q)
        by_query_type[q["query_type"]].append(q)
        by_top_k[q["primary_top_k"]].append(q)

    return {
        "n_total": len(per_query),
        "n_errors": sum(1 for q in per_query if q.get("error")),
        "overall": _summarize(per_query),
        "by_language": {k: _summarize(v) for k, v in sorted(by_lang.items())},
        "by_query_type": {
            k: _summarize(v) for k, v in sorted(by_query_type.items())
        },
        "by_top_k": {str(k): _summarize(v) for k, v in sorted(by_top_k.items())},
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_results(
    out_dir: Path,
    per_query: list[dict],
    summary: dict,
    config_path: Path,
    args_record: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for r in per_query:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if config_path.exists():
        shutil.copy(config_path, out_dir / "config_snapshot.yaml")

    with (out_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(args_record, f, ensure_ascii=False, indent=2)


def print_headline(summary: dict) -> None:
    """Quick console summary so the user knows the result without opening files."""
    print()
    print("-" * 70)
    print(
        f"Overall ({summary['n_total']} queries, {summary['n_errors']} errors):"
    )
    o = summary["overall"]
    print(
        f"  precision@1={o['precision_at_1']['mean']}  "
        f"@3={o['precision_at_3']['mean']}  "
        f"@5={o['precision_at_5']['mean']}  "
        f"@10={o['precision_at_10']['mean']}"
    )
    print(
        f"  recall@1   ={o['recall_at_1']['mean']}  "
        f"@3={o['recall_at_3']['mean']}  "
        f"@5={o['recall_at_5']['mean']}  "
        f"@10={o['recall_at_10']['mean']}"
    )
    print(
        f"  hit@1      ={o['hit_at_1']['mean']}  "
        f"@3={o['hit_at_3']['mean']}  "
        f"@5={o['hit_at_5']['mean']}  "
        f"@10={o['hit_at_10']['mean']}"
    )
    print(f"  mrr        ={o['mrr']['mean']}")
    print()
    if "by_query_type" in summary:
        print("By query_type (mean precision@5):")
        for qt, s in summary["by_query_type"].items():
            v = s.get("precision_at_5", {}).get("mean")
            print(f"  {qt:<20} {v}  (n={s['n']})")
    print()


# ---------------------------------------------------------------------------
# Validation gate (reuses scripts.eval.validate_test_cases helpers)
# ---------------------------------------------------------------------------


def validate_or_die(test_cases_path: Path) -> list[TestCase]:
    """Load + validate. Print problems and exit 1 on any failure."""
    if not test_cases_path.exists():
        print(f"error: file not found: {test_cases_path}", file=sys.stderr)
        sys.exit(1)

    with test_cases_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        print("error: top-level JSON must be a list.", file=sys.stderr)
        sys.exit(1)

    cases, schema_errors = parse_cases(raw)
    lang_subject_pairs = {(c.language, c.subject) for c in cases}
    topic_index = load_topic_index(lang_subject_pairs)
    missing, _recall_capped = cross_check(cases, topic_index)
    subject_mismatches = check_subject_consistency(cases)

    problems = []
    if schema_errors:
        problems.append(f"{len(schema_errors)} schema errors")
    if missing:
        problems.append(f"{len(missing)} missing target_quiz_title")
    if subject_mismatches:
        problems.append(f"{len(subject_mismatches)} subject mismatches")

    if problems:
        print(
            "VALIDATION FAILED: " + ", ".join(problems) + ".",
            file=sys.stderr,
        )
        print(
            "Run `python -m scripts.eval.validate_test_cases "
            f"{test_cases_path}` for details.",
            file=sys.stderr,
        )
        sys.exit(1)

    return cases


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run retriever eval against an LLM-generated test-cases JSON."
    )
    parser.add_argument(
        "test_cases",
        type=Path,
        help="Path to test-cases JSON (e.g. eval/english_retriever_test_cases.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases (sanity check before full run).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override results folder. Default: eval/results/<lang>_<utc_timestamp>/",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Models config to use (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--ready-jsonl",
        type=Path,
        default=DEFAULT_READY_JSONL,
        help=f"Payload JSONL (default: {DEFAULT_READY_JSONL}).",
    )
    args = parser.parse_args(argv)

    print(f"Validating {args.test_cases} ...")
    cases = validate_or_die(args.test_cases)
    print(f"  OK: {len(cases)} cases valid")

    if args.limit is not None and args.limit < len(cases):
        cases = cases[: args.limit]
        print(f"  Limited to first {len(cases)} cases (--limit).")

    print("Loading retriever (model + vector store + payload) ...")
    # Local import — keeps validation/CLI cheap if the user just wants -h.
    from src.retrieval.retriever import Retriever

    retriever = Retriever(
        config_path=args.config,
        ready_jsonl_path=args.ready_jsonl,
    )
    print("  Loaded.")

    lang_subject_pairs = {(c.language, c.subject) for c in cases}
    print(f"Loading ground truth for: {sorted(lang_subject_pairs)}")
    ground_truth = load_ground_truth(lang_subject_pairs)
    print(f"  {len(ground_truth)} (language, quiz_title) pairs in ground truth")

    print(f"Running eval on {len(cases)} cases ...")
    per_query = run_eval(cases, retriever, ground_truth)

    print("Aggregating ...")
    summary = aggregate(per_query)

    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        languages = sorted({l for l, _s in lang_subject_pairs})
        lang_tag = "-".join(languages)
        out_dir = RESULTS_DIR / f"{lang_tag}_{timestamp}"

    args_record = {
        "test_cases": str(args.test_cases),
        "limit": args.limit,
        "config": str(args.config),
        "ready_jsonl": str(args.ready_jsonl),
        "k_retrieve_max": K_RETRIEVE_MAX,
        "k_values": list(K_VALUES),
        "n_cases_run": len(cases),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    write_results(out_dir, per_query, summary, args.config, args_record)
    print(f"Results written to {out_dir}/")

    print_headline(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
