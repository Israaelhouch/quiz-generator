"""Layer 1 evaluation — automatable metrics over runs.jsonl.

Reads `logs/runs.jsonl` (auto-populated by every /quiz/generate call)
and prints a numerical summary covering:

  * Latency: min, median, mean, p95, max
  * Retrieval distance: min, median, mean, p75, p95, max
  * Per-query distance stats (best/worst chunks)
  * Distance distribution histogram (helps pick a threshold)
  * Success rate, count-match rate
  * Top slowest queries
  * Top worst-matched queries (highest min distance)

Usage:
    python -m scripts.analyze_runs                  # default: logs/runs.jsonl
    python -m scripts.analyze_runs path/to/file.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def _percentile(sorted_vals: list[float], p: float) -> float:
    """p in [0, 1]. Returns the value at percentile p."""
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(p * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _hr() -> str:
    return "─" * 70


def analyze(log_path: Path) -> None:
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        sys.exit(1)

    runs: list[dict] = []
    with log_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"⚠ Skipping malformed line {line_num}: {exc}")

    if not runs:
        print("❌ No runs found in the log.")
        sys.exit(1)

    print()
    print(_hr())
    print(f"  RUN-LOG ANALYSIS — {log_path}")
    print(f"  total runs: {len(runs)}")
    print(_hr())

    # ----------------------------------------------------------------------
    # Success rate / failure modes
    # ----------------------------------------------------------------------
    successes = [r for r in runs if r.get("response", {}).get("questions")]
    failures = [r for r in runs if not r.get("response", {}).get("questions")]

    count_matches = [
        r for r in successes
        if len(r["response"]["questions"]) == r["request"]["count"]
    ]

    print("\n📊 SUCCESS / FAILURE")
    print(f"  successes:        {len(successes)}/{len(runs)}  "
          f"({len(successes)/len(runs)*100:.1f}%)")
    print(f"  failures:         {len(failures)}/{len(runs)}")
    if successes:
        print(f"  count-match rate: {len(count_matches)}/{len(successes)}  "
              f"({len(count_matches)/len(successes)*100:.1f}%)")

    # ----------------------------------------------------------------------
    # Latency
    # ----------------------------------------------------------------------
    durations = [
        r["response"].get("duration_seconds")
        for r in runs
        if r.get("response", {}).get("duration_seconds") is not None
    ]
    if durations:
        sd = sorted(durations)
        print("\n⏱  LATENCY (seconds)")
        print(f"  min:    {min(durations):>7.2f}")
        print(f"  median: {statistics.median(durations):>7.2f}")
        print(f"  mean:   {statistics.mean(durations):>7.2f}")
        print(f"  p95:    {_percentile(sd, 0.95):>7.2f}")
        print(f"  max:    {max(durations):>7.2f}")

    # ----------------------------------------------------------------------
    # Distance — across ALL retrieved chunks
    # ----------------------------------------------------------------------
    all_distances: list[float] = []
    per_query_stats: list[dict] = []

    for r in runs:
        retrieval = r.get("response", {}).get("retrieval") or []
        if not retrieval:
            continue
        ds = [c["distance"] for c in retrieval if c.get("distance") is not None]
        if not ds:
            continue
        all_distances.extend(ds)
        per_query_stats.append({
            "topic": r["request"]["topic"],
            "min": min(ds),
            "max": max(ds),
            "avg": statistics.mean(ds),
            "spread": max(ds) - min(ds),
            "n": len(ds),
        })

    if all_distances:
        sa = sorted(all_distances)
        print(f"\n📏 RETRIEVAL DISTANCE — across all {len(all_distances)} chunks")
        print(f"  min:    {min(all_distances):>7.4f}    (best single match)")
        print(f"  median: {statistics.median(all_distances):>7.4f}")
        print(f"  mean:   {statistics.mean(all_distances):>7.4f}")
        print(f"  p75:    {_percentile(sa, 0.75):>7.4f}")
        print(f"  p95:    {_percentile(sa, 0.95):>7.4f}")
        print(f"  max:    {max(all_distances):>7.4f}    (worst single chunk)")

        # Best-match-per-query stats — i.e. how good is the TOP retrieved chunk
        best_per_q = [q["min"] for q in per_query_stats]
        worst_per_q = [q["max"] for q in per_query_stats]
        print(f"\n📏 BEST MATCH per query (n={len(best_per_q)})")
        print(f"  avg of mins:  {statistics.mean(best_per_q):>7.4f}")
        print(f"  range:        [{min(best_per_q):.4f}, {max(best_per_q):.4f}]")
        print(f"\n📏 WORST MATCH per query (top-K's tail)")
        print(f"  avg of maxs:  {statistics.mean(worst_per_q):>7.4f}")
        print(f"  range:        [{min(worst_per_q):.4f}, {max(worst_per_q):.4f}]")

        # ---- Histogram (for picking the threshold) ----
        print("\n📊 DISTANCE DISTRIBUTION (helps pick `max_distance` threshold)")
        bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        total = len(all_distances)
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            count = sum(1 for d in all_distances if lo <= d < hi)
            pct = count / total * 100
            bar = "█" * int(pct / 2)  # 50 chars max
            print(f"  {lo:.1f}–{hi:.1f}  {bar:<50s} {count:>4d}  ({pct:>5.1f}%)")

        # ---- Threshold guidance ----
        print("\n🎯 THRESHOLD GUIDANCE")
        # Where does the bulk of "useful" chunks sit?
        below_03 = sum(1 for d in all_distances if d < 0.3) / total * 100
        below_05 = sum(1 for d in all_distances if d < 0.5) / total * 100
        below_07 = sum(1 for d in all_distances if d < 0.7) / total * 100
        print(f"  {below_03:.1f}% of chunks have distance < 0.3 (strong matches)")
        print(f"  {below_05:.1f}% of chunks have distance < 0.5 (moderate or better)")
        print(f"  {below_07:.1f}% of chunks have distance < 0.7 (anything related)")
        print("  → If gen quality drops sharply above some value X, set max_distance≈X.")
        print("  → A safe default for educational content is often 0.55–0.65.")

    # ----------------------------------------------------------------------
    # Top slowest queries
    # ----------------------------------------------------------------------
    if durations and runs:
        sorted_by_speed = sorted(
            (r for r in runs if r.get("response", {}).get("duration_seconds")),
            key=lambda r: r["response"]["duration_seconds"],
            reverse=True,
        )
        print("\n🐌 TOP 5 SLOWEST QUERIES")
        for r in sorted_by_speed[:5]:
            d = r["response"]["duration_seconds"]
            topic = r["request"]["topic"][:55]
            print(f"  {d:>6.2f}s  |  {topic}")

    # ----------------------------------------------------------------------
    # Top worst-matched queries (high min distance = retriever struggled)
    # ----------------------------------------------------------------------
    if per_query_stats:
        worst = sorted(per_query_stats, key=lambda q: q["min"], reverse=True)
        print("\n🔍 TOP 5 WORST-MATCHED QUERIES (highest 'best chunk' distance)")
        for q in worst[:5]:
            topic = q["topic"][:50]
            print(f"  min={q['min']:.4f}  avg={q['avg']:.4f}  | {topic}")
        print("\n   These are the queries where even the best retrieved chunk was weak.")
        print("   → Either the corpus has no good match, or the embedder is failing on this concept.")

    # ----------------------------------------------------------------------
    # Per-language breakdown
    # ----------------------------------------------------------------------
    by_lang: dict[str, list[dict]] = {}
    for r in runs:
        lang = r["request"].get("language", "?")
        by_lang.setdefault(lang, []).append(r)
    if len(by_lang) > 1:
        print("\n🌐 PER-LANGUAGE BREAKDOWN")
        for lang, rs in sorted(by_lang.items()):
            ds = [
                r["response"]["duration_seconds"]
                for r in rs
                if r.get("response", {}).get("duration_seconds") is not None
            ]
            ok = sum(1 for r in rs if r.get("response", {}).get("questions"))
            avg_lat = statistics.mean(ds) if ds else float("nan")
            print(f"  {lang}:  {len(rs)} runs, {ok}/{len(rs)} success, "
                  f"avg latency {avg_lat:.2f}s")

    print()
    print(_hr())


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "logs/runs.jsonl"
    analyze(Path(arg))
