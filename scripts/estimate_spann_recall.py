#!/usr/bin/env python3
"""Estimate SPANN recall from a trace file using configurable centroid selection policies.

SPANN traces (output via --trace in et) contain per-query centroid observations. Each
centroid carries `tcnt` — the count of ground-truth vectors assigned to it.
By re-applying the same selection policies used at search time, we estimate what fraction
of the traced vectors would be found.

Input files ending with `.gz` are automatically gzip-decompressed.

Usage:
  # Top-N policy: select the 10 closest centroids
  uv run estimate_spann_recall.py trace.jsonl --policy top_n --top-n 10

  # Vector-count policy: select centroids until we cover 50k vectors
  uv run estimate_spann_recall.py trace.jsonl --policy vector_count --vector-count 50000

  # Gzipped input
  uv run estimate_spann_recall.py trace.jsonl.gz --policy top_n --top-n 10

  # JSON output
  uv run estimate_spann_recall.py trace.jsonl --policy top_n --top-n 10 --json
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "orjson",
# ]
# ///

import argparse
import gzip
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import orjson


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CentroidTrace:
    cid: int
    dist: float
    cnt: int
    tcnt: int


@dataclass
class QueryResult:
    query_index: int
    total_traced: int = 0
    estimated_found: int = 0
    recall: float = 0.0
    centroids_searched: int = 0
    vectors_covered: int = 0


# ---------------------------------------------------------------------------
# Policy interface & implementations
# ---------------------------------------------------------------------------


class CentroidPolicy(Protocol):
    """A centroid-selection policy with a human-readable name."""

    name: str

    def select(self, centroids: list[CentroidTrace]) -> list[CentroidTrace]:
        """Return the subset of centroids this policy selects."""
        ...

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> CentroidPolicy:
        """Build an instance from the CLI args dict."""
        ...


class TopNPolicy(CentroidPolicy):
    name: str

    def __init__(self, n: int) -> None:
        self._n = n
        self.name = f"top_n({n})"

    def select(self, centroids: list[CentroidTrace]) -> list[CentroidTrace]:
        sorted_c = sorted(centroids, key=lambda c: c.dist)
        return sorted_c[: self._n]

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> CentroidPolicy:
        return cls(args["top_n"])


class VectorCountPolicy(CentroidPolicy):
    name: str

    def __init__(self, vector_count: int) -> None:
        self._vector_count = vector_count
        self.name = f"vector_count({vector_count})"

    def select(self, centroids: list[CentroidTrace]) -> list[CentroidTrace]:
        sorted_c = sorted(centroids, key=lambda c: c.dist)
        selected: list[CentroidTrace] = []
        accumulated = 0
        for c in sorted_c:
            if accumulated >= self._vector_count:
                break
            selected.append(c)
            accumulated += c.cnt
        return selected

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> CentroidPolicy:
        return cls(args["vector_count"])


class AlphaPolicy(CentroidPolicy):
    name: str

    def __init__(self, vector_count: int, alpha: float) -> None:
        self._vector_count = vector_count
        self._alpha = alpha
        self.name = f"alpha({vector_count}, {alpha})"

    def select(self, centroids: list[CentroidTrace]) -> list[CentroidTrace]:
        sorted_c = sorted(centroids, key=lambda c: c.dist)
        selected: list[CentroidTrace] = []
        accumulated = 0
        for c in sorted_c:
            if accumulated >= self._vector_count:
                break
            selected.append(c)
            accumulated += c.cnt

        mean = sum(c.dist for c in selected) / len(selected)
        var = sum((c.dist - mean) ** 2 for c in selected) / len(selected)
        stddev = math.sqrt(var)

        selected = []
        accumulated = 0
        dist_limit = mean + stddev * self._alpha
        for c in sorted_c:
            if accumulated >= self._vector_count and c.dist > dist_limit:
                break
            selected.append(c)
            accumulated += c.cnt

        return selected

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> CentroidPolicy:
        return cls(args["vector_count"], args["alpha"])


class BetaPolicy(CentroidPolicy):
    name: str

    def __init__(self, sample_size: int, beta: float) -> None:
        self._sample_size = sample_size
        self._beta = beta
        self.name = f"beta({sample_size}, {beta})"

    def select(self, centroids: list[CentroidTrace]) -> list[CentroidTrace]:
        sorted_c = sorted(centroids, key=lambda c: c.dist)

        l = min(len(centroids), self._sample_size)
        mean = sum(c.dist for c in centroids[:l]) / l
        var = sum((c.dist - mean) ** 2 for c in centroids[:l]) / l
        stddev = math.sqrt(var)

        selected = []
        accumulated = 0
        m = 0
        dist_limit = mean + stddev * self._beta
        for c in sorted_c:
            if accumulated >= self._sample_size and c.dist > dist_limit:
                break
            selected.append(c)
            accumulated += c.cnt
            m += c.tcnt

        return selected

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> CentroidPolicy:
        return cls(args["top_n"], args["beta"])


class OraclePolicy(CentroidPolicy):
    name: str

    def __init__(self) -> None:
        self.name = f"oracle()"

    def select(self, centroids: list[CentroidTrace]) -> list[CentroidTrace]:
        sorted_c = sorted(centroids, key=lambda c: c.dist)
        last_idx = 0
        for i, c in enumerate(sorted_c):
            if c.tcnt > 0:
                last_idx = i
        return sorted_c[: last_idx + 1]

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> CentroidPolicy:
        return cls()


POLICY_MAP: dict[str, type[CentroidPolicy]] = {
    "top_n": TopNPolicy,
    "vector_count": VectorCountPolicy,
    "alpha": AlphaPolicy,
    "beta": BetaPolicy,
    "oracle": OraclePolicy,
}


def make_policy(name: str, args: dict[str, Any]) -> CentroidPolicy:
    """Instantiate a selection policy from its name and CLI args.

    Raises ValueError if *name* is unknown.
    """
    cls = POLICY_MAP.get(name)
    if cls is None:
        available = ", ".join(sorted(POLICY_MAP))
        raise ValueError(f"Unknown policy '{name}'. Available: {available}")
    return cls.from_args(args)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def load_centroid(obj: dict) -> CentroidTrace:
    return CentroidTrace(
        cid=int(obj["cid"]),
        dist=float(obj["dist"]),
        cnt=int(obj["cnt"]),
        tcnt=int(obj["tcnt"]),
    )


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------


def apply_policy(
    centroids: list[CentroidTrace],
    policy: CentroidPolicy,
    total_traced: int,
) -> QueryResult:
    if not centroids or total_traced == 0:
        return QueryResult(
            query_index=-1, total_traced=total_traced,
            estimated_found=0, recall=0.0,
            centroids_searched=0, vectors_covered=0,
        )

    selected = policy.select(centroids)
    estimated_found = sum(c.tcnt for c in selected)
    recall = estimated_found / total_traced

    return QueryResult(
        query_index=-1, total_traced=total_traced,
        estimated_found=estimated_found, recall=recall,
        centroids_searched=len(selected),
        vectors_covered=sum(c.cnt for c in selected),
    )


def process_trace_file(
    path: Path,
    policy: CentroidPolicy,
    limit: int | None = None,
) -> list[QueryResult]:
    results: list[QueryResult] = []

    open_fn = gzip.open if path.suffix == ".gz" else open
    mode = "rb"

    with open_fn(path, mode) as f:
        for line in f:
            if limit is not None and len(results) >= limit:
                break

            line = line.strip()
            if not line:
                continue

            obj = orjson.loads(line)
            qidx = int(obj["query_index"])
            traces = obj["traces"]
            centroids_raw = obj["centroids"]

            # Total ground-truth vectors = length of traces array
            total_traced = len(traces) if traces else 0
            centroids = [load_centroid(c) for c in centroids_raw]

            result = apply_policy(centroids, policy, total_traced)
            result.query_index = qidx
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _stddev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5


def print_summary(results: list[QueryResult], policy_name: str) -> None:
    recalls = [float(r.recall) for r in results]
    cents = [float(r.centroids_searched) for r in results]
    vecs = [float(r.vectors_covered) for r in results]

    def _stats(label: str, vals: list[float]) -> None:
        n = len(vals)
        mean = sum(vals) / n
        lo, hi = min(vals), max(vals)
        sd = _stddev(vals)
        print(f"  {label:>20s}: mean={mean:11.4f}  min={lo:11.4f}  max={hi:11.4f}  stddev={sd:11.4f}")

    print(f"SPANN Recall Estimation \u2014 Policy: {policy_name}")
    print(f"Queries analyzed: {len(results)}")
    _stats("recall", recalls)
    _stats("centroids searched", cents)
    _stats("vectors covered", vecs)
    print()

    show_n = min(20, len(results))
    # Show queries with lowest recall
    by_recall = sorted(results, key=lambda r: r.recall)
    print(f"Queries with lowest recall (Top-{show_n}):")
    print(f"{'Query':>6s}  {'Traced':>8s}  {'Found':>8s}  {'Recall':>8s}  {'Cents':>6s}  {'Vecs':>8s}")
    print("-" * 56)
    for r in by_recall[:show_n]:
        print(f"{r.query_index:>6d}  {r.total_traced:>8d}  {r.estimated_found:>8d}  {r.recall:>8.6f}  {r.centroids_searched:>6d}  {r.vectors_covered:>8d}")
    if len(results) > show_n:
        print(f"... ({len(results) - show_n} more queries)")
    print(f"Queries with highest recall (Top-{show_n}):")
    print(f"{'Query':>6s}  {'Traced':>8s}  {'Found':>8s}  {'Recall':>8s}  {'Cents':>6s}  {'Vecs':>8s}")
    print("-" * 56)
    for r in by_recall[-show_n:]:
        print(f"{r.query_index:>6d}  {r.total_traced:>8d}  {r.estimated_found:>8d}  {r.recall:>8.6f}  {r.centroids_searched:>6d}  {r.vectors_covered:>8d}")


def print_json(results: list[QueryResult]) -> None:
    output = [
        {
            "query_index": r.query_index,
            "total_traced": r.total_traced,
            "estimated_found": r.estimated_found,
            "recall": round(r.recall, 6),
            "centroids_searched": r.centroids_searched,
            "vectors_covered": r.vectors_covered,
        }
        for r in results
    ]
    print(orjson.dumps(output).decode())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate SPANN recall from a trace file using a centroid selection policy.",
    )
    parser.add_argument(
        "trace_file",
        type=Path,
        help="Path to a SPANN trace NDJSON file (one JSON object per line).",
    )
    parser.add_argument(
        "--policy",
        choices=list(POLICY_MAP.keys()),
        required=True,
        help="Centroid selection policy name.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="For top_n policy: number of closest centroids to select (default: 10).",
    )
    parser.add_argument(
        "--vector-count",
        type=int,
        default=50000,
        help="For vector_count policy: target number of traced vectors to cover (default: 50000).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="For alpha policy",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="For beta policy",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N query entries, then exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output per-query results as JSON instead of a human-readable table.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        policy = make_policy(args.policy, vars(args))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    results = process_trace_file(args.trace_file, policy, limit=args.limit)

    if not results:
        print("No queries found in trace file.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print_json(results)
    else:
        print_summary(results, policy.name)


if __name__ == "__main__":
    main()
