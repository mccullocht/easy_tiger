#!/usr/bin/env python3
"""Simulate posting-search cutoff rules against an ET_SEARCH_TRACE jsonl trace.

Each trace line is one query:
    {"limit":10, "num_rerank":50, ..., "last_contribution_posting_rank":72,
     "centroids":[[distance, posting vector count, contributing vector count], ...]}

with centroids in distance-from-query order. `contributing` is the number of vectors from that
posting that are in the final rerank candidate pool (results + overflow).

For each rule we compute the simulated search depth (number of postings opened) and report how
many rerank candidates the cutoff drops, relative to the full pool.

Caveats when interpreting results:
- `contributing` is final-pool membership: a vector that transiently entered the pool but was
  later outcompeted counts as 0. A runtime barren-run rule that counts pushes would stop later
  than simulated here, so these savings/recall numbers are pessimistic for barren-run rules.
- Dedup: a posting whose vectors were all already seen reads as contributing=0 even though it was
  scanned. Fine for a stop rule (nothing new contributed), but barren stretches caused by dedup
  look identical to genuinely unproductive stretches.
- The trace only covers the postings the selector actually searched; savings are measured
  against that set, not against the whole index.
"""

import argparse
import json
import sys


def load_queries(path):
    """Load query documents from the trace. The CLI writes summary output to stdout too, so
    lines that aren't JSON query objects are skipped. Returns (queries, skipped line count)."""
    queries = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(q, dict) and "centroids" in q:
                queries.append(q)
            else:
                skipped += 1
    return queries, skipped


def last_contributing_rank(q):
    """1-based rank of the last posting with a contribution; 0 when the pool is empty."""
    if "last_contribution_posting_rank" in q:
        return q["last_contribution_posting_rank"]
    centroids = q["centroids"]
    return max((i + 1 for i, (_, _, c) in enumerate(centroids) if c > 0), default=0)


def floor_depth(centroids, floor_vectors):
    """Depth needed to reach the cumulative posting-vector floor (always the full set if the
    floor is never reached)."""
    cumulative = 0
    for i, (_, v, _) in enumerate(centroids):
        cumulative += v
        if cumulative >= floor_vectors:
            return i + 1
    return len(centroids)


def ratio_depth(centroids, alpha, spread_k=0, spread_beta=0.0):
    """Depth keeping all centroids within `alpha * d_1`. With spread_beta != 0, alpha is scaled
    by the relative spread of the first `spread_k` distances: alpha_eff = alpha * (d_k/d_1)^beta,
    a per-query adaptive ratio (beta=0 disables)."""
    if not centroids:
        return 0
    d1 = centroids[0][0]
    if spread_beta and spread_k > 1 and len(centroids) >= 2:
        dk = centroids[min(spread_k, len(centroids)) - 1][0]
        alpha = alpha * (dk / d1) ** spread_beta
    for i, (d, _, _) in enumerate(centroids):
        if d > alpha * d1:
            return i
    return len(centroids)


def barren_depth(centroids, m):
    """Depth of a barren-run rule: stop m postings past the last contributing posting."""
    last = max((i + 1 for i, (_, _, c) in enumerate(centroids) if c > 0), default=0)
    return min(len(centroids), last + m)


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = min(int(round(p / 100 * (len(sorted_vals) - 1))), len(sorted_vals) - 1)
    return sorted_vals[idx]


def evaluate(queries, rules):
    """Returns per-rule aggregates plus per-query details for dumping."""
    results = {}
    for name, fn in rules.items():
        dropped_ratios = []
        undercovered = 0
        vec_saved_fracs = []
        total_pool = 0
        total_dropped = 0
        per_query = []
        for q in queries:
            centroids = q["centroids"]
            pool = sum(c for _, _, c in centroids)
            depth = fn(q)
            kept = sum(c for _, _, c in centroids[:depth])
            dropped = pool - kept
            vectors = sum(v for _, v, _ in centroids)
            vectors_read = sum(v for _, v, _ in centroids[:depth])
            last = last_contributing_rank(q)
            if pool > 0:
                dropped_ratios.append(dropped / pool)
            if last > depth:
                undercovered += 1
            if vectors > 0:
                vec_saved_fracs.append(1 - vectors_read / vectors)
            total_pool += pool
            total_dropped += dropped
            per_query.append({
                "depth": depth,
                "pool": pool,
                "dropped": dropped,
                "last_contribution_posting_rank": last,
                "undercovered": last > depth,
            })
        ratios = sorted(dropped_ratios)
        saved = sorted(vec_saved_fracs)
        results[name] = {
            "mean_depth": mean_or(per_query, lambda r: r["depth"], 0),
            "dropped_mean": mean_or(per_query, lambda r: r["dropped"] / r["pool"] if r["pool"] else 0.0, 0),
            "dropped_p95": pct(ratios, 95),
            "dropped_max": ratios[-1] if ratios else 0.0,
            "dropped_weighted": total_dropped / total_pool if total_pool else 0.0,
            "undercovered_pct": 100 * undercovered / len(queries) if queries else 0,
            "vectors_saved_mean_pct": 100 * (sum(saved) / len(saved) if saved else 0),
            "per_query": per_query,
        }
    return results


def mean_or(records, key_fn, default):
    vals = [key_fn(r) for r in records]
    return sum(vals) / len(vals) if vals else default


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", help="JSONL trace from ET_SEARCH_TRACE=1 et spann search ...")
    ap.add_argument("--alpha", type=float, default=1.4,
                    help="ratio cutoff: keep centroids with distance <= alpha * d_1 (default 1.4)")
    ap.add_argument("--floor-vectors", type=int, default=None,
                    help="cumulative posting vectors to always search before any cutoff applies "
                         "(default: 4 * num_rerank per query)")
    ap.add_argument("--barren-m", type=int, default=2,
                    help="stop after this many consecutive postings contributing nothing (default 2)")
    ap.add_argument("--topn", type=int, default=50, help="fixed-depth baseline (default 50)")
    ap.add_argument("--spread-k", type=int, default=16,
                    help="number of front centroids used for the adaptive ratio spread")
    ap.add_argument("--spread-beta", type=float, default=0.0,
                    help="adaptive ratio exponent: alpha_eff = alpha * (d_k/d_1)^beta (default 0, off)")
    ap.add_argument("--rules", default="none,topn,floor,ratio,hybrid,barren,combined",
                    help="comma-separated rules to evaluate")
    ap.add_argument("--dump", metavar="PREFIX", default=None,
                    help="write per-query simulation records to PREFIX.jsonl")
    args = ap.parse_args()

    queries, skipped = load_queries(args.trace)
    if skipped:
        print(f"skipped {skipped} non-query lines", file=sys.stderr)
    if not queries:
        sys.exit("no queries in trace")
    queries = [q for q in queries if q["centroids"]]
    n = len(queries)
    total_pool = sum(sum(c for _, _, c in q["centroids"]) for q in queries)
    total_vectors = sum(sum(v for _, v, _ in q["centroids"]) for q in queries)
    print(f"trace: {n} queries, {total_pool} pool vectors total, "
          f"{total_vectors} posting vectors total\n")

    def floor_for(q):
        if args.floor_vectors is not None:
            return args.floor_vectors
        # Auto floor: enough scanned vectors to plausibly fill the rerank budget.
        return 4 * (q.get("num_rerank") or q.get("limit") or 10)

    floor_key = (
        f"floor:{args.floor_vectors}" if args.floor_vectors is not None else "floor:auto")

    all_rules = {
        "none": lambda q: len(q["centroids"]),
        f"topn:{args.topn}": lambda q: min(args.topn, len(q["centroids"])),
        # Pure cumulative-vector floor: search postings in distance order until the scanned
        # vector count reaches the floor. This matches the VectorCount centroid selector.
        floor_key: lambda q: floor_depth(q["centroids"], floor_for(q)),
        f"ratio:{args.alpha:g}": lambda q: ratio_depth(
            q["centroids"], args.alpha, args.spread_k, args.spread_beta),
        "hybrid": lambda q: max(
            floor_depth(q["centroids"], floor_for(q)),
            ratio_depth(q["centroids"], args.alpha, args.spread_k, args.spread_beta)),
        f"barren:m={args.barren_m}": lambda q: max(
            floor_depth(q["centroids"], floor_for(q)),
            barren_depth(q["centroids"], args.barren_m)),
        # Recommended: stop only when the vector floor is satisfied AND the ratio cutoff AND the
        # barren-run cutoff have all triggered. Each signal fails on a different class of query;
        # the max fails only where all of them fail at once.
        "combined": lambda q: max(
            floor_depth(q["centroids"], floor_for(q)),
            ratio_depth(q["centroids"], args.alpha, args.spread_k, args.spread_beta),
            barren_depth(q["centroids"], args.barren_m)),
    }

    selected = {}
    for name in args.rules.split(","):
        name = name.strip()
        # Accept bare aliases ("topn") for parameterized keys ("topn:50").
        matches = [k for k in all_rules if k == name or k.split(":")[0] == name]
        if not matches:
            sys.exit(f"unknown rule '{name}'; available: {', '.join(all_rules)}")
        selected[matches[0]] = all_rules[matches[0]]

    results = evaluate(queries, selected)

    header = (f"{'rule':<16} {'depth':>7} {'drop mean':>10} {'drop p95':>9} "
              f"{'drop max':>9} {'drop wtd':>9} {'undercov':>9} {'vec saved':>10}")
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<16} {r['mean_depth']:>7.1f} {r['dropped_mean']:>10.1%} "
              f"{r['dropped_p95']:>9.1%} {r['dropped_max']:>9.1%} {r['dropped_weighted']:>9.1%} "
              f"{r['undercovered_pct']:>8.1f}% {r['vectors_saved_mean_pct']:>9.1f}%")

    print("\ncolumns: dropped ratio = rerank candidates cut / pool size; undercov = queries where "
          "the cutoff is shallower than the last contributing posting; vec saved = posting "
          "vectors not scanned, vs the full trace.")

    if args.dump:
        with open(f"{args.dump}.jsonl", "w") as f:
            for q, per_rule in zip(queries, zip(*(r["per_query"] for r in results.values()))):
                record = {
                    "last_contribution_posting_rank": last_contributing_rank(q),
                    "rules": {name: pr for name, pr in zip(results, per_rule)},
                }
                f.write(json.dumps(record) + "\n")
        print(f"per-query records written to {args.dump}.jsonl")


if __name__ == "__main__":
    main()
