#!/usr/bin/env python3
"""Summarize round diagnostics without removing any observations (stdlib only)."""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(source, output_dir, expected_rounds, levels):
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No completed query rounds in " + str(source))
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (int(row["L"]), int(row["round"]))
        if key in seen:
            raise ValueError("Duplicate L/round: " + str(key))
        seen.add(key)
        for field in ("qps", "wall_seconds", "queries", "mean_us"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid " + field + " in " + str(key))
        groups[key[0]].append(row)
    summary, flagged = [], []
    for level, group in sorted(groups.items()):
        group.sort(key=lambda row: int(row["round"]))
        qps = [float(row["qps"]) for row in group]
        latency = [float(row["mean_us"]) for row in group]
        wall_per_query = [float(row["wall_seconds"]) * 1e6 / int(row["queries"]) for row in group]
        def upper_fence(values):
            q1, q3 = quantile(values, .25), quantile(values, .75)
            return q3 + 1.5 * (q3 - q1)
        wall_fence, latency_fence = upper_fence(wall_per_query), upper_fence(latency)
        for row, wall, mean in zip(group, wall_per_query, latency):
            flagged.append(dict(row, first_round=int(row["round"]) == 1,
                                slow_wall_outlier=wall > wall_fence,
                                slow_query_latency_outlier=mean > latency_fence))
        rec = [float(row["recall_percent"]) for row in group if row["recall_percent"]]
        others = [float(row["qps"]) for row in group if int(row["round"]) != 1]
        summary.append(dict(
            L=level, completed_rounds=len(group), expected_rounds=expected_rounds,
            complete={int(row["round"]) for row in group} == set(range(1, expected_rounds + 1)),
            qps_mean=statistics.mean(qps), qps_median=statistics.median(qps),
            qps_min=min(qps), qps_max=max(qps),
            qps_cv=statistics.stdev(qps) / statistics.mean(qps) if len(qps) > 1 else 0,
            aggregate_qps=sum(int(row["queries"]) for row in group) /
                          sum(float(row["wall_seconds"]) for row in group),
            first_round_qps=qps[0] if int(group[0]["round"]) == 1 else "",
            qps_mean_excluding_first_diagnostic_only=statistics.mean(others) if others else "",
            mean_us=statistics.mean(latency), median_round_mean_us=statistics.median(latency),
            max_round_mean_us=max(latency),
            mean_p999_us=statistics.mean(float(row["p999_us"]) for row in group),
            recall_min=min(rec) if rec else "", recall_max=max(rec) if rec else "",
            mean_ios=statistics.mean(float(row["mean_ios"]) for row in group),
            mean_io_us=statistics.mean(float(row["mean_io_us"]) for row in group),
            mean_cpu_us=statistics.mean(float(row["mean_cpu_us"]) for row in group),
            decode_us_per_query=statistics.mean(float(row["decode_us"]) / int(row["queries"]) for row in group),
            max_current_rss_gib=max(int(row["current_rss_kib"]) for row in group) / 2**20,
            total_minor_faults=sum(int(row["minor_faults"] or 0) for row in group),
            total_major_faults=sum(int(row["major_faults"] or 0) for row in group),
            slow_wall_rounds=sum(wall > wall_fence for wall in wall_per_query),
            slow_query_latency_rounds=sum(mean > latency_fence for mean in latency)))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "rounds_flagged.csv", flagged)
    complete = (set(groups) == set(levels) and all(row["complete"] for row in summary))
    print("COMPLETE" if complete else "PARTIAL", str(source), "rounds=" + str(len(rows)))
    for row in summary:
        print("L={L} rounds={completed_rounds} QPS mean={qps_mean:.2f} median={qps_median:.2f} "
              "aggregate={aggregate_qps:.2f} CV={qps_cv:.3f} max_mean_us={max_round_mean_us:.2f}"
              .format(**row))
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-rounds", type=int, default=10)
    parser.add_argument("--L", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10])
    args = parser.parse_args()
    if args.expected_rounds < 1:
        parser.error("--expected-rounds must be positive")
    raise SystemExit(0 if analyze(args.source, args.output_dir or args.source.parent,
                                 args.expected_rounds, args.L) else 1)
