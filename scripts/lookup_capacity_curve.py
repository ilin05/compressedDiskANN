#!/usr/bin/env python3
"""Query exact lookup byte intervals for prefixes of the saved sorted cache IDs."""
import argparse
import csv
import json
from pathlib import Path


def load_curve(path):
    with Path(path).open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("Empty lookup curve")
    previous = -1
    for row in rows:
        for field in ("first_nodes", "last_nodes", "nhood_buckets", "coord_buckets",
                      "lookup_bytes", "nhood_bucket_bytes", "coord_bucket_bytes"):
            row[field] = int(row[field])
        if row["first_nodes"] != previous + 1 or row["last_nodes"] < row["first_nodes"]:
            raise ValueError("Lookup intervals must be contiguous, starting at zero")
        if row["lookup_bytes"] != (row["nhood_buckets"] * row["nhood_bucket_bytes"] +
                                   row["coord_buckets"] * row["coord_bucket_bytes"]):
            raise ValueError("Bucket bytes do not sum to lookup bytes")
        previous = row["last_nodes"]
    return rows


def lookup_at(rows, nodes):
    for row in rows:
        if row["first_nodes"] <= nodes <= row["last_nodes"]:
            return {key: row[key] for key in ("nhood_buckets", "coord_buckets", "lookup_bytes")}
    raise ValueError("Node count {} outside measured range 0..{}; do not extrapolate".format(
        nodes, rows[-1]["last_nodes"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--nodes", nargs="+", type=int)
    parser.add_argument("--limit-bytes", type=int)
    parser.add_argument("--require-limit-exceeded", action="store_true")
    args = parser.parse_args()
    rows = load_curve(args.curve)
    for nodes in args.nodes or []:
        print(json.dumps(dict(nodes=nodes, **lookup_at(rows, nodes))))
    if args.limit_bytes is not None:
        feasible = [row["last_nodes"] for row in rows if row["lookup_bytes"] <= args.limit_bytes]
        over = [row for row in rows if row["lookup_bytes"] > args.limit_bytes]
        print(json.dumps(dict(limit_bytes=args.limit_bytes,
                              largest_measured_feasible_nodes=max(feasible, default=0),
                              first_exceeding_nodes=over[0]["first_nodes"] if over else None,
                              first_exceeding_bytes=over[0]["lookup_bytes"] if over else None,
                              measured_through=rows[-1]["last_nodes"]), indent=2))
        if args.require_limit_exceeded and not over:
            raise SystemExit("Limit not reached: increase MAX_NODES and regenerate/reprofile its ID list")


if __name__ == "__main__":
    main()
