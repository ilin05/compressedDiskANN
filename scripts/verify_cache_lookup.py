#!/usr/bin/env python3
"""Check BIGANN1M curve predictions against production Raw cache loading."""
import argparse
from pathlib import Path
import re
import struct
import subprocess

from lookup_capacity_curve import load_curve, lookup_at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--index-prefix", required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--limit-bytes", type=int, required=True)
    args = parser.parse_args()
    rows = load_curve(args.curve)
    feasible = max(r["last_nodes"] for r in rows if r["lookup_bytes"] <= args.limit_bytes)
    over = next((r["first_nodes"] for r in rows if r["lookup_bytes"] > args.limit_bytes), None)
    if over is None:
        raise ValueError("Correctness test must include the first over-limit insertion")
    out = args.curve.parent
    query = out / "probe_query.fbin"
    # Search CLI loads this dummy query, but probe-only exits before any query executes.
    query.write_bytes(struct.pack("<II", 1, 128) + bytes(128 * 4))
    for nodes in sorted({min(1000, feasible), feasible, over}):
        if not nodes:
            continue
        log = out / "production_raw_{}.log".format(nodes)
        command = [str(args.binary.resolve()), "--data_type", "float", "--dist_fn", "l2",
                   "--index_path_prefix", args.index_prefix, "--query_file", str(query.resolve()),
                   "--gt_file", "null", "--cache_codec", "raw", "--num_nodes_to_cache", str(nodes),
                   "--cache_node_ids_file", str(args.ids_file.resolve()), "--cache_budget_probe_only",
                   "-T", "1", "-K", "1", "-L", "1", "-W", "1", "-R", "1",
                   "--result_path", str(out / "production_probe")]
        with log.open("w") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        lines = [line for line in log.read_text().splitlines() if "CACHE_BUDGET " in line]
        if not lines:
            raise ValueError("No CACHE_BUDGET in {}".format(log))
        actual = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", lines[-1])}
        predicted = lookup_at(rows, nodes)
        if (actual["nodes"] != nodes or actual["graph_lookup_bytes"] != predicted["lookup_bytes"] or
                actual["nhood_buckets"] != predicted["nhood_buckets"] or
                actual["coord_buckets"] != predicted["coord_buckets"]):
            raise ValueError("Production lookup differs from curve: {}".format(log))
        print("PASS production lookup nodes={} bytes={}".format(nodes, predicted["lookup_bytes"]), flush=True)


if __name__ == "__main__":
    main()
