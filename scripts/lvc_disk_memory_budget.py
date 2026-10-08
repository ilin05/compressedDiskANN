#!/usr/bin/env python3
"""Select a disk cache under a graph + vector byte budget.

The budget includes adjacency storage, estimated node lookup buckets, resident
vector payload and differential record metadata. PQ, decoding state cache,
queries, search scratch and temporary construction buffers are excluded.
The search log and summary preserve RSS so physical memory can be audited.
The selector brackets and bisects the byte threshold, then probes the adjacent
node count. This treats cache bytes as monotone in node count near the boundary.
"""

import argparse
import csv
import pathlib
import re
import shutil
import struct
import subprocess


BUDGET_RE = re.compile(
    r"CACHE_BUDGET nodes=(\d+) graph_bytes=(\d+) graph_lookup_bytes=(\d+) vector_payload_bytes=(\d+) "
    r"vector_metadata_bytes=(\d+) total_bytes=(\d+) limit_bytes=(\d+)"
)
CODECS = ("raw", "alp", "dexor", "gorilla", "elf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--index-prefix", type=pathlib.Path, required=True)
    parser.add_argument("--query", type=pathlib.Path, required=True)
    parser.add_argument("--gt", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--budget-mib", type=float, default=32)
    parser.add_argument("--pilot-nodes", type=int, default=8192)
    parser.add_argument("--max-probes", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--L", dest="L_values", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6, 8, 10])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--performance-beamwidth", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget = int(args.budget_mib * 1024 * 1024)
    if budget <= 0 or args.pilot_nodes <= 0 or args.max_probes <= 0 or args.rounds <= 0:
        parser.error("budget, pilot-nodes, max-probes, and rounds must be positive")
    if not args.L_values or any(value < args.k for value in args.L_values):
        parser.error("every L must be at least K")
    pq_path = pathlib.Path(str(args.index_prefix) + "_pq_compressed.bin")
    pq_file_bytes = pq_path.stat().st_size
    with pq_path.open("rb") as f:
        points, pq_bytes = struct.unpack("<II", f.read(8))
    max_nodes = max(1, round(points * 0.5))  # cache_bfs_levels has the same cap
    print(f"points={points} pq_bytes_per_vector={pq_bytes} pq_file_bytes={pq_file_bytes} "
          f"non_pq_budget_bytes={budget}", flush=True)

    def run(codec, nodes, phase, beamwidth=1, probe=False):
        prefix = args.output_dir / f"{codec}_{phase}"
        command = [str(args.binary), "--data_type", "float", "--dist_fn", "l2",
                   "--index_path_prefix", str(args.index_prefix), "--query_file", str(args.query),
                   "--gt_file", str(args.gt), "-K", str(args.k), "-L",
                   *(str(value) for value in args.L_values),
                   "-W", str(beamwidth), "-T", str(args.threads), "-R", str(args.rounds),
                   "--num_nodes_to_cache", str(nodes), "--cache_codec", codec,
                   "--lvc_state_cache_ratio", "0.01", "--cache_budget_bytes", str(budget),
                   "--result_path", str(prefix)]
        if probe:
            command.append("--cache_budget_probe_only")
        elif codec in ("dexor", "gorilla", "elf") and phase == "check":
            command.append("--verify_cache")
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        log = args.output_dir / f"{codec}_{phase}.log"
        log.write_text(completed.stdout)
        match = BUDGET_RE.search(completed.stdout)
        if completed.returncode or match is None:
            raise RuntimeError(f"{codec} {phase} failed; see {log}")
        graph, lookup, payload, metadata, total = map(int, match.groups()[1:6])
        if graph + payload + metadata != total or int(match.group(1)) != nodes:
            raise RuntimeError(f"Inconsistent budget accounting in {log}")
        stats = dict(nodes=nodes, graph_bytes=graph, graph_lookup_bytes=lookup,
                     vector_payload_bytes=payload,
                     vector_metadata_bytes=metadata, total_bytes=total)
        if not probe:
            result_csv = pathlib.Path(f"{args.index_prefix}_K{args.k}_T{args.threads}_C{nodes}_search_result.csv")
            shutil.copyfile(result_csv, args.output_dir / f"{codec}_{phase}.csv")
            with result_csv.open(newline="") as f:
                result = next(csv.DictReader(f))
            for key, column in (("peak_rss_after_cache_kib", "RSS_After_Cache_Load(KB)"),
                                ("current_rss_after_cache_kib", "Current_RSS_After_Cache_Load(KB)"),
                                ("peak_rss_after_search_kib", "RSS_After_Search(KB)"),
                                ("current_rss_after_search_kib", "Current_RSS_After_Search(KB)")):
                stats[key] = result.get(column, "")
            state = re.search(r"state_cache_payload_bytes=(\d+)", completed.stdout)
            stats["state_cache_payload_bytes"] = int(state.group(1)) if state else 0
        return stats

    selected = {}
    for codec in CODECS:
        probes = {}

        def measure(nodes):
            if nodes not in probes:
                if len(probes) >= args.max_probes:
                    raise RuntimeError(f"{codec}: max-probes exhausted before adjacent budget boundary")
                stats = run(codec, nodes, f"probe_{len(probes)}", probe=True)
                probes[nodes] = stats
                print(f"{codec} probe={len(probes) - 1} nodes={nodes} "
                      f"bytes={stats['total_bytes']} utilization={stats['total_bytes'] / budget:.3f}",
                      flush=True)
            return probes[nodes]

        pilot = min(args.pilot_nodes, max_nodes)
        first = measure(pilot)
        if first["total_bytes"] <= budget:
            low, low_stats = pilot, first
            high = None
            candidate = min(max_nodes, max(pilot + 1,
                int(pilot * budget / max(1, first["total_bytes"]) * 1.05)))
            while candidate > low:
                current = measure(candidate)
                if current["total_bytes"] > budget:
                    high = candidate
                    break
                low, low_stats = candidate, current
                candidate = min(max_nodes, max(candidate + 1, candidate * 2))
            if high is None and low != max_nodes:
                raise RuntimeError(f"{codec}: failed to bracket the budget boundary")
        else:
            high = pilot
            candidate = max(1, min(pilot - 1,
                int(pilot * budget / first["total_bytes"] * 0.95)))
            while True:
                current = measure(candidate)
                if current["total_bytes"] <= budget:
                    low, low_stats = candidate, current
                    break
                if candidate == 1:
                    raise RuntimeError(f"{codec}: even one cache node exceeds the budget")
                high = candidate
                candidate = max(1, min(candidate - 1,
                    int(candidate * budget / current["total_bytes"] * 0.95)))

        while high is not None and high - low > 1:
            mid = low + (high - low) // 2
            current = measure(mid)
            if current["total_bytes"] <= budget:
                low, low_stats = mid, current
            else:
                high = mid
        selected[codec] = low_stats
        boundary = "index cap" if high is None else f"node {high} exceeds budget"
        print(f"SELECT {codec}: nodes={low} budget_bytes={low_stats['total_bytes']} "
              f"utilization={low_stats['total_bytes'] / budget:.3f}; {boundary}", flush=True)

    rows = []
    for codec, stats in selected.items():
        for phase, width in (("check", 1), ("performance", args.performance_beamwidth)):
            measured = run(codec, stats["nodes"], phase, beamwidth=width)
            if any(measured[key] != value for key, value in stats.items()) or measured["total_bytes"] > budget:
                raise RuntimeError(f"{codec} {phase}: cache bytes changed or budget exceeded")
            with (args.output_dir / f"{codec}_{phase}.csv").open(newline="") as f:
                results = list(csv.DictReader(f))
            if [int(result["L"]) for result in results] != args.L_values:
                raise RuntimeError(f"{codec} {phase}: missing or reordered L results")
            for result in results:
                recall_key = f"Recall@{args.k}"
                rows.append(dict(codec=codec, phase=phase, k=args.k, L=int(result["L"]),
                                 beamwidth=width, rounds=args.rounds, budget_bytes=budget,
                                 pq_file_bytes=pq_file_bytes,
                                 budget_utilization=measured["total_bytes"] / budget,
                                 **{key: value for key, value in measured.items()
                                    if not key.endswith("_rss_after_cache_kib") and
                                    not key.endswith("_rss_after_search_kib")},
                                 peak_rss_after_cache_kib=result.get("RSS_After_Cache_Load(KB)", ""),
                                 current_rss_after_cache_kib=result.get("Current_RSS_After_Cache_Load(KB)", ""),
                                 peak_rss_after_search_kib=result.get("RSS_After_Search(KB)", ""),
                                 current_rss_after_search_kib=result.get("Current_RSS_After_Search(KB)", ""),
                                 recall=result.get(recall_key, ""), qps=result["QPS"],
                                 mean_us=result["Mean Latency (mus)"],
                                 p999_us=result["99.9 Latency"], mean_ios=result["Mean IOs"]))

    # Differential codecs are lossless. With beamwidth=1, cached nodes may
    # differ but returned IDs and full-precision L2 bits must match raw.
    for codec in ("dexor", "gorilla", "elf"):
        for L in args.L_values:
            for suffix in ("idx_uint32.bin", "dists_float.bin"):
                raw = (args.output_dir / f"raw_check_{L}_{suffix}").read_bytes()
                actual = (args.output_dir / f"{codec}_check_{L}_{suffix}").read_bytes()
                if actual != raw:
                    offset = next((i for i, pair in enumerate(zip(actual, raw)) if pair[0] != pair[1]),
                                  min(len(actual), len(raw)))
                    raise RuntimeError(f"{codec} L={L} {suffix} differs from raw at byte {offset}")
        print(f"PASS {codec}: exact raw IDs and L2 bits at beamwidth=1 for all L", flush=True)

    summary = args.output_dir / "summary.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {summary}", flush=True)


if __name__ == "__main__":
    main()
