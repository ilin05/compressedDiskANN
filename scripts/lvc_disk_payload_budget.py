#!/usr/bin/env python3
"""Single-attempt payload/adjacency/record/state budgets; PQ and node lookups excluded."""
import argparse
import json
import math
from pathlib import Path
import struct

from lvc_disk_sampled_budget import (budget_from_log, compare_results, csv_rows,
                                     forest_from_log, run_logged, save_csv,
                                     save_json, verify_samples)

CASES = (("raw", 0), ("alp", 0), ("dexor", 2), ("gorilla", 2), ("elf", 2),
         ("dexor", -1), ("gorilla", -1), ("elf", -1))
FIELDS = ("case", "codec", "chain_max", "status", "error", "exit_code", "within_budget",
          "budget_bytes", "planning_budget_bytes", "estimated_budget_bytes", "actual_budget_bytes",
          "budget_utilization", "planned_nodes", "actual_nodes", "pq_file_bytes", "adjacency_bytes",
          "payload_bytes", "record_metadata_bytes", "state_cache_bytes", "excluded_lookup_bytes",
          "cache_with_lookup_bytes", "nhood_buckets", "coord_buckets", "sample_nodes",
          "sample_payload_ratio_f32", "actual_payload_ratio_f32", "roots", "average_depth", "max_depth",
          "index_load_seconds", "cache_selection_seconds", "cache_data_load_seconds", "cache_load_seconds",
          "forest_build_seconds", "total_load_seconds", "process_wall_seconds", "k", "L", "threads",
          "beamwidth", "rounds", "recall_at_1_percent", "qps", "mean_us", "p999_us", "mean_ios",
          "peak_rss_after_cache_kib", "current_rss_after_cache_kib", "peak_rss_after_search_kib",
          "current_rss_after_search_kib", "result_bits_match_raw", "extra_candidate_filtering")


def estimate(row, nodes):
    count = int(row["nodes"])
    value = {"adjacency_bytes": nodes * int(row["adjacency_bytes_per_node"])}
    for target, source in (("payload_bytes", "payload_bytes"),
                           ("record_metadata_bytes", "record_metadata_bytes"),
                           ("state_cache_bytes", "state_cache_bytes")):
        value[target] = (nodes * int(row[source]) + count - 1) // count
    value["budget_bytes"] = sum(value.values())
    return value


def select_nodes(row, target, maximum):
    low, high = 0, maximum + 1
    while high - low > 1:
        middle = (low + high) // 2
        if estimate(row, middle)["budget_bytes"] <= target:
            low = middle
        else:
            high = middle
    if not low:
        raise ValueError("One node exceeds the planning budget")
    return low


def actual_budget(stats):
    adjacency = stats["graph_bytes"] - stats["graph_lookup_bytes"]
    if adjacency < 0 or stats.get("includes_state") != 1:
        raise ValueError("Invalid cache accounting / state omitted")
    total = adjacency + stats["vector_payload_bytes"] + stats["vector_metadata_bytes"] + stats["state_cache_bytes"]
    if total + stats["graph_lookup_bytes"] != stats["total_bytes"]:
        raise ValueError("Cache components do not sum to CACHE_BUDGET total")
    return total


def write_results(path, rows):
    save_csv(path, [{field: row.get(field, "") for field in FIELDS} for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--index-prefix", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-points", type=int, required=True)
    parser.add_argument("--sample-nodes", type=int, default=1000000)
    parser.add_argument("--budget-mib", type=float, default=20 * 1024)
    parser.add_argument("--planning-budget-mib", type=float, default=19.5 * 1024)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--L", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--beamwidth", type=int, default=2)
    parser.add_argument("--state-cache-ratio", type=float, default=.01)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-cache", action="store_true")
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    nominal = int(args.budget_mib * (1 << 20))
    target = int(args.planning_budget_mib * (1 << 20))
    if (not 0 < target <= nominal or min(args.sample_nodes, args.threads, args.rounds, args.beamwidth) < 1 or
            not args.L or min(args.L) < 1 or len(args.L) != len(set(args.L)) or
            not 0 <= args.state_cache_ratio <= .01 or (args.check_results and args.beamwidth != 1)):
        parser.error("Invalid budget/search parameters; bitwise comparison requires beamwidth=1")
    # Preserve the provided path spelling (the workspace may use case aliases).
    out = args.output_dir.absolute()
    out.mkdir(parents=True, exist_ok=True)
    index = str(args.index_prefix.absolute())
    pq = Path(index + "_pq_compressed.bin")
    with pq.open("rb") as handle:
        points, pq_bytes = struct.unpack("<II", handle.read(8))
    if points != args.expected_points or pq_bytes != 32 or pq.stat().st_size != 8 + points * pq_bytes:
        raise ValueError("Unexpected BIGANN PQ shape/file size")
    maximum = max(1, math.floor(points * .5 + .5))
    if args.sample_nodes > maximum:
        raise ValueError("Sample exceeds the production 50% cache cap")
    common = [str(args.binary.absolute()), "--data_type", "float", "--dist_fn", "l2",
              "--index_path_prefix", index, "--query_file", str(args.query.absolute()),
              "--gt_file", str(args.gt.absolute()), "-T", str(args.threads), "-K", "1",
              "-L", *map(str, args.L), "-W", str(args.beamwidth), "-R", str(args.rounds),
              "--lvc_state_cache_ratio", str(args.state_cache_ratio)]
    identity = {}
    paths = {"binary": args.binary, "disk": Path(index + "_disk.index"), "pq": pq,
             "pivots": Path(index + "_pq_pivots.bin"), "query": args.query, "gt": args.gt}
    for name, path in paths.items():
        stat = path.stat()
        identity[name] = [str(path.absolute()), stat.st_size, stat.st_mtime_ns]
    sample_identity = dict(identity, sample_nodes=args.sample_nodes, state_ratio=args.state_cache_ratio)
    save_json(out / "run_status.json", dict(status="running", budget_definition="non_pq_non_lookup",
              nominal_budget_bytes=nominal, planning_budget_bytes=target, cases=list(CASES)))
    sample_csv, sample_stamp = out / "sample.csv", out / "sample.ok.json"
    if not (args.resume and sample_csv.exists() and sample_stamp.exists() and
            json.loads(sample_stamp.read_text()) == sample_identity):
        temporary = out / "sample.pending.csv"
        command = common + ["--cache_codec", "raw", "--num_nodes_to_cache", str(args.sample_nodes),
                            "--cache_budget_sample_csv", str(temporary), "--result_path", str(out / "sample")]
        if run_logged(command, out / "sample.log"):
            raise RuntimeError("Shared sample failed; see sample.log")
        verify_samples(csv_rows(temporary), args.sample_nodes)
        temporary.replace(sample_csv)
        save_json(sample_stamp, sample_identity)
    samples = verify_samples(csv_rows(sample_csv), args.sample_nodes)
    plans, summary, failed = [], [], []
    raw_prefix = out / "raw"
    raw_reference_ready = False
    for codec, chain in CASES:
        tag = codec if not chain else codec + ("_ch2" if chain == 2 else "_chunbounded")
        row = samples[codec, chain]
        selected = select_nodes(row, target, maximum)
        prediction = estimate(row, selected)
        print("PLAN {} nodes={} counted_bytes={} planning_bytes={} nominal_bytes={} (PQ/node lookups excluded)".format(
            tag, selected, prediction["budget_bytes"], target, nominal), flush=True)
        plans.append(dict(case=tag, codec=codec, chain_max=chain, nodes=selected, **prediction))
        save_csv(out / "node_plans.csv", plans)
        signature = dict(sample_identity, budget_definition="non_pq_non_lookup_v1", codec=codec, chain=chain,
                         nominal=nominal, target=target, L=args.L, threads=args.threads, rounds=args.rounds,
                         beamwidth=args.beamwidth, verify=args.verify_cache, compare=args.check_results)
        stamp, prefix = out / (tag + ".done.json"), out / tag
        destination = out / (tag + "_results.csv")
        if args.resume and stamp.exists() and destination.exists() and json.loads(stamp.read_text()) == signature:
            rows = csv_rows(destination)
            print("REUSE {}: {}".format(tag, destination), flush=True)
        else:
            if stamp.exists():
                stamp.unlink()
            base = dict(case=tag, codec=codec, chain_max=chain, status="failed", error="", exit_code="",
                        budget_bytes=nominal, planning_budget_bytes=target,
                        estimated_budget_bytes=prediction["budget_bytes"], planned_nodes=selected,
                        pq_file_bytes=pq.stat().st_size, sample_nodes=args.sample_nodes,
                        sample_payload_ratio_f32=int(row["nodes"]) * int(row["dim"]) * 4 / int(row["payload_bytes"]),
                        k=1, threads=args.threads, beamwidth=args.beamwidth, rounds=args.rounds,
                        extra_candidate_filtering=0)
            rows = []
            source = Path(str(prefix) + "_search_summary.csv")
            if source.exists():
                source.unlink()  # Never consume a stale summary after a failed subprocess.
            try:
                # Intentionally NO hard-cap argument: over-budget attempts still search and are recorded.
                command = common + ["--cache_codec", codec, "--num_nodes_to_cache", str(selected),
                    "--lvc_chain_max", str(chain or 2), "--cache_budget_include_state", "--result_path", str(prefix)]
                if args.verify_cache and chain:
                    command.append("--verify_cache")
                log = out / (tag + "_attempt0.log")
                code = run_logged(command, log)
                base["exit_code"] = code
                base["process_wall_seconds"] = json.loads(log.with_suffix(".timing.json").read_text())["wall_seconds"]
                actual = budget_from_log(log)
                forest = forest_from_log(log)
                if actual:
                    used = actual_budget(actual)
                    base.update(actual_nodes=actual["nodes"], actual_budget_bytes=used,
                        within_budget=used <= nominal, budget_utilization=used / nominal,
                        adjacency_bytes=actual["graph_bytes"] - actual["graph_lookup_bytes"],
                        payload_bytes=actual["vector_payload_bytes"], record_metadata_bytes=actual["vector_metadata_bytes"],
                        state_cache_bytes=actual["state_cache_bytes"], excluded_lookup_bytes=actual["graph_lookup_bytes"],
                        cache_with_lookup_bytes=actual["total_bytes"], nhood_buckets=actual["nhood_buckets"],
                        coord_buckets=actual["coord_buckets"], roots=forest.get("roots", 0),
                        average_depth=forest.get("average_depth", 0), max_depth=forest.get("max_depth", 0),
                        actual_payload_ratio_f32=actual["nodes"] * int(row["dim"]) * 4 / actual["vector_payload_bytes"])
                if source.exists():
                    for result in csv_rows(source):
                        record = dict(base, L=result["L"], recall_at_1_percent=result.get("Recall@1", ""),
                            qps=result["QPS"], mean_us=result["Mean Latency (mus)"], p999_us=result["99.9 Latency"],
                            mean_ios=result["Mean IOs"], index_load_seconds=result["IndexLoadSeconds"],
                            cache_selection_seconds=result["CacheSelectionSeconds"], cache_load_seconds=result["CacheLoadSeconds"],
                            cache_data_load_seconds=result["CacheDataLoadSeconds"], forest_build_seconds=result["ForestBuildSeconds"],
                            total_load_seconds=result["TotalLoadSeconds"], peak_rss_after_cache_kib=result["RSS_After_Cache_Load(KB)"],
                            current_rss_after_cache_kib=result["Current_RSS_After_Cache_Load(KB)"],
                            peak_rss_after_search_kib=result["RSS_After_Search(KB)"],
                            current_rss_after_search_kib=result["Current_RSS_After_Search(KB)"])
                        rows.append(record)
                if code or not actual or actual["nodes"] != selected or [int(r["L"]) for r in rows] != args.L:
                    raise RuntimeError("Incomplete/failed attempt: {}".format(log))
                if args.check_results and tag != "raw":
                    if not raw_reference_ready:
                        raise RuntimeError("Raw reference attempt failed; cannot validate against stale result files")
                    for level in args.L:
                        for suffix in ("idx_uint32.bin", "dists_float.bin"):
                            compare_results("{}_{}_{}".format(raw_prefix, level, suffix),
                                            "{}_{}_{}".format(prefix, level, suffix))
                    for record in rows:
                        record["result_bits_match_raw"] = True
                for record in rows:
                    record["status"] = "ok" if record["within_budget"] else "over_budget"
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                if not rows:
                    rows = [base]
                for record in rows:
                    record.update(status="failed", error=str(error))
                print("FAILED {}: {} (recorded; continuing)".format(tag, error), flush=True)
            write_results(destination, rows)
            # A saved failure is also an observed attempt, not an automatic retry on resume.
            save_json(stamp, signature)
        summary.extend(rows)
        write_results(out / "summary.csv", summary)
        if tag == "raw":
            raw_reference_ready = all(r["status"] != "failed" for r in rows)
        if any(r["status"] == "failed" for r in rows):
            failed.append(tag)
        print("RECORDED {}: {} status={}".format(tag, destination, rows[0]["status"]), flush=True)
    save_json(out / "run_status.json", dict(status="complete" if not failed else "complete_with_failures",
              failed_cases=failed, nominal_budget_bytes=nominal, planning_budget_bytes=target,
              budget_definition="non_pq_non_lookup", attempts_per_case=1))
    print("Results:", out / "summary.csv", flush=True)
    if failed:
        raise SystemExit("Failed cases were recorded: " + ", ".join(failed))


if __name__ == "__main__":
    main()
