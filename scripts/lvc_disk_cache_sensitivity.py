#!/usr/bin/env python3
"""Fixed-node BIGANN differential-cache sensitivity; one load per codec/count."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
from datetime import datetime, timezone

from lvc_disk_payload_budget import FIELDS as BUDGET_FIELDS, actual_budget, phase_timings
from lvc_disk_sampled_budget import budget_from_log, csv_rows, forest_from_log, run_logged, save_csv, save_json

DEFAULT_CASES = ["dexor:10000000", "gorilla:10000000", "elf:10000000",
                 "dexor:25000000", "gorilla:25000000", "elf:25000000",
                 "gorilla:50000000", "dexor:100000000", "elf:100000000"]
FIELDS = ("case", "codec", "chain_max", "status", "error", "exit_code", "requested_nodes",
          "actual_nodes", "cache_node_hash", "forest_hash", "pq_file_bytes", "adjacency_bytes",
          "payload_bytes", "record_metadata_bytes", "state_cache_bytes", "payload_budget_bytes",
          "lookup_bytes", "cache_with_lookup_bytes", "nhood_buckets", "coord_buckets",
          "actual_payload_ratio_f32", "roots", "average_depth", "max_depth", "k", "L", "threads",
          "beamwidth", "rounds", "recall_at_1_percent", "qps", "mean_us", "p999_us", "mean_ios",
          "peak_rss_after_cache_kib", "current_rss_after_cache_kib", "peak_rss_after_search_kib",
          "current_rss_after_search_kib", "current_rss_after_search_gib", "state_cache_ratio",
          "extra_candidate_filtering", "build_threads", "fallback_root_candidates",
          "propagation_rounds", "fallback_nodes") + tuple(f for f in BUDGET_FIELDS if f.endswith("_seconds"))


def write_results(path, rows, fields=FIELDS):
    save_csv(path, [{f: r.get(f, "") for f in fields} for r in rows])


def parse_case(value):
    try:
        codec, nodes = value.split(":")
        nodes = int(nodes)
        if codec not in ("dexor", "gorilla", "elf") or nodes < 1:
            raise ValueError()
        return codec, nodes
    except ValueError:
        raise argparse.ArgumentTypeError("Expected dexor:N, gorilla:N or elf:N with N>0")


def command_for(args, codec, nodes, prefix):
    command = [str(args.binary.absolute()), "--data_type", "float", "--dist_fn", "l2",
               "--index_path_prefix", str(args.index_prefix.absolute()),
               "--query_file", str(args.query.absolute()), "--gt_file", str(args.gt.absolute()),
               "--cache_codec", codec, "--num_nodes_to_cache", str(nodes), "--lvc_chain_max", "2",
               "--lvc_build_threads", str(args.build_threads), "--lvc_fallback_root_candidates", "1",
               "--lvc_state_cache_ratio", str(args.state_cache_ratio), "--cache_budget_include_state",
               "-K", "1", "-L", *map(str, args.L), "-T", "1", "-W", str(args.beamwidth),
               "-R", str(args.rounds), "--result_path", str(prefix)]
    if args.verify_cache:
        command.append("--verify_cache")
    # No budget probe, hard memory limit, extra candidate filtering, or K=10 run.
    return command


def input_identity(args):
    index = str(args.index_prefix.absolute())
    pq = Path(index + "_pq_compressed.bin")
    with pq.open("rb") as handle:
        points, width = struct.unpack("<II", handle.read(8))
    if points != args.expected_points or width != 32 or pq.stat().st_size != 8 + points * width:
        raise ValueError("Expected BIGANN PQ file with expected-points rows and 32 bytes/vector")
    with args.query.open("rb") as handle:
        nq, dim = struct.unpack("<II", handle.read(8))
    with args.gt.open("rb") as handle:
        ng, kg = struct.unpack("<II", handle.read(8))
    if (not nq or dim != 128 or args.query.stat().st_size != 8 + nq * dim * 4 or
            ng != nq or not kg or args.gt.stat().st_size not in (8 + ng * kg * 4, 8 + ng * kg * 8)):
        raise ValueError("Invalid BIGANN query F32/ground-truth bin shape or size")
    paths = {"runner": Path(__file__), "budget_parser": Path(__file__).with_name("lvc_disk_payload_budget.py"),
             "log_parser": Path(__file__).with_name("lvc_disk_sampled_budget.py"),
             "binary": args.binary, "disk": Path(index + "_disk.index"), "pq": pq,
             "pivots": Path(index + "_pq_pivots.bin"), "query": args.query, "gt": args.gt}
    for suffix in ("_disk.index_medoids.bin", "_disk.index_centroids.bin"):
        path = Path(index + suffix)
        if path.exists():
            paths[suffix] = path
    identity = {}
    for name, path in paths.items():
        stat = path.stat()
        identity[name] = [str(path.absolute()), stat.st_size, stat.st_mtime_ns]
    return identity, pq.stat().st_size, dim


def parse_attempt(args, tag, codec, nodes, prefix, log, code, pq_bytes, dim):
    base = dict(case=tag, codec=codec, chain_max=2, status="failed", error="", exit_code=code,
                requested_nodes=nodes, pq_file_bytes=pq_bytes, k=1, threads=1, beamwidth=args.beamwidth,
                rounds=args.rounds, state_cache_ratio=args.state_cache_ratio, extra_candidate_filtering=0,
                build_threads=args.build_threads, fallback_root_candidates=1)
    rows = []
    try:
        base["process_wall_seconds"] = json.loads(log.with_suffix(".timing.json").read_text())["wall_seconds"]
        stats = budget_from_log(log)
        if not stats:
            raise ValueError("Missing CACHE_BUDGET; inspect " + str(log))
        forest = forest_from_log(log)
        text = log.read_text(errors="replace")
        hashes = re.findall(r"Cache node list: count=(\d+) hash=(\d+)", text)
        if len(hashes) != 1 or int(hashes[0][0]) != nodes or stats["nodes"] != nodes:
            raise ValueError("Actual cache population differs from the requested node count")
        if (int(forest["chain_max"]) != 2 or int(forest["max_depth"]) != 1 or
                int(forest["roots"]) != max(1, math.floor(nodes * .01))):
            raise ValueError("Unexpected forest roots/depth (require degree roots 1%, chain=2)")
        base.update(actual_nodes=nodes, cache_node_hash=hashes[0][1], forest_hash=forest["forest_hash"],
                    adjacency_bytes=stats["graph_bytes"] - stats["graph_lookup_bytes"],
                    payload_bytes=stats["vector_payload_bytes"], record_metadata_bytes=stats["vector_metadata_bytes"],
                    state_cache_bytes=stats["state_cache_bytes"], payload_budget_bytes=actual_budget(stats),
                    lookup_bytes=stats["graph_lookup_bytes"], cache_with_lookup_bytes=stats["total_bytes"],
                    nhood_buckets=stats["nhood_buckets"], coord_buckets=stats["coord_buckets"],
                    actual_payload_ratio_f32=nodes * dim * 4 / stats["vector_payload_bytes"],
                    roots=forest["roots"], average_depth=forest["average_depth"], max_depth=forest["max_depth"])
        base.update(phase_timings(log, 2, args.build_threads, 1, required=True))
        source = Path(str(prefix) + "_search_summary.csv")
        mapping = {"qps": "QPS", "mean_us": "Mean Latency (mus)", "p999_us": "99.9 Latency",
                   "mean_ios": "Mean IOs", "recall_at_1_percent": "Recall@1",
                   "peak_rss_after_cache_kib": "RSS_After_Cache_Load(KB)",
                   "current_rss_after_cache_kib": "Current_RSS_After_Cache_Load(KB)",
                   "peak_rss_after_search_kib": "RSS_After_Search(KB)",
                   "current_rss_after_search_kib": "Current_RSS_After_Search(KB)",
                   "index_load_seconds": "IndexLoadSeconds", "total_load_seconds": "TotalLoadSeconds",
                   "cache_selection_seconds": "CacheSelectionSeconds", "cache_load_seconds": "CacheLoadSeconds",
                   "cache_data_load_seconds": "CacheDataLoadSeconds", "forest_build_seconds": "ForestBuildSeconds"}
        for result in csv_rows(source):
            record = dict(base, L=int(result["L"]), **{k: result[v] for k, v in mapping.items()})
            rows.append(record)
            if (int(result["ActualCachedNodes"]) != nodes or int(result["ChainMax"]) != 2 or
                    int(result["Beamwidth"]) != args.beamwidth):
                raise ValueError("Search summary configuration mismatch")
            if (not 0 <= float(record["recall_at_1_percent"]) <= 100 or
                    any(not math.isfinite(float(record[k])) or float(record[k]) <= 0
                        for k in ("qps", "mean_us", "current_rss_after_search_kib"))):
                raise ValueError("Missing/invalid recall, latency, QPS or current RSS")
            record["current_rss_after_search_gib"] = float(record["current_rss_after_search_kib"]) / (1 << 20)
            record["load_without_verification_seconds"] = (
                float(record["total_load_seconds"]) - float(record.get("verify_seconds", 0)))
        if code or [r["L"] for r in rows] != args.L:
            raise ValueError("Failed process or incomplete L sweep; inspect " + str(log))
        for row in rows:
            row["status"] = "ok"
    except (OSError, ValueError, KeyError, TypeError) as error:
        rows = rows or [base]
        for row in rows:
            row.update(status="failed", error=str(error))
    return rows


def validate_shared_forest(rows, completed):
    first = rows[0]
    if first["status"] != "ok":
        return
    for previous in completed.values():
        other = previous[0]
        if other["status"] == "ok" and int(other["requested_nodes"]) == int(first["requested_nodes"]):
            keys = ("cache_node_hash", "forest_hash", "roots", "average_depth", "max_depth")
            if any(str(first[k]) != str(other[k]) for k in keys):
                for row in rows:
                    row.update(status="failed", error="Shared cache/forest mismatch with " + other["case"])
                return


def publish(out, cases, completed):
    groups = [completed[tag] for tag in cases if tag in completed]
    write_results(out / "summary.csv", [r for group in groups for r in group])
    # Load/wall times are per process and repeat across L in summary.csv. Count them once here.
    case_fields = tuple(f for f in FIELDS if f not in
                        ("L", "recall_at_1_percent", "qps", "mean_us", "p999_us", "mean_ios"))
    write_results(out / "case_summary.csv", [group[0] for group in groups], case_fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "index-prefix", "query", "gt", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-points", type=int, default=1000000000)
    parser.add_argument("--cases", type=parse_case, nargs="+", default=[parse_case(v) for v in DEFAULT_CASES])
    parser.add_argument("--L", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10])
    parser.add_argument("--build-threads", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--beamwidth", type=int, default=2)
    parser.add_argument("--state-cache-ratio", type=float, default=.01)
    parser.add_argument("--verify-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (min(args.expected_points, args.build_threads, args.rounds, args.beamwidth, *args.L) < 1 or
            len(set(args.L)) != len(args.L) or len(set(args.cases)) != len(args.cases) or
            not 0 <= args.state_cache_ratio <= .01 or
            any(n < 2 or n > math.floor(args.expected_points * .5 + .5) for _, n in args.cases)):
        parser.error("Require unique cases/L, positive parameters, 2<=nodes<=50% points, 0<=state ratio<=1%")
    out = args.output_dir.absolute()
    cases = {"{}_ch2_C{}".format(c, n): (c, n) for c, n in args.cases}
    commands = {tag: command_for(args, c, n, out / tag) for tag, (c, n) in cases.items()}
    if args.dry_run:
        for tag, command in commands.items():
            print(tag + ": " + shlex.join(command))
        return
    identity, pq_bytes, dim = input_identity(args)
    os.environ.update(OMP_NUM_THREADS=str(args.build_threads), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another sensitivity run is using " + str(out))
        save_json(out / "config.json", dict(cases=list(cases), commands=commands, chain_max=2,
                  root_policy="degree", root_ratio=.01, k=1, L=args.L, build_threads=args.build_threads,
                  search_threads=1, rounds=args.rounds, beamwidth=args.beamwidth, state_cache_ratio=args.state_cache_ratio,
                  extra_candidate_filtering=0, memory_limit_bytes=None, identity=identity,
                  memory_definition="Current RSS after search includes PQ, lookups and state; payload_budget_bytes excludes PQ/lookups"))
        completed = {}
        for tag, (codec, nodes) in cases.items():
            destination, stamp = out / (tag + "_results.csv"), out / (tag + ".done.json")
            signature = dict(identity=identity, command=commands[tag])
            reused = False
            if args.resume and stamp.exists() and destination.exists():
                try:
                    saved = json.loads(stamp.read_text())
                    if saved.get("signature") == signature:
                        rows = csv_rows(destination)
                        reused = ([int(r["L"]) for r in rows] == args.L and
                                  all(r["status"] == "ok" for r in rows))
                except (OSError, ValueError, KeyError):
                    pass
            save_json(out / "run_status.json", dict(status="running", case=tag,
                      updated_utc=datetime.now(timezone.utc).isoformat(), completed_cases=list(completed)))
            if reused:
                print("REUSE " + tag, flush=True)
            else:
                if stamp.exists():
                    stamp.unlink()
                source = Path(str(out / tag) + "_search_summary.csv")
                if source.exists():
                    source.unlink()  # A failed process must not consume a previous successful summary.
                log = out / (tag + ".log")
                code = run_logged(commands[tag], log)
                rows = parse_attempt(args, tag, codec, nodes, out / tag, log, code, pq_bytes, dim)
            validate_shared_forest(rows, completed)
            write_results(destination, rows)
            if all(r["status"] == "ok" for r in rows):
                save_json(stamp, dict(signature=signature))
            elif stamp.exists():
                stamp.unlink()
            completed[tag] = rows
            publish(out, cases, completed)
            print("RECORDED {} status={} -> {}".format(tag, rows[0]["status"], destination), flush=True)
        failures = [tag for tag, rows in completed.items() if any(r["status"] != "ok" for r in rows)]
        save_json(out / "run_status.json", dict(status="complete_with_failures" if failures else "complete",
                  failed_cases=failures, completed_cases=list(completed),
                  updated_utc=datetime.now(timezone.utc).isoformat()))
        print("Results: " + str(out / "summary.csv"), flush=True)
        if failures:
            raise SystemExit("Failed cases saved; --resume retries them: " + ", ".join(failures))


if __name__ == "__main__":
    main()
