#!/usr/bin/env python3
"""One shared cache sample, arithmetic capacity selection, then one full load/search per case."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import time

DIFFERENTIAL = ("dexor", "gorilla", "elf")
CODECS = ("raw", "alp") + DIFFERENTIAL


def csv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_json(path, value):
    temporary = Path(path).with_suffix(".pending.json")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def buckets(nodes, load):
    if not 0 < load <= 1:
        raise ValueError("Invalid lookup load factor")
    count = 1
    while int(count * load) < nodes:
        count *= 2
    return count


def estimate(row, nodes):
    n = int(row["nodes"])
    # ts::robin_map may grow early when probes become long, not only at max_load.
    # Use observed sample occupancy as a conservative additional constraint.
    nl = min(float(row["nhood_max_load"]), n / int(row["nhood_sample_buckets"]))
    cl = min(float(row["coord_max_load"]), n / int(row["coord_sample_buckets"]))
    lookup = (buckets(nodes, nl) * int(row["nhood_bucket_bytes"]) +
              buckets(nodes, cl) * int(row["coord_bucket_bytes"]))
    def extrapolate(column):
        return (nodes * int(row[column]) + n - 1) // n
    result = dict(nodes=nodes, adjacency_bytes=nodes * int(row["adjacency_bytes_per_node"]),
                  lookup_bytes=lookup, payload_bytes=extrapolate("payload_bytes"),
                  record_metadata_bytes=extrapolate("record_metadata_bytes"),
                  state_cache_bytes=extrapolate("state_cache_bytes"))
    result["total_bytes"] = sum(value for key, value in result.items() if key != "nodes")
    return result


def select_nodes(row, target, maximum):
    low, high = 0, maximum + 1
    while high - low > 1:
        mid = (low + high) // 2
        if estimate(row, mid)["total_bytes"] <= target:
            low = mid
        else:
            high = mid
    if not low:
        raise RuntimeError("Even one cache node exceeds the planning budget")
    return low


def budget_from_log(path):
    for line in reversed(Path(path).read_text(errors="replace").splitlines()):
        if "CACHE_BUDGET " in line:
            return {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", line)}
    return None


def forest_from_log(path):
    for line in reversed(Path(path).read_text(errors="replace").splitlines()):
        if "CACHE_FOREST " in line:
            return dict(re.findall(r"(\w+)=([^\s]+)", line))
    return {}


def verify_samples(rows, expected):
    index = {(r["codec"], int(r["chain_max"])): r for r in rows}
    keys = {(c, 0) for c in ("raw", "alp")} | {(c, ch) for c in DIFFERENTIAL for ch in (2, -1)}
    if len(rows) != len(keys) or set(index) != keys:
        raise RuntimeError("Incomplete or duplicate sample CSV")
    for row in rows:
        if int(row["nodes"]) != expected or int(row["payload_bytes"]) <= 0:
            raise RuntimeError("Invalid sample count/payload")
    for chain in (2, -1):
        forests = {(index[c, chain]["forest_hash"], index[c, chain]["roots"], index[c, chain]["average_depth"])
                   for c in DIFFERENTIAL}
        if len(forests) != 1:
            raise RuntimeError("Differential codecs did not share the sample forest")
    return index


def compare_results(raw, actual):
    with Path(raw).open("rb") as a, Path(actual).open("rb") as b:
        ah, bh = a.read(8), b.read(8)
        if ah != bh or len(ah) != 8:
            raise RuntimeError("Search result header mismatch")
        n, k = struct.unpack("<II", ah)
        if Path(raw).stat().st_size != 8 + n * k * 4 or Path(actual).stat().st_size != 8 + n * k * 4:
            raise RuntimeError("Truncated search output")
        position = 0
        while True:
            x, y = a.read(1 << 20), b.read(1 << 20)
            if x != y:
                for offset in range(0, len(x), 4):
                    if x[offset:offset+4] != y[offset:offset+4]:
                        ordinal = (position + offset) // 4
                        raise RuntimeError(f"Result differs query={ordinal // k} rank={ordinal % k} "
                                           f"raw_bits={x[offset:offset+4].hex()} actual_bits={y[offset:offset+4].hex()}")
            if not x:
                return
            position += len(x)


def run_logged(command, log):
    print(f"START {log.name}: {' '.join(command)}", flush=True)
    start = time.monotonic()
    with log.open("w", buffering=1) as handle:
        code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
    wall = time.monotonic() - start
    save_json(log.with_suffix(".timing.json"), dict(command=command,returncode=code,wall_seconds=wall))
    print(f"END {log.name}: code={code} wall_s={wall:.1f}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--index-prefix", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-nodes", type=int, default=1000000)
    parser.add_argument("--budget-mib", type=float, default=32768)
    parser.add_argument("--planning-budget-mib", type=float, default=31744)
    parser.add_argument("--expected-points", type=int)
    parser.add_argument("--expected-pq-bytes", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--L", type=int, nargs="+", default=[1,2,3,4,5,6,8,10])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--beamwidth", type=int, default=2)
    parser.add_argument("--state-cache-ratio", type=float, default=.01)
    parser.add_argument("--codecs", nargs="+", choices=CODECS, default=list(CODECS))
    parser.add_argument("--chains", nargs="+", type=int, choices=[2,-1], default=[2,-1])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-cache", action="store_true")
    parser.add_argument("--check-results", action="store_true", help="Compare result bits with raw; requires beamwidth=1")
    parser.add_argument("--max-retries", type=int, choices=[0,1], default=1)
    args = parser.parse_args()
    hard, target = int(args.budget_mib * (1 << 20)), int(args.planning_budget_mib * (1 << 20))
    if not 0 < target <= hard or min(args.sample_nodes,args.threads,args.k,args.rounds,args.beamwidth) < 1:
        parser.error("Invalid budgets or search parameters")
    if not 0 <= args.state_cache_ratio <= .01 or not args.L or min(args.L) < args.k:
        parser.error("Invalid state ratio or L")
    if len(set(args.codecs)) != len(args.codecs) or len(set(args.chains)) != len(args.chains):
        parser.error("Duplicate codecs/chains")
    if args.check_results and (args.beamwidth != 1 or "raw" not in args.codecs):
        parser.error("check-results requires raw and beamwidth=1")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir/"run_status.json", dict(status="running",sample_nodes=args.sample_nodes,
              hard_budget_bytes=hard,planning_budget_bytes=target,check_results=args.check_results))
    index = str(args.index_prefix.resolve())
    pq = Path(index + "_pq_compressed.bin")
    with pq.open("rb") as handle:
        points, pq_bytes = struct.unpack("<II", handle.read(8))
    if pq.stat().st_size != 8 + points * pq_bytes or (args.expected_points and points != args.expected_points):
        parser.error("Unexpected PQ size or point count")
    if args.expected_pq_bytes is not None and pq_bytes != args.expected_pq_bytes:
        parser.error("Unexpected PQ bytes per vector")
    maximum = max(1, math.floor(points * .5 + .5))
    if args.sample_nodes > maximum:
        parser.error("sample-nodes exceeds the shared 50% cache cap")
    common = [str(args.binary.resolve()), "--data_type", "float", "--dist_fn", "l2", "--index_path_prefix", index,
              "--query_file", str(args.query.resolve()), "--gt_file", str(args.gt.resolve()),
              "-T", str(args.threads), "-K", str(args.k), "-L", *map(str,args.L),
              "-W", str(args.beamwidth), "-R", str(args.rounds),
              "--lvc_state_cache_ratio", str(args.state_cache_ratio)]
    identity = {"binary": [str(args.binary.resolve()), args.binary.stat().st_mtime_ns]}
    for name, path in (("disk", Path(index+"_disk.index")), ("pq", pq), ("pivots", Path(index+"_pq_pivots.bin")),
                       ("query", args.query.resolve()), ("gt", args.gt.resolve())):
        stat = path.stat()
        identity[name] = [str(path), stat.st_size, stat.st_mtime_ns]
    sample_csv, sample_stamp = args.output_dir/"sample.csv", args.output_dir/"sample.ok.json"
    sample_identity = dict(identity, sample_nodes=args.sample_nodes, state_ratio=args.state_cache_ratio)
    if not (args.resume and sample_csv.exists() and sample_stamp.exists() and
            json.loads(sample_stamp.read_text()) == sample_identity):
        temporary = args.output_dir/"sample.pending.csv"
        command = common + ["--cache_codec", "raw", "--num_nodes_to_cache", str(args.sample_nodes),
                            "--cache_budget_sample_csv", str(temporary), "--result_path", str(args.output_dir/"sample")]
        if run_logged(command, args.output_dir/"sample.log"):
            raise RuntimeError("Sample profiling failed; see sample.log")
        verify_samples(csv_rows(temporary), args.sample_nodes)
        temporary.replace(sample_csv)
        save_json(sample_stamp, sample_identity)
    else:
        print("REUSE shared sample", flush=True)
    samples = verify_samples(csv_rows(sample_csv), args.sample_nodes)
    plans, summary, successful = [], [], {}
    cases = [(c,ch) for c in args.codecs for ch in (args.chains if c in DIFFERENTIAL else [0])]
    for codec, chain in cases:
        tag = codec + ("_chunbounded" if chain == -1 else f"_ch{chain}" if chain else "")
        row = dict(samples[codec,chain])
        selected = select_nodes(row,target,maximum)
        prediction = estimate(row,selected)
        plans.append(dict(codec=codec,chain_max=chain,**prediction))
        save_csv(args.output_dir/"node_plans.csv",plans)
        prefix = args.output_dir/tag
        stamp = args.output_dir/f"{tag}.ok.json"
        signature = dict(sample_identity,codec=codec,chain=chain,hard=hard,target=target,L=args.L,k=args.k,
                         rounds=args.rounds,threads=args.threads,beam=args.beamwidth,verify=args.verify_cache)
        summary_path = Path(str(prefix)+"_search_summary.csv")
        record = json.loads(stamp.read_text()) if stamp.exists() else {}
        have_bins = all(Path(f"{prefix}_{level}_{suffix}").exists() for level in args.L
                        for suffix in ("idx_uint32.bin", "dists_float.bin"))
        reused = args.resume and record.get("signature") == signature and summary_path.exists() and (not args.check_results or have_bins)
        if reused:
            actual, selected = record["actual"], record["nodes"]
            forest_stats = record.get("forest", {})
            print(f"REUSE completed {tag} nodes={selected}",flush=True)
        else:
            if stamp.exists(): stamp.unlink()
            for attempt in range(args.max_retries+1):
                command = common + ["--cache_codec",codec,"--num_nodes_to_cache",str(selected),
                    "--lvc_chain_max",str(chain if chain else 2),"--cache_budget_include_state",
                    "--cache_budget_bytes",str(hard),"--result_path",str(prefix)]
                if args.verify_cache and codec in DIFFERENTIAL: command.append("--verify_cache")
                log = args.output_dir/f"{tag}_attempt{attempt}.log"
                code = run_logged(command,log)
                actual = budget_from_log(log)
                forest_stats = forest_from_log(log)
                if actual and actual["total_bytes"] > hard and attempt < args.max_retries:
                    # One correction from observed allocation, not repeated full-index probing.
                    corrected = dict(row,nodes=str(actual["nodes"]),payload_bytes=str(actual["vector_payload_bytes"]),
                        record_metadata_bytes=str(actual["vector_metadata_bytes"]),state_cache_bytes=str(actual["state_cache_bytes"]),
                        nhood_sample_buckets=str(actual["nhood_buckets"]),coord_sample_buckets=str(actual["coord_buckets"]))
                    selected = select_nodes(corrected,target,selected-1)
                    print(f"CORRECT {tag}: actual exceeded hard cap; retry once at nodes={selected}",flush=True)
                    continue
                if code or not actual or actual["total_bytes"] > hard or actual["nodes"] != selected or actual.get("includes_state") != 1:
                    raise RuntimeError(f"{tag} failed/over budget; inspect {log}")
                break
        results = csv_rows(summary_path)
        if [int(r["L"]) for r in results] != args.L or any(int(r["CacheBytes"]) != actual["total_bytes"] for r in results):
            raise RuntimeError(f"Invalid final summary: {summary_path}")
        if (actual["graph_bytes"] + actual["vector_payload_bytes"] + actual["vector_metadata_bytes"] +
                actual["state_cache_bytes"] != actual["total_bytes"]):
            raise RuntimeError("Actual budget components do not sum to total")
        if not reused:
            save_json(stamp, dict(signature=signature,actual=actual,nodes=selected,forest=forest_stats))
        for result in results:
            summary.append(dict(codec=codec,chain_max=chain,nodes=selected,hard_budget_bytes=hard,
                planning_budget_bytes=target,pq_file_bytes=pq.stat().st_size,
                budget_utilization=actual["total_bytes"]/hard,
                sample_payload_ratio_f32=int(row["nodes"])*int(row["dim"])*4/int(row["payload_bytes"]),
                actual_payload_ratio_f32=selected*int(row["dim"])*4/actual["vector_payload_bytes"],
                sample_average_depth=row["average_depth"],sample_max_depth=row["max_depth"],
                actual_roots=forest_stats.get("roots", ""),actual_average_depth=forest_stats.get("average_depth", ""),
                actual_max_depth=forest_stats.get("max_depth", ""),
                peak_rss_after_cache_kib=result.get("RSS_After_Cache_Load(KB)", ""),
                current_rss_after_cache_kib=result.get("Current_RSS_After_Cache_Load(KB)", ""),
                peak_rss_after_search_kib=result.get("RSS_After_Search(KB)", ""),
                current_rss_after_search_kib=result.get("Current_RSS_After_Search(KB)", ""),
                estimated_initial_bytes=prediction["total_bytes"], actual_total_bytes=actual["total_bytes"],
                graph_bytes=actual["graph_bytes"],lookup_bytes=actual["graph_lookup_bytes"],
                payload_bytes=actual["vector_payload_bytes"],record_metadata_bytes=actual["vector_metadata_bytes"],
                state_cache_bytes=actual["state_cache_bytes"], L=result["L"],K=args.k,
                QPS=result["QPS"],mean_us=result["Mean Latency (mus)"],p999_us=result["99.9 Latency"],
                mean_ios=result["Mean IOs"],recall=result.get(f"Recall@{args.k}",""),
                cache_load_seconds=result["CacheLoadSeconds"],beamwidth=args.beamwidth,rounds=args.rounds,
                extra_candidate_filtering=0))
        save_csv(args.output_dir/"summary.csv",summary)
        successful[tag] = prefix
    if args.check_results:
        for tag,prefix in successful.items():
            if tag == "raw": continue
            for level in args.L:
                for suffix in ("idx_uint32.bin","dists_float.bin"):
                    compare_results(f"{successful['raw']}_{level}_{suffix}",f"{prefix}_{level}_{suffix}")
            print(f"PASS {tag}: exact IDs and distance bits vs raw for all L",flush=True)
    save_json(args.output_dir/"run_status.json", dict(status="complete",cases=len(cases),
              sample_nodes=args.sample_nodes,hard_budget_bytes=hard,planning_budget_bytes=target,
              results_compared_with_raw=args.check_results))
    print("PASS sampled budget: every final cache fits the hard cap including state cache", flush=True)
    print(f"Results: {args.output_dir/'summary.csv'}",flush=True)


if __name__ == "__main__":
    main()
