#!/usr/bin/env python3
"""Profile production DeXOR cache construction; queries are intentionally skipped."""
import argparse
import json
import os
from pathlib import Path
import re
import struct

from lvc_disk_payload_budget import actual_budget, estimate, select_nodes
from lvc_disk_sampled_budget import (budget_from_log, csv_rows, run_logged,
                                     save_csv, save_json, verify_samples)


def build_timings(path):
    lines = [line for line in path.read_text().splitlines() if line.startswith("LVC_BUILD_TIMING ")]
    if len(lines) != 1:
        raise ValueError("Expected one LVC_BUILD_TIMING line; rebuild search_disk_index: " + str(path))
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "index-prefix", "query", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--sample-nodes", type=int, default=1000)
    parser.add_argument("--budget-mib", type=float, default=20)
    parser.add_argument("--planning-budget-mib", type=float, default=19.5)
    args = parser.parse_args()
    budget = int(args.budget_mib * 2**20)
    planning = int(args.planning_budget_mib * 2**20)
    if not 0 < planning <= budget or not 1 <= args.sample_nodes <= 500000:
        parser.error("Invalid sample size or budget")
    out = args.output_dir.absolute()
    out.mkdir(parents=True, exist_ok=True)
    index = str(args.index_prefix.absolute())
    with open(index + "_pq_compressed.bin", "rb") as handle:
        points, pq_dim = struct.unpack("<II", handle.read(8))
    if points != 1000000 or pq_dim != 32:
        raise ValueError("Expected BIGANN-1M with 32-byte PQ codes")
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    common = [str(args.binary.absolute()), "--data_type", "float", "--dist_fn", "l2",
              "--index_path_prefix", index, "--query_file", str(args.query.absolute()),
              "--gt_file", "null", "-T", "1", "-K", "1", "-L", "1", "-W", "1", "-R", "1",
              "--lvc_state_cache_ratio", "0.01"]
    save_json(out / "config.json", dict(index=index, threads=1, budget_bytes=budget,
              planning_budget_bytes=planning, sample_nodes=args.sample_nodes, codec="dexor",
              chains=[2, -1], query_search=False, verify_cache=True,
              budget_definition="adjacency + payload + record metadata + state cache; excludes PQ/node lookups"))
    sample = out / "sample.csv"
    if sample.exists():
        sample.unlink()
    print("Sampling cache nodes for budget estimates (shared existing sampler, all codecs)", flush=True)
    code = run_logged(common + ["--cache_codec", "raw", "--num_nodes_to_cache", str(args.sample_nodes),
                      "--cache_budget_sample_csv", str(sample), "--result_path", str(out / "sample")],
                      out / "sample.log")
    if code:
        raise RuntimeError("Sampling failed; see sample.log")
    samples = verify_samples(csv_rows(sample), args.sample_nodes)
    rows = []
    failed = False
    for chain in (2, -1):
        tag = "dexor_ch2" if chain == 2 else "dexor_chunbounded"
        row = samples["dexor", chain]
        nodes = select_nodes(row, planning, points // 2)
        record = dict(case=tag, chain_max=chain, threads=1, planned_nodes=nodes,
                      budget_bytes=budget, estimated_budget_bytes=estimate(row, nodes)["budget_bytes"],
                      status="failed", error="")
        log = out / (tag + ".log")
        print("PROFILE {} nodes={} budget={} bytes".format(tag, nodes, budget), flush=True)
        try:
            code = run_logged(common + ["--cache_codec", "dexor", "--num_nodes_to_cache", str(nodes),
                "--lvc_chain_max", str(chain), "--cache_budget_include_state", "--cache_budget_probe_only",
                "--verify_cache", "--result_path", str(out / tag)], log)
            record.update(exit_code=code, process_wall_seconds=json.loads(
                log.with_suffix(".timing.json").read_text())["wall_seconds"])
            if code:
                raise RuntimeError("Cache construction failed: " + str(log))
            timing = build_timings(log)
            if int(timing["nodes"]) != nodes or int(timing["chain_max"]) != chain or int(timing["mode"]) != 2:
                raise ValueError("Timing configuration mismatch")
            stats = budget_from_log(log)
            if not stats or stats["nodes"] != nodes:
                raise ValueError("Missing or inconsistent cache budget statistics")
            used = actual_budget(stats)
            text = log.read_text()
            verified = re.search(r"LVC disk cache PASS bitwise vectors=(\d+)/(\d+)", text)
            if not verified or tuple(map(int, verified.groups())) != (nodes, nodes):
                raise ValueError("Full cache lossless verification missing")
            record.update(actual_nodes=stats["nodes"], actual_budget_bytes=used,
                          budget_utilization=used / budget, within_budget=used <= budget,
                          verified_vectors=nodes, status="ok" if used <= budget else "over_budget")
            for key, value in timing.items():
                record[key] = float(value) if key.endswith("seconds") else int(value)
            total = record["forest_seconds"] + record["encode_seconds"]
            for stage in ("root_selection", "propagation", "fallback", "finalize", "encode"):
                record[stage + "_percent"] = 100 * record[stage + "_seconds"] / total if total else 0
            record["forest_plus_encode_seconds"] = total
            print("{}: roots={:.6f}s propagation={:.6f}s fallback={:.6f}s encode={:.6f}s".format(
                tag, *(record[key + "_seconds"] for key in
                       ("root_selection", "propagation", "fallback", "encode"))), flush=True)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            record.update(status="failed", error=str(error))
            failed = True
            print("FAILED {}: {}".format(tag, error), flush=True)
        rows.append(record)
        save_csv(out / (tag + "_timings.csv"), [record])
        # Union fields also preserves a completed case if the other case fails.
        fields = list(dict.fromkeys(key for item in rows for key in item))
        save_csv(out / "summary.csv", [{key: item.get(key, "") for key in fields} for item in rows])
    print("Timing results:", out / "summary.csv", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
