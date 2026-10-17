#!/usr/bin/env python3
"""Raw/ALP correctness with test_bigann's original batch scheduler and libaio."""
import argparse
import csv
import math
from pathlib import Path
import struct
import subprocess


def compare_bin(raw, alp, kind):
    fmt = "I" if kind == "ids" else "f"
    with raw.open("rb") as a, alp.open("rb") as b:
        ah, bh = a.read(8), b.read(8)
        if ah != bh or len(ah) != 8:
            raise RuntimeError(f"Shape mismatch: {raw} vs {alp}")
        n, k = struct.unpack("<II", ah)
        expected_size = 8 + n * k * 4
        if raw.stat().st_size != expected_size or alp.stat().st_size != expected_size:
            raise RuntimeError("Truncated search output")
        position = 0
        while True:
            x, y = a.read(1024 * 1024), b.read(1024 * 1024)
            if not x:
                break
            if x != y:
                for offset in range(0, len(x), 4):
                    if x[offset:offset + 4] != y[offset:offset + 4]:
                        ordinal = position + offset // 4
                        v1 = struct.unpack("<" + fmt, x[offset:offset + 4])[0]
                        v2 = struct.unpack("<" + fmt, y[offset:offset + 4])[0]
                        raise RuntimeError(f"FAIL {kind} query={ordinal // k} rank={ordinal % k} "
                                           f"raw={v1} alp={v2} raw_bits={x[offset:offset+4].hex()} "
                                           f"alp_bits={y[offset:offset+4].hex()}")
            position += len(x) // 4
    print(f"PASS {kind}: {n} queries x {k}, exact bits", flush=True)


def read_summary(prefix):
    with Path(str(prefix) + "_search_summary.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--index-prefix", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-nodes", type=int, default=10000)
    parser.add_argument("--threads", type=int, nargs="+", default=[1])
    parser.add_argument("--beamwidths", type=int, nargs="+", default=[1])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--k", type=int, nargs="+", default=[1])
    parser.add_argument("--cap-check", action="store_true")
    args = parser.parse_args()
    with Path(str(args.index_prefix) + "_pq_compressed.bin").open("rb") as handle:
        points, _ = struct.unpack("<II", handle.read(8))
    raw_limit = max(1, math.floor(points * .1 + .5))
    if not 0 < args.cache_nodes <= raw_limit:
        parser.error(f"Same-node correctness requires cache-nodes in [1, {raw_limit}]")
    if args.rounds < 1 or min(args.threads + args.beamwidths + args.k) < 1:
        parser.error("rounds/threads/beamwidths/K must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / "summary.csv"
    if summary.exists():
        summary.unlink()  # Do not leave a previous PASS summary after a failed rerun.
    summaries = []

    def run(codec, nodes, threads, beam, k, levels, tag):
        prefix = args.output.resolve() / f"{tag}_{codec}"
        command = [str(args.binary.resolve()), "--data_type", "float", "--dist_fn", "l2",
                   "--index_path_prefix", str(args.index_prefix.resolve()),
                   "--query_file", str(args.query.resolve()), "--gt_file", str(args.gt.resolve()),
                   "--result_path", str(prefix), "--cache_codec", codec,
                   "--num_nodes_to_cache", str(nodes), "-T", str(threads), "-W", str(beam),
                   "-K", str(k), "-R", str(args.rounds), "-L", *map(str, levels)]
        if codec == "alp":
            command.append("--verify_cache")
        with Path(str(prefix) + ".log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        rows = read_summary(prefix)
        if {int(r["L"]) for r in rows} != set(levels):
            raise RuntimeError(f"Missing L rows: {prefix}")
        for row in rows:
            summaries.append(dict(Phase=tag, Threads=threads, K=k, **row))
        return prefix, {int(r["L"]): r for r in rows}

    for threads in args.threads:
        for beam in args.beamwidths:
            for k in args.k:
                levels = [x for x in (1, 2, 3, 4, 5, 6, 8, 10) if x >= k]
                if not levels:
                    levels = [k]
                tag = f"check_T{threads}_W{beam}_K{k}"
                raw, rr = run("raw", args.cache_nodes, threads, beam, k, levels, tag)
                alp, ar = run("alp", args.cache_nodes, threads, beam, k, levels, tag)
                for level in levels:
                    a, b = rr[level], ar[level]
                    for column in ("ActualCachedNodes", "CacheListHash", "RawPayloadBytes"):
                        if a[column] != b[column]:
                            raise RuntimeError(f"Raw/ALP mismatch {column}: {a[column]} != {b[column]}")
                    if int(a["ActualCachedNodes"]) != args.cache_nodes:
                        raise RuntimeError("Failed to load requested cache count")
                    if a["Mean IOs"] != b["Mean IOs"]:
                        raise RuntimeError(f"I/O differs at L={level}: {a['Mean IOs']} != {b['Mean IOs']}")
                    for kind, suffix in (("ids", "idx_uint32.bin"), ("distances", "dists_float.bin")):
                        compare_bin(Path(f"{raw}_{level}_{suffix}"), Path(f"{alp}_{level}_{suffix}"), kind)
                    print(f"PASS {tag} L={level}: IDs/L2/cache list/I/O; "
                          f"QPS raw={a['QPS']} alp={b['QPS']} payload_ratio={b['PayloadRatio']}", flush=True)
    if args.cap_check:
        requested = min(points, max(raw_limit + 1, math.floor(points * .6 + .5)))
        for codec, fraction in (("raw", .1), ("alp", .5)):
            _, rows = run(codec, requested, 1, 1, 1, [1], "cap")
            expected = min(requested, max(1, math.floor(points * fraction + .5)))
            if int(rows[1]["ActualCachedNodes"]) != expected:
                raise RuntimeError(f"{codec} cap failed: expected {expected}")
            print(f"PASS {codec} cache cap: nodes={expected}", flush=True)
    path = args.output / "summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"PASS ALP-only correctness; summary: {path}")


if __name__ == "__main__":
    main()
