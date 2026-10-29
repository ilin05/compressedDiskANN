#!/usr/bin/env python3
"""Complete DEEP cross-graph results without rebuilding either graph or PQ."""

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import time

from summarize_lvc_cross_graph_transfer import CODECS, CHAINS, DEFAULT_OUTPUT_DIR, read_csv, write_csv

DATASET = "deep-image-96-angular"
HNSW_EFS = [10, 20, 30, 40, 60, 80, 100, 150, 200]
VAMANA_LS = [1, 2, 3, 4, 5, 6, 8, 10]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bin_shape(path, item_bytes=4):
    with path.open("rb") as stream:
        n, d = struct.unpack("<II", stream.read(8))
    require(n > 0 and d > 0 and path.stat().st_size == 8 + n * d * item_bytes,
            f"Invalid binary shape/size: {path}")
    return n, d


def signature(path):
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def check_input_samples(path, n, dim, fvecs):
    with path.open("rb") as stream:
        for row in (0, n // 2, n - 1):
            stream.seek(8 + row * dim * 4)
            payload = stream.read(dim * 4)
            values = struct.unpack("<%df" % dim, payload)
            require(all(math.isfinite(x) for x in values) and
                    abs(sum(x * x for x in values) - 1) < 1e-4,
                    f"Expected unit-normalized F32: {path} row={row}")
            if fvecs.exists():
                with fvecs.open("rb") as source:
                    source.seek(row * (4 + dim * 4))
                    require(source.read(4) == struct.pack("<I", dim) and
                            source.read(dim * 4) == payload,
                            f"Normalized HNSW/Vamana inputs differ: row={row} {path}")


def valid(check):
    try:
        check()
        return True
    except (OSError, ValueError, KeyError, struct.error, AssertionError):
        return False


def compression_check(prefix, n, dim, chain):
    rows = read_csv(Path(str(prefix) + "_compression.csv"))
    require(len(rows) == 3 and {r["codec"] for r in rows} == set(CODECS),
            f"Incomplete compression CSV: {prefix}")
    require(Path(str(prefix) + "_forest.csv").stat().st_size > 50, "Missing forest export")
    for row in rows:
        require(int(row["nodes"]) == n and int(row["roots"]) == max(1, n // 100), "Wrong nodes/roots")
        require(int(row["raw_f32_bytes"]) == n * dim * 4, "Wrong F32 baseline")
        require(chain == -1 or int(row["max_depth"]) < chain, "Depth bound exceeded")
        for key in ("payload_bytes", "metadata_bytes", "payload_ratio_f32", "full_index_ratio_f32"):
            require(math.isfinite(float(row[key])) and float(row[key]) > 0, f"Invalid {key}")
        require(int(row["record_file_materialized"]) == int(chain == 2), "Wrong record format")
    if chain == 2:
        for mode, codec in enumerate(("raw",) + CODECS, 1):
            graph, data = Path(str(prefix) + "_" + codec), Path(str(prefix) + "_" + codec + ".data")
            require(graph.stat().st_size == int(rows[0]["graph_bytes"]), f"Truncated graph: {graph}")
            with data.open("rb") as stream:
                header = struct.unpack("<IIQII", stream.read(24))
                require(header == (n, dim, 0x314154414456434C, 1, mode), f"Invalid header: {data}")
                stream.seek(24 + (0 if mode == 1 else n * 21))
                payload, = struct.unpack("<Q", stream.read(8))
            require(data.stat().st_size == 32 + (0 if mode == 1 else n * 21) + payload,
                    f"Truncated payload: {data}")
            if codec != "raw":
                row = next(r for r in rows if r["codec"] == codec)
                require(payload == int(row["payload_bytes"]) and
                        data.stat().st_size == payload + int(row["metadata_bytes"]),
                        f"CSV/record byte totals disagree: {data}")


def hnsw_search_check(path, rounds):
    rows = read_csv(path)
    expected = {(c, str(tls), ef) for c in CODECS for tls in (0, 1) for ef in HNSW_EFS}
    require(len(rows) == len(expected) and
            {(r["Algorithm"].lower(), r["UseTwoLevelSearch"], int(r["ef"])) for r in rows} == expected,
            "Incomplete HNSW ef/TLS sweep")
    for r in rows:
        require(r["Dataset"] == DATASET and int(r["K"]) == 1 and int(r["ChainMaxLength"]) == 2,
                "Wrong HNSW dataset/K/chain")
        require(r["RootPolicy"] == "degree" and int(r["UseCacheTop1Pct"]) == 1 and
                int(r["Num_Rounds"]) == rounds, "Wrong HNSW policy/cache/rounds")
        require(r["UseTwoLevelSearch"] == "0" or abs(float(r["TlsRatio"]) - .2) < 1e-6,
                "Wrong HNSW alpha")
        require(0 <= float(r["Recall"]) <= 1 and float(r["TimePerQuery(us)"]) > 0, "Invalid HNSW metrics")


class Runner:
    def __init__(self, directory):
        self.directory = directory
        self.directory.mkdir(exist_ok=True)
        self.failures = []

    def run(self, name, command, inputs=(), outputs=(), check=lambda: None, cwd=None, finish=lambda: None):
        marker = self.directory / (name + ".json")
        key = dict(command=[str(x) for x in command], inputs=[signature(p) for p in inputs])
        if marker.exists() and valid(check):
            saved = json.loads(marker.read_text())
            if saved.get("key") == key and saved.get("outputs") == [signature(p) for p in outputs]:
                print("REUSE", name, flush=True)
                return True
        log = self.directory / (name + ".log")
        started = time.time()
        status, code = "failed", -1
        print("RUN", name, shlex.join([str(x) for x in command]), "log=", log, flush=True)
        try:
            with log.open("w") as handle:
                result = subprocess.run([str(x) for x in command], cwd=cwd, stdout=handle,
                                        stderr=subprocess.STDOUT)
            code = result.returncode
            require(code == 0, f"exit={code}; see {log}")
            finish()
            check()
            marker.write_text(json.dumps(dict(key=key, outputs=[signature(p) for p in outputs]), indent=2))
            status = "ok"
            print("PASS", name, flush=True)
            return True
        except (OSError, ValueError) as error:
            self.failures.append(f"{name}: {error}")
            print("FAIL", self.failures[-1], flush=True)
            return False
        finally:
            journal = self.directory / "status.csv"
            new = not journal.exists()
            with journal.open("a", newline="") as handle:
                writer = csv.writer(handle)
                if new:
                    writer.writerow(("stage", "status", "exit_code", "seconds", "log", "started_unix"))
                writer.writerow((name, status, code, time.time() - started, str(log), started))


def make_smoke(graph, train, out, count, dim):
    """Induced prefix subgraph, used only as a crash/losslessness gate."""
    sample = out / "smoke.fbin"
    with train.open("rb") as src, sample.open("wb") as dst:
        src.seek(8)
        dst.write(struct.pack("<II", count, dim))
        dst.write(src.read(count * dim * 4))
    target = out / "smoke_graph"
    with graph.open("rb") as src, target.open("wb") as dst:
        _, degree, entry, frozen = struct.unpack("<QIIQ", src.read(24))
        require(frozen == 0, "Frozen Vamana points are unsupported")
        dst.write(struct.pack("<QIIQ", 0, degree, entry if entry < count else 0, 0))
        for _ in range(count):
            length, = struct.unpack("<I", src.read(4))
            neighbors = struct.unpack("<%dI" % length, src.read(length * 4))
            keep = [i for i in neighbors if i < count]
            dst.write(struct.pack("<I", len(keep)))
            dst.write(struct.pack("<%dI" % len(keep), *keep))
        size = dst.tell()
        dst.seek(0)
        dst.write(struct.pack("<Q", size))
    return target, sample


def vamana_search_check(out, codec, tls, nq):
    base = out / f"vamana_{codec}_k1_tls{tls}"
    rows = read_csv(Path(str(base) + ".csv"))
    require(len(rows) == len(VAMANA_LS) and sorted(int(r["Ls"]) for r in rows) == VAMANA_LS,
            f"Incomplete Vamana L sweep: {base}")
    for r in rows:
        require(0 <= float(r["Recall@1"]) <= 100 and float(r["Mean Latency (mus)"]) > 0,
                f"Invalid Vamana metrics: {base}")
    for width in VAMANA_LS:
        for kind in ("idx_uint32.bin", "dists_float.bin"):
            require(bin_shape(Path(f"{base}_{width}_{kind}")) == (nq, 1), "Wrong result shape")


def retrieval_summary(out, rounds):
    hnsw_search_check(out / "hnsw_search_k1.csv", rounds)
    rows = []
    for r in read_csv(out / "hnsw_search_k1.csv"):
        rows.append(dict(dataset=DATASET, graph="HNSW", codec=r["Algorithm"].lower(), chain_max=2,
                         k=1, tls=r["UseTwoLevelSearch"], alpha=r["TlsRatio"], search_width=r["ef"],
                         recall=r["Recall"], qps=r["QPS"], mean_us=r["TimePerQuery(us)"],
                         tail_us=r["P99_Latency(us)"], tail_quantile="0.99"))
    for codec in ("raw",) + CODECS:
        for tls in (0, 1):
            for r in read_csv(out / f"vamana_{codec}_k1_tls{tls}.csv"):
                rows.append(dict(dataset=DATASET, graph="Vamana", codec=codec, chain_max=2, k=1,
                                 tls=tls, alpha=.2 if tls else 0, search_width=r["Ls"],
                                 recall=float(r["Recall@1"]) / 100, qps=r["QPS"],
                                 mean_us=r["Mean Latency (mus)"], tail_us=r["99.9 Latency"],
                                 tail_quantile="0.999"))
    write_csv(out / "retrieval_ch2.csv", rows)


def main():
    repo = Path(__file__).resolve().parents[1]
    default_hnsw = repo.parent / "hnswlib_cpp_py"
    if not default_hnsw.exists():
        default_hnsw = Path.home() / "HNSWCW/compressedhnsw"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(os.getenv("OUTPUT_DIR", str(repo / "build/data/cross_graph_transfer" / DATASET))))
    parser.add_argument("--hnsw-dir", type=Path, default=Path(os.getenv("HNSW_DIR", str(default_hnsw))))
    parser.add_argument("--diskann-build", type=Path, default=Path(os.getenv("DISKANN_BUILD_DIR", str(repo / "build"))))
    parser.add_argument("--hnsw-build", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=int(os.getenv("BUILD_THREADS", "32")))
    parser.add_argument("--rounds", type=int, default=int(os.getenv("ROUNDS", "3")))
    parser.add_argument("--smoke-nodes", type=int, default=1024)
    parser.add_argument("--plan", action="store_true", help="Validate prerequisites and print reuse decisions; do not build/run")
    parser.add_argument("--smoke-only", action="store_true", help="Only run the small real-DEEP losslessness gate")
    parser.add_argument("--skip-build", action="store_true", default=os.getenv("SKIP_BUILD", "0") == "1")
    args = parser.parse_args()
    require(args.threads > 0 and args.rounds > 0 and args.smoke_nodes >= 100, "Invalid threads/rounds/sample size")
    out = args.output.resolve()
    train, query, gt = [out / f"{DATASET}_{s}" for s in ("train.fbin", "test.fbin", "neighbors.bin")]
    n, dim = bin_shape(train)
    nq, qdim = bin_shape(query)
    require(dim == 96 and qdim == dim and n == 9990000, "Expected complete DEEP: 9,990,000 x 96")
    require(bin_shape(gt)[0] == nq, "Query/GT count mismatch")
    config = json.loads((out / "config.json").read_text())
    require(config.get("preprocessing") == "unit_normalized_f32" and config.get("input_precision") == "F32",
            "Keep the original common normalized-F32 inputs; config disagrees")
    normalized = out / "normalized_inputs"
    check_input_samples(train, n, dim, normalized / f"{DATASET}_train.fvecs")
    check_input_samples(query, nq, dim, normalized / f"{DATASET}_test.fvecs")
    graph = out / "vamana_R32_L50_A1.2"
    with graph.open("rb") as stream:
        size, _, _, frozen = struct.unpack("<QIIQ", stream.read(24))
    require(size == graph.stat().st_size and frozen == 0, "Incomplete Vamana graph")
    pq = [Path(str(train) + f"pq12_{kind}.bin") for kind in ("pivots", "compressed")]
    require(bin_shape(pq[1], 1) == (n, 12) and pq[0].stat().st_size > 0, "Invalid existing dim/8 PQ")
    hrows = read_csv(out / "hnsw_build.csv")
    require(len(hrows) == 12 and {(r["Algorithm"].lower(), int(r["ChainMaxLength"])) for r in hrows}
            == {(c, ch) for c in CODECS for ch, _ in CHAINS}, "Incomplete HNSW compression matrix")
    for r in hrows:
        require(r["RootPolicy"] == "degree" and int(r["RawF32Bytes"]) == n * dim * 4,
                "HNSW must use degree roots and the same full F32 input")
    prefixes = {ch: out / ("vamana_lvc" + suffix) for ch, suffix in CHAINS}
    reusable = {ch: valid(lambda ch=ch: compression_check(prefixes[ch], n, dim, ch)) for ch, _ in CHAINS}
    hsearch_ok = valid(lambda: hnsw_search_check(out / "hnsw_search_k1.csv", args.rounds))
    print(json.dumps(dict(dataset=DATASET, vectors=n, dim=dim, reuse_hnsw_build=True,
                          reuse_hnsw_search=hsearch_ok, reuse_vamana_graph=True, reuse_pq=True,
                          compression_reusable=reusable, load_threads=args.threads, search_threads=1,
                          rounds=args.rounds, hnsw_ef=HNSW_EFS, vamana_L=VAMANA_LS), indent=2), flush=True)
    if args.plan:
        return 0

    state = out / "deep_resume"
    state.mkdir(exist_ok=True)
    lock = (state / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runner = Runner(state)
    dbuild = args.diskann_build.resolve()
    hbuild = (args.hnsw_build or Path(os.getenv("HNSW_BUILD_DIR", str(args.hnsw_dir / "build_lvc_sift")))).resolve()
    converter, verifier, searcher = [dbuild / "apps" / p for p in ("convert_lvc_vamana", "verify_lvc_vamana", "search_memory_index")]
    if not args.skip_build:
        targets = ["convert_lvc_vamana", "verify_lvc_vamana"] + ([] if args.smoke_only else ["search_memory_index"])
        subprocess.run(["cmake", "--build", str(dbuild), "--target"] + targets +
                       ["--parallel", str(args.threads)], check=True)
    for binary in (converter, verifier) if args.smoke_only else (converter, verifier, searcher):
        require(binary.is_file() and os.access(binary, os.X_OK), f"Missing executable: {binary}")

    # Keep smoke inputs unchanged on resume so completed gate manifests remain valid.
    smoke_graph, smoke_train = state / "smoke_graph", state / "smoke.fbin"
    smoke_input = dict(train=signature(train), graph=signature(graph), count=min(args.smoke_nodes, n))
    smoke_key = state / "smoke_inputs.json"
    if not (smoke_key.exists() and json.loads(smoke_key.read_text()) == smoke_input
            and smoke_graph.exists() and smoke_train.exists()):
        make_smoke(graph, train, state, smoke_input["count"], dim)
        smoke_key.write_text(json.dumps(smoke_input))
    for ch, _ in CHAINS:
        prefix = state / f"smoke_ch{ch}"
        # Smoke chain=2 also materializes records, exercising the searchable format.
        command = [converter, smoke_graph, smoke_train, prefix, "--chain_max", ch, "--threads", args.threads]
        if ch != 2:
            command.append("--compression_only")
        if not runner.run(f"smoke_ch{ch}", command, inputs=(converter, smoke_graph, smoke_train),
                          outputs=(Path(str(prefix) + "_compression.csv"),),
                          check=lambda ch=ch, p=prefix: compression_check(p, smoke_input["count"], dim, ch)):
            (state / "decision.json").write_text(json.dumps(dict(status="blocked_by_smoke_failure", failures=runner.failures), indent=2))
            return 1
    # Even root-child records must pass full decode, not just conversion.
    for codec in ("raw",) + CODECS:
        if not runner.run("smoke_verify_" + codec, [verifier, smoke_train, state / f"smoke_ch2_{codec}.data"],
                          inputs=(verifier, smoke_train, state / f"smoke_ch2_{codec}.data")):
            return 1
    if args.smoke_only:
        print("PASS real-DEEP subset: all codecs and chain bounds; full DEEP remains untested")
        return 0

    for ch, _ in CHAINS:
        prefix = prefixes[ch]
        if reusable[ch]:
            print("REUSE complete compression:", prefix, flush=True)
            continue
        # Publish only after all three codecs complete; preserve original artifacts on failure.
        work = state / f"pending_ch{ch}"
        command = [converter, graph, train, work, "--chain_max", ch, "--threads", args.threads]
        if ch != 2:
            command.append("--compression_only")
        def publish(ch=ch, work=work, prefix=prefix):
            compression_check(work, n, dim, ch)
            suffixes = ["_forest.csv", "_compression.csv"]
            if ch == 2:
                suffixes = ["_" + c + ext for c in ("raw",) + CODECS for ext in ("", ".data")] + suffixes
            for suffix in suffixes:
                Path(str(work) + suffix).replace(Path(str(prefix) + suffix))
        runner.run(f"compression_ch{ch}", command, inputs=(converter, graph, train),
                   outputs=(Path(str(prefix) + "_compression.csv"),), finish=publish,
                   check=lambda ch=ch: compression_check(prefixes[ch], n, dim, ch))

    chain2_ok = valid(lambda: compression_check(prefixes[2], n, dim, 2))
    verified = searched = chain2_ok
    if not chain2_ok:
        runner.failures.append("Chain=2 compression/records are incomplete; skip dependent verification/search")
    for codec in ("raw",) + CODECS:
        if not chain2_ok:
            continue
        data = Path(str(prefixes[2]) + "_" + codec + ".data")
        if not data.exists():
            runner.failures.append("Missing chain=2 records: " + str(data))
            verified = searched = False
            continue
        if not runner.run("verify_" + codec, [verifier, train, data], inputs=(verifier, train, data)):
            verified = searched = False
            continue
        index = Path(str(prefixes[2]) + "_" + codec)
        for tls in (0, 1):
            base = out / f"vamana_{codec}_k1_tls{tls}"
            result_csv = Path(str(base) + ".csv")
            bins = [Path(f"{base}_{width}_{kind}") for width in VAMANA_LS
                    for kind in ("idx_uint32.bin", "dists_float.bin")]
            command = [searcher, "--data_type", "float", "--dist_fn", "l2", "--index_path_prefix", index,
                       "--query_file", query, "--gt_file", gt, "-K", 1, "-L"] + VAMANA_LS + [
                       "-T", 1, "-R", args.rounds, "--use_pq_dist", tls, "--pq_bytes", 12,
                       "--pq_prefix", train, "--pq_exact_rerank_ratio", .2, "--lvc_state_cache_ratio", .01,
                       "--result_path", base]
            success = runner.run(f"search_{codec}_tls{tls}", command,
                                 inputs=(searcher, index, data, query, gt, *pq), outputs=(result_csv, *bins),
                                 check=lambda c=codec, t=tls: vamana_search_check(out, c, t, nq),
                                 finish=lambda i=index, r=result_csv: shutil.copyfile(Path(str(i) + "_K1_T1_search_result.csv"), r))
            searched = success and searched
    if not hsearch_ok:
        normalized = out / "normalized_inputs"
        for suffix in ("test.fvecs", "neighbors.ivecs"):
            require((normalized / f"{DATASET}_{suffix}").is_file(), "Missing normalized HNSW queries/GT")
        for codec in ("DeXOR", "Gorilla", "Elf"):
            require((out / f"{DATASET}_{codec}_ch2_degree_pq.bin").is_file(), "Missing HNSW chain=2 index")
        if not args.skip_build:
            subprocess.run(["cmake", "--build", str(hbuild), "--target", "test_search_compressed_hnsw_ablation",
                            "--parallel", str(args.threads)], check=True)
        hbinary = hbuild / "test_search_compressed_hnsw_ablation"
        command = [hbinary, "--base_dir", normalized, "--dataset", DATASET, "--algorithm", "DeXOR", "Gorilla", "Elf",
                   "--chain_max", 2, "--root_policy", "degree", "--threads", 1, "--k", 1, "--ef"] + HNSW_EFS + [
                   "--use_cache", 1, "--use_tls", 0, 1, "--tls_ratio", .2, "--no_early_stop", "--num_rounds", args.rounds,
                   "--output_csv", "hnsw_search_k1.csv"]
        hsearch_ok = runner.run("hnsw_search", command, inputs=(hbinary,), outputs=(out / "hnsw_search_k1.csv",),
                               check=lambda: hnsw_search_check(out / "hnsw_search_k1.csv", args.rounds), cwd=out)
    correctness = []
    if verified and searched:
        # Cache/codecs must preserve the result under each fixed filtering setting.
        # TLS-on is compared with raw TLS-on, never with exact TLS-off.
        for tls in (0, 1):
            for width in VAMANA_LS:
                for codec in CODECS:
                    for kind in ("idx_uint32.bin", "dists_float.bin"):
                        baseline = (out / f"vamana_raw_k1_tls{tls}_{width}_{kind}").read_bytes()
                        actual = (out / f"vamana_{codec}_k1_tls{tls}_{width}_{kind}").read_bytes()
                        same = baseline == actual
                        first = "" if same else next((i for i, (a, b) in enumerate(zip(baseline, actual)) if a != b), min(len(baseline), len(actual)))
                        correctness.append(dict(codec=codec, tls=tls, L=width, kind=kind, exact_match=int(same), first_byte=first))
        write_csv(state / "correctness.csv", correctness)
    all_exact = bool(correctness) and all(r["exact_match"] for r in correctness)
    if verified and searched and hsearch_ok and all_exact:
        retrieval_summary(out, args.rounds)
    compression_complete = all(valid(lambda ch=ch: compression_check(prefixes[ch], n, dim, ch)) for ch, _ in CHAINS)
    complete = compression_complete and verified and searched and hsearch_ok and all_exact and not runner.failures
    (state / "decision.json").write_text(json.dumps(dict(
        status="complete" if complete else "incomplete", compression_complete=compression_complete,
        full_f32_lossless=verified, retrieval_complete=searched and hsearch_ok, exact_raw_matches=all_exact,
        include_deep=complete, exclude_expanding_codecs=False,
        note="Retain DEEP and compression ratios below 1; never infer success from a forest CSV alone.",
        failures=runner.failures), indent=2))
    if complete:
        subprocess.run(["python3", str(Path(__file__).with_name("summarize_lvc_cross_graph_transfer.py")),
                        "--root", str(out.parent), "--output-dir", str(DEFAULT_OUTPUT_DIR)], check=True)
        print("PASS DEEP complete; full matrix and recall-aligned summaries refreshed:", out.parent)
        print("Published summary CSVs:", DEFAULT_OUTPUT_DIR)
    else:
        print("INCOMPLETE: inspect", state / "decision.json", "and", state / "status.csv")
    return 0 if complete else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
