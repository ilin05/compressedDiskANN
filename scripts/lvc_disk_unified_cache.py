#!/usr/bin/env python3
"""Run serial fixed-node cache cases with unified index accounting and per-round results."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import struct
import subprocess
import time

from lvc_disk_sampled_budget import budget_from_log, forest_from_log

CODECS = ("raw", "alp", "dexor", "gorilla", "elf")
FIELDS = ("case", "codec", "nodes", "status", "error", "attempt", "exit_code", "L", "rounds", "measured_rounds",
          "recall_percent", "qps_mean", "aggregate_qps", "mean_us", "p999_us", "mean_ios", "mean_io_us",
          "mean_cpu_us", "decode_us_per_query", "minor_faults", "major_faults", "current_rss_kib", "peak_rss_kib",
          "cache_hash", "forest_hash", "roots", "average_depth", "max_depth", "pq_code_bytes", "pq_table_bytes", "entrypoint_bytes", "adjacency_bytes",
          "lookup_bytes", "payload_bytes", "record_metadata_bytes", "state_cache_bytes", "cache_bytes",
          "accounted_index_bytes", "payload_ratio_f32", "index_load_seconds", "cache_selection_seconds",
          "cache_data_load_seconds", "total_load_seconds", "forest_seconds", "alp_read_seconds",
          "alp_encode_seconds", "alp_merge_seconds", "lvc_encode_seconds", "propagation_seconds", "fallback_seconds", "process_wall_seconds", "cache_load_threads",
          "alp_encode_threads", "lvc_build_threads", "query_threads", "beamwidth", "source_dir")


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(str(tmp), str(path))


def write_summary(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(str(tmp), str(path))


def fields_from_line(text, marker):
    lines = [line for line in text.splitlines() if marker + " " in line]
    if len(lines) != 1:
        raise ValueError("Missing/duplicate " + marker)
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[0]))


def file_identity(path):
    stat = path.stat()
    return [str(path.absolute()), stat.st_size, stat.st_mtime_ns]


def read_bin_header(path):
    with path.open("rb") as f:
        data = f.read(8)
    if len(data) != 8:
        raise ValueError("Invalid bin header: " + str(path))
    return struct.unpack("<II", data)


def summarize_rounds(path, args):
    with path.open(newline="") as f:
        rounds = list(csv.DictReader(f))
    rows = []
    expected = {(L, n) for L in args.L for n in range(1, args.rounds + 1)}
    actual = [(int(r["L"]), int(r["round"])) for r in rounds]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete/duplicate query rounds")
    for L in args.L:
        group = [r for r in rounds if int(r["L"]) == L and int(r["round"]) >= 2]
        if any(int(r["K"]) != 1 or int(r["threads"]) != 1 or int(r["beamwidth"]) != args.beamwidth for r in group):
            raise ValueError("Query configuration mismatch")
        if any(int(r["queries"]) != args.expected_queries for r in group):
            raise ValueError("Query count differs from the full query file")
        mean = lambda key: statistics.mean(float(r[key]) for r in group)
        if any(not r["recall_percent"] for r in group):
            raise ValueError("Ground-truth recall missing")
        for r in group:
            if any(not math.isfinite(float(r[k])) or float(r[k]) <= 0 for k in ("qps", "wall_seconds", "mean_us")):
                raise ValueError("Invalid timing/QPS in query rounds")
            if not 0 <= float(r["recall_percent"]) <= 100:
                raise ValueError("Invalid recall in query rounds")
        rows.append(dict(L=L, rounds=args.rounds, measured_rounds=len(group), recall_percent=mean("recall_percent"),
                         qps_mean=mean("qps"), aggregate_qps=sum(int(r["queries"]) for r in group)/sum(float(r["wall_seconds"]) for r in group),
                         mean_us=mean("mean_us"), p999_us=mean("p999_us"), mean_ios=mean("mean_ios"),
                         mean_io_us=mean("mean_io_us"), mean_cpu_us=mean("mean_cpu_us"),
                         decode_us_per_query=statistics.mean(float(r["decode_us"])/int(r["queries"]) for r in group),
                         minor_faults=sum(int(r["minor_faults"]) for r in group),
                         major_faults=sum(int(r["major_faults"]) for r in group),
                         current_rss_kib=mean("current_rss_kib"), peak_rss_kib=max(int(r["peak_rss_kib"]) for r in group)))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for option in ("binary", "index", "query", "gt", "output"):
        p.add_argument("--"+option, type=Path, required=True)
    p.add_argument("--expected-points", type=int, default=1000000000)
    p.add_argument("--nodes", type=int, nargs="+", default=[10000000,25000000,50000000])
    p.add_argument("--codecs", nargs="+", choices=CODECS, default=list(CODECS))
    p.add_argument("--L", type=int, nargs="+", default=[1,2,3,4,5,6,8,10])
    p.add_argument("--load-threads", type=int, default=32)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--beamwidth", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verify-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args=p.parse_args()
    if args.rounds < 2 or args.load_threads < 1 or args.beamwidth < 1 or min(args.nodes) < 1 or min(args.L) < 1:
        p.error("positive nodes/threads/L/beamwidth and at least two rounds required")
    if len(set(args.nodes)) != len(args.nodes) or len(set(args.codecs)) != len(args.codecs) or len(set(args.L)) != len(args.L):
        p.error("Duplicate nodes, codecs or L")
    cases=[(codec,nodes) for codec in args.codecs for nodes in args.nodes]
    def command(codec,nodes,run_dir):
        cmd=[str(args.binary),"--data_type","float","--dist_fn","l2","--index_path_prefix",str(args.index),
             "--query_file",str(args.query),"--gt_file",str(args.gt),"--cache_codec",codec,
             "--num_nodes_to_cache",str(nodes),"--lvc_chain_max","2","--lvc_state_cache_ratio","0.01",
             "--cache_load_threads",str(args.load_threads),"--alp_encode_threads",str(args.load_threads),
             "--lvc_build_threads",str(args.load_threads),"--lvc_fallback_root_candidates","1",
             "--cache_budget_include_state","-K","1","-L",*map(str,args.L),"-T","1","-W",str(args.beamwidth),
             "-R",str(args.rounds),"--result_path",str(run_dir/"search"),"--per_round_csv",str(run_dir/"rounds.csv")]
        if args.verify_cache: cmd.append("--verify_cache")
        return cmd
    if args.dry_run:
        for codec,nodes in cases: print(shlex.join(command(codec,nodes,args.output/(codec+"_"+str(nodes))/"attempt_1")))
        return 0
    pq=Path(str(args.index)+"_pq_compressed.bin"); pivots=Path(str(args.index)+"_pq_pivots.bin")
    disk=Path(str(args.index)+"_disk.index")
    count,width=read_bin_header(pq); nq,dim=read_bin_header(args.query); ng,gtwidth=read_bin_header(args.gt)
    if count != args.expected_points or width != 32 or pq.stat().st_size != 8+count*width:
        raise ValueError("PQ shape/size differs from expected BIGANN configuration")
    if dim != 128 or not nq or args.query.stat().st_size != 8+nq*dim*4 or ng != nq or not gtwidth or args.gt.stat().st_size not in (8+ng*gtwidth*4,8+ng*gtwidth*8):
        raise ValueError("Query/ground-truth shape or file size invalid")
    args.expected_queries=nq
    if max(args.nodes)>count//2:
        raise ValueError("Requested cache exceeds 50% of indexed nodes")
    help_text=subprocess.check_output([str(args.binary),"--help"],stderr=subprocess.STDOUT,text=True)
    if any(flag not in help_text for flag in ("--cache_load_threads","--alp_encode_threads","--per_round_csv")):
        raise RuntimeError("search_disk_index missing required options; rebuild updated binary")
    paths=dict(binary=args.binary,disk=disk,pq=pq,pivots=pivots,query=args.query,gt=args.gt,
               runner=Path(__file__),log_parser=Path(__file__).with_name("lvc_disk_sampled_budget.py"))
    for suffix in ("_disk.index_medoids.bin", "_disk.index_centroids.bin", "_pq_pivots.bin_rotation_matrix.bin"):
        path=Path(str(args.index)+suffix)
        if path.exists(): paths[suffix]=path
    identity={"inputs":{k:file_identity(v) for k,v in paths.items()},
              "config":{"nodes":args.nodes,"codecs":args.codecs,"L":args.L,"load_threads":args.load_threads,
                        "rounds":args.rounds,"beamwidth":args.beamwidth,"verify_cache":args.verify_cache}}
    args.output.mkdir(parents=True,exist_ok=True)
    print("Output directory:",args.output,flush=True)
    with (args.output/".lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=args.output/"config.json"
        if config.exists():
            if not args.resume: raise RuntimeError("Existing output; set RESUME=1 to continue")
            if json.loads(config.read_text()) != identity: raise RuntimeError("Inputs/config changed; choose a new output directory")
        else:
            if args.resume: raise RuntimeError("Cannot resume without existing config")
            write_json(config,identity)
        all_rows=[]; hashes={}; forest_hashes={}; failed=[]
        for case_number,(codec,nodes) in enumerate(cases, 1):
            tag=codec+"_"+str(nodes); case_dir=args.output/tag; case_dir.mkdir(exist_ok=True)
            done=case_dir/"done.json"
            if done.exists():
                saved=json.loads(done.read_text()); rows=saved["rows"]
                print("REUSE",tag,flush=True)
            else:
                attempt=1
                while (case_dir/("attempt_"+str(attempt))).exists():attempt+=1
                run_dir=case_dir/("attempt_"+str(attempt));run_dir.mkdir()
                cmd=command(codec,nodes,run_dir); (run_dir/"command.txt").write_text(shlex.join(cmd)+"\n")
                print("RUN",tag,"attempt",attempt,flush=True)
                start=time.monotonic()
                with (run_dir/"search.log").open("w") as log:
                    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
                wall=time.monotonic()-start
                base=dict(case=tag,codec=codec,nodes=nodes,attempt=attempt,exit_code=result.returncode,
                          process_wall_seconds=wall,cache_load_threads=args.load_threads,
                          alp_encode_threads=args.load_threads if codec=="alp" else 0,
                          lvc_build_threads=args.load_threads if codec in ("dexor","gorilla","elf") else 0,
                          query_threads=1,beamwidth=args.beamwidth,source_dir=str(run_dir))
                try:
                    text=(run_dir/"search.log").read_text(errors="replace")
                    if result.returncode: raise ValueError("search failed: exit "+str(result.returncode))
                    cache=budget_from_log(run_dir/"search.log"); memory=fields_from_line(text,"INDEX_MEMORY")
                    cfg=fields_from_line(text,"CACHE_READ_CONFIG"); timing=fields_from_line(text,"LVC_LOAD_TIMING")
                    h=re.findall(r"Cache node list: count=(\d+) hash=(\d+)",text)
                    if not cache or len(h)!=1 or int(h[0][0])!=nodes or cache["nodes"]!=nodes or int(cfg["threads"])!=args.load_threads:
                        raise ValueError("Cache population/parallel load mismatch")
                    if cache.get("includes_state")!=1: raise ValueError("State cache not accounted")
                    index_bytes=int(memory["accounted_index_bytes"])
                    if int(memory["cache_bytes"])!=cache["total_bytes"] or index_bytes!=sum(int(memory[k]) for k in ("pq_code_bytes","pq_table_bytes","entrypoint_bytes","cache_bytes")):
                        raise ValueError("Index memory components do not sum")
                    forest=forest_from_log(run_dir/"search.log") if codec in ("dexor","gorilla","elf") else {}
                    if codec in ("dexor","gorilla","elf"):
                        if not forest or int(forest["chain_max"])!=2 or int(forest["max_depth"])>1 or int(forest["roots"])!=max(1,nodes//100):
                            raise ValueError("Missing/unexpected differential forest")
                        ft=fields_from_line(text,"LVC_BUILD_TIMING")
                        if int(ft["build_threads"])!=args.load_threads: raise ValueError("Differential build thread mismatch")
                        base.update(lvc_encode_seconds=ft["encode_seconds"],propagation_seconds=ft["propagation_seconds"],fallback_seconds=ft["fallback_seconds"])
                    base.update(status="ok",error="",cache_hash=h[0][1],forest_hash=forest.get("forest_hash",""),
                                roots=forest.get("roots",""),average_depth=forest.get("average_depth",""),max_depth=forest.get("max_depth",""),
                                pq_code_bytes=memory["pq_code_bytes"],pq_table_bytes=memory["pq_table_bytes"],entrypoint_bytes=memory["entrypoint_bytes"],
                                adjacency_bytes=cache["graph_bytes"]-cache["graph_lookup_bytes"],lookup_bytes=cache["graph_lookup_bytes"],
                                payload_bytes=cache["vector_payload_bytes"],record_metadata_bytes=cache["vector_metadata_bytes"],
                                state_cache_bytes=cache["state_cache_bytes"],cache_bytes=cache["total_bytes"],accounted_index_bytes=index_bytes,
                                payload_ratio_f32=nodes*dim*4/cache["vector_payload_bytes"],forest_seconds=forest.get("forest_seconds", ""))
                    base.update({k:timing.get(k,"") for k in ("index_load_seconds","cache_selection_seconds","cache_data_load_seconds","total_load_seconds")})
                    if codec=="alp":
                        t=fields_from_line(text,"ALP_CACHE_TIMING")
                        if int(t["threads"])!=args.load_threads: raise ValueError("ALP encoding thread mismatch")
                        base.update(alp_read_seconds=t["read_seconds"],alp_encode_seconds=t["encode_seconds"],alp_merge_seconds=t["merge_seconds"])
                    if args.verify_cache and codec=="alp" and "ALP_CACHE_VERIFY vectors=" not in text:
                        raise ValueError("ALP verification missing")
                    if args.verify_cache and codec in ("dexor","gorilla","elf") and "LVC disk cache PASS bitwise" not in text:
                        raise ValueError("Differential bitwise verification missing")
                    rows=[dict(base,**r) for r in summarize_rounds(run_dir/"rounds.csv",args)]
                    if nodes in hashes and hashes[nodes]!=base["cache_hash"]:raise ValueError("Cache-node hash differs for same node count")
                    if base["forest_hash"] and nodes in forest_hashes and forest_hashes[nodes]!=base["forest_hash"]:raise ValueError("Differential forest hashes differ")
                    write_json(done,dict(rows=rows))
                    print("PASS",tag,"index GiB",round(index_bytes/2**30,3),flush=True)
                except (OSError,ValueError,KeyError,TypeError,ZeroDivisionError) as exc:
                    rows=[dict(base,status="failed",error=str(exc))];failed.append(tag)
                    print("FAIL",tag,str(exc),"log",run_dir/"search.log",flush=True)
                write_json(run_dir/"result.json",dict(rows=rows))
            if rows[0].get("status")=="ok":
                h=rows[0]["cache_hash"]
                if nodes in hashes and hashes[nodes]!=h:raise RuntimeError("Reused cache hash mismatch")
                hashes[nodes]=h
                fh=rows[0].get("forest_hash")
                if fh:
                    if nodes in forest_hashes and forest_hashes[nodes]!=fh:raise RuntimeError("Reused forest hash mismatch")
                    forest_hashes[nodes]=fh
            all_rows.extend(rows)
            write_summary(args.output/"summary.csv",all_rows)
            write_json(args.output/"status.json",dict(completed_cases=case_number,total_cases=len(cases),failed_cases=failed,last_case=tag))
        print("Results:",args.output/"summary.csv",flush=True)
        return 1 if failed else 0


if __name__=="__main__":
    raise SystemExit(main())
