#!/usr/bin/env python3
"""Budget arithmetic and subprocess-flow checks; fixtures are not ANN performance results."""
import csv
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from lvc_disk_payload_budget import CASES, actual_budget, estimate, select_nodes


FAKE_SEARCH = r'''#!/usr/bin/env python3
import argparse,csv,json,struct
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('index_path_prefix','query_file','gt_file','data_type','dist_fn','lvc_state_cache_ratio','lvc_chain_max','result_path','cache_budget_sample_csv'):
 p.add_argument('--'+name)
p.add_argument('--cache_codec');p.add_argument('--num_nodes_to_cache',type=int)
p.add_argument('-L',nargs='+',type=int)
a,rest=p.parse_known_args()
root=Path(__file__).parent
with (root/'calls.jsonl').open('a') as f:f.write(json.dumps(dict(codec=a.cache_codec,chain=a.lvc_chain_max,sample=bool(a.cache_budget_sample_csv),rest=rest))+'\n')
n=a.num_nodes_to_cache;codec=a.cache_codec;ch=int(a.lvc_chain_max or 0)
if a.cache_budget_sample_csv:
 rows=[]
 for c in ('raw','alp','dexor','gorilla','elf'):
  for chain in ((0,) if c in ('raw','alp') else (2,-1)):
   diff=chain!=0
   rows.append(dict(codec=c,chain_max=chain,nodes=n,dim=128,payload_bytes=n*(512 if c=='raw' else 128),
    record_metadata_bytes=n*24 if diff else 0,state_cache_bytes=n*64 if diff else 0,
    adjacency_bytes_per_node=132,forest_hash=chain,roots=n//100,average_depth=0.99,max_depth=1))
 with open(a.cache_budget_sample_csv,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 raise SystemExit(0)
if codec=='elf' and ch==2 and (root/'fail.flag').exists():raise SystemExit(7)
diff=codec not in ('raw','alp');payload=n*(512 if codec=='raw' else 256 if codec=='alp' else 128)
meta=n*24 if diff else 0;state=n*64 if diff else 0;lookup=1000000000
print('CACHE_BUDGET nodes={} graph_bytes={} graph_lookup_bytes={} vector_payload_bytes={} vector_metadata_bytes={} total_bytes={} state_cache_bytes={} includes_state=1 nhood_buckets=16 coord_buckets=16'.format(n,132*n+lookup,lookup,payload,meta,132*n+lookup+payload+meta+state,state))
if diff:print('CACHE_FOREST roots={} average_depth=0.99 max_depth=1'.format(n//100))
fields=['L','Recall@1','QPS','Mean Latency (mus)','99.9 Latency','Mean IOs','IndexLoadSeconds','CacheSelectionSeconds','CacheLoadSeconds','CacheDataLoadSeconds','ForestBuildSeconds','TotalLoadSeconds','RSS_After_Cache_Load(KB)','Current_RSS_After_Cache_Load(KB)','RSS_After_Search(KB)','Current_RSS_After_Search(KB)']
with open(a.result_path+'_search_summary.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for level in a.L:
  r={k:1 for k in fields};r.update(L=level,QPS=42,IndexLoadSeconds=1,CacheSelectionSeconds=2,CacheDataLoadSeconds=3,CacheLoadSeconds=5,TotalLoadSeconds=6,ForestBuildSeconds=0.25 if diff else 0)
  w.writerow(r)
for level in a.L:
 for suffix in ('idx_uint32.bin','dists_float.bin'):
  Path(a.result_path+'_'+str(level)+'_'+suffix).write_bytes(struct.pack('<II2I',2,1,0,0))
'''


class PayloadBudgetTests(unittest.TestCase):
    def test_accounting_and_selection(self):
        row = dict(nodes=1000, payload_bytes=128000, record_metadata_bytes=24000,
                   state_cache_bytes=64000, adjacency_bytes_per_node=132)
        before = estimate(row, 10000)
        row.update(nhood_sample_buckets=2**40, coord_sample_buckets=2**40)
        self.assertEqual(estimate(row, 10000), before)  # lookup never changes the plan
        for target in (1000, int(19.5 * 2**20), int(19.5 * 2**30)):
            n = select_nodes(row, target, 500000000)
            self.assertLessEqual(estimate(row, n)["budget_bytes"], target)
            self.assertGreater(estimate(row, n + 1)["budget_bytes"], target)
        stats = dict(graph_bytes=1000000132, graph_lookup_bytes=1000000000,
                     vector_payload_bytes=128, vector_metadata_bytes=24, state_cache_bytes=64,
                     total_bytes=1000000348, includes_state=1)
        self.assertEqual(actual_budget(stats), 348)
        stats["total_bytes"] += 1
        with self.assertRaises(ValueError):
            actual_budget(stats)

    def test_order_overbudget_failure_persistence_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "search_disk_index"
            binary.write_text(FAKE_SEARCH); binary.chmod(0o755)
            index = root / "index"
            Path(str(index) + "_disk.index").write_bytes(b"disk")
            Path(str(index) + "_pq_pivots.bin").write_bytes(b"pivots")
            with Path(str(index) + "_pq_compressed.bin").open("wb") as f:
                f.write(struct.pack("<II", 1000000, 32)); f.truncate(32000008)
            (root / "query").write_bytes(b"query"); (root / "gt").write_bytes(b"gt")
            (root / "fail.flag").touch()
            out = root / "results"
            command = [sys.executable, str(Path(__file__).with_name("lvc_disk_payload_budget.py")),
                "--binary", str(binary), "--index-prefix", str(index), "--query", str(root/"query"),
                "--gt", str(root/"gt"), "--output-dir", str(out), "--expected-points", "1000000",
                "--sample-nodes", "1000", "--budget-mib", "20", "--planning-budget-mib", "19.5",
                "--beamwidth", "1", "--rounds", "1", "--L", "1", "2", "--check-results", "--resume"]
            run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertNotEqual(run.returncode, 0)
            calls = [json.loads(x) for x in (root/"calls.jsonl").read_text().splitlines()]
            self.assertEqual(len(calls), 9)  # shared sample + exactly eight final attempts
            self.assertEqual([(c["codec"], int(c["chain"]) if c["codec"] not in ("raw","alp") else 0)
                              for c in calls[1:]], list(CASES))
            self.assertTrue(all("--cache_budget_bytes" not in c["rest"] for c in calls[1:]))
            with (out/"alp_results.csv").open() as f:
                alp = list(csv.DictReader(f))
            self.assertTrue(all(r["status"] == "over_budget" and r["qps"] == "42" for r in alp))
            self.assertTrue((out/"elf_chunbounded_results.csv").exists())
            with (out/"elf_ch2_results.csv").open() as f:
                self.assertEqual(next(csv.DictReader(f))["status"], "failed")
            with (out/"dexor_ch2_results.csv").open() as f:
                row = next(csv.DictReader(f))
                self.assertEqual(row["forest_build_seconds"], "0.25")
                self.assertEqual(row["total_load_seconds"], "6")
                self.assertEqual(row["result_bits_match_raw"], "True")
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(len((root/"calls.jsonl").read_text().splitlines()), 9)
            (root/"fail.flag").unlink()
            rerun = subprocess.run([c for c in command if c != "--resume"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(rerun.returncode, 0, rerun.stdout)
            self.assertEqual(json.loads((out/"run_status.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
