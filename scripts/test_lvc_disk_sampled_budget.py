#!/usr/bin/env python3
"""Small deterministic checks for the sampled-budget arithmetic and validators."""
import tempfile
from pathlib import Path
import struct
import unittest
from lvc_disk_sampled_budget import buckets, estimate, select_nodes, verify_samples, compare_results, budget_from_log


def sample():
    return dict(nodes="1000000",payload_bytes="512000000",record_metadata_bytes="24000000",
        state_cache_bytes="66560000",adjacency_bytes_per_node="132",nhood_bucket_bytes="32",
        coord_bucket_bytes="24",nhood_max_load="0.5",coord_max_load="0.5",
        nhood_sample_buckets="2097152",coord_sample_buckets="2097152")


class BudgetTests(unittest.TestCase):
    def test_lookup_bytes(self):
        self.assertEqual(estimate(sample(),1000000)["lookup_bytes"],112 * (1 << 20))
        self.assertEqual(buckets(1048576,.5),2097152)
        self.assertEqual(buckets(1048577,.5),4194304)

    def test_all_components_and_boundary(self):
        row=sample()
        value=estimate(row,1000000)
        self.assertEqual(value["total_bytes"],512000000+24000000+66560000+132000000+112*(1<<20))
        for budget in (1<<20,31<<20,31<<30):
            n=select_nodes(row,budget,500000000)
            self.assertLessEqual(estimate(row,n)["total_bytes"],budget)
            self.assertGreater(estimate(row,n+1)["total_bytes"],budget)

    def test_hash_probe_growth(self):
        row=sample()
        row["nhood_sample_buckets"]="8388608"
        self.assertGreater(estimate(row,2000000)["lookup_bytes"],estimate(sample(),2000000)["lookup_bytes"])

    def test_common_forest(self):
        rows=[]
        for codec in ("raw","alp","dexor","gorilla","elf"):
            for chain in ((2,-1) if codec not in ("raw","alp") else (0,)):
                rows.append(dict(sample(),codec=codec,chain_max=str(chain),forest_hash=str(chain),roots="10000",average_depth="1"))
        self.assertEqual(len(verify_samples(rows,1000000)),8)
        rows[-1]["forest_hash"]="different"
        with self.assertRaises(RuntimeError): verify_samples(rows,1000000)

    def test_log_and_first_difference(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/"log").write_text("CACHE_BUDGET nodes=1 graph_bytes=132 vector_payload_bytes=512 vector_metadata_bytes=24 total_bytes=734 limit_bytes=1024 state_cache_bytes=66 includes_state=1\n")
            self.assertEqual(budget_from_log(p/"log")["total_bytes"],734)
            a,b=p/"a",p/"b"
            a.write_bytes(struct.pack("<II2I",2,1,3,4)); b.write_bytes(a.read_bytes())
            compare_results(a,b)
            b.write_bytes(struct.pack("<II2I",2,1,3,5))
            with self.assertRaisesRegex(RuntimeError,"query=1 rank=0"):
                compare_results(a,b)


if __name__ == "__main__":
    unittest.main()
