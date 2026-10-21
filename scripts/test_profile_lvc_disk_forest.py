#!/usr/bin/env python3
"""Driver/accounting fixtures only; no ANN performance numbers are measured."""
import csv
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from test_lvc_disk_payload_budget import FAKE_SEARCH

SAMPLE_FIXTURE = FAKE_SEARCH.replace('rows.append(dict(codec=c,',
    "rows.append(dict(fallback_root_candidates=int(rest[rest.index('--lvc_fallback_root_candidates')+1]),codec=c,")
FIXTURE = SAMPLE_FIXTURE + r'''
threads=int(rest[rest.index('--lvc_build_threads')+1])
policy=int(rest[rest.index('--lvc_fallback_root_candidates')+1])
print('LVC_BUILD_TIMING nodes={} chain_max={} mode=2 roots={} rounds=3 fallback_nodes=2 build_threads={} fallback_root_candidates={} fallback_tree_distances=20 fallback_root_distances=10 fallback_bucket_seconds=0.005 fallback_root_search_seconds=0.01 fallback_tree_search_seconds=0.015 fallback_max_tree_nodes=15 root_selection_seconds=0.01 propagation_seconds=0.02 fallback_seconds=0.03 finalize_seconds=0.001 forest_seconds=0.062 encode_seconds=0.04'.format(n,ch,n//100,threads,policy))
print('LVC_CACHE_LOAD_TIMING cache_read_seconds=0.1 remap_seconds=0.1 compact_seconds=0.1 state_seconds=0.1 verify_seconds=0.1 cache_total_seconds=0.7 cache_other_seconds=0.098')
print('LVC_LOAD_TIMING index_load_seconds=1 cache_selection_seconds=0.2 cache_data_load_seconds=0.8 total_load_seconds=2')
print('LVC disk cache PASS bitwise vectors={}/{} coordinates=1/1'.format(n,n))
'''


class ForestProfileTests(unittest.TestCase):
    def test_configuration_accounting_and_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'fake'
            binary.write_text(FIXTURE); binary.chmod(0o755)
            index = root / 'index'
            Path(str(index) + '_pq_compressed.bin').write_bytes(struct.pack('<II', 1000000000, 32))
            query = root / 'query'; query.write_bytes(b'')
            command = [sys.executable, '-B', str(Path(__file__).with_name('profile_lvc_disk_forest.py')),
                       '--binary', str(binary), '--index-prefix', str(index), '--query', str(query),
                       '--output-dir', str(root / 'out'), '--expected-points', '1000000000',
                       '--budget-mib', '1024', '--planning-budget-mib', '998.4', '--build-threads', '32']
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            rows = list(csv.DictReader((root / 'out/summary.csv').read_text().splitlines()))
            self.assertEqual([int(r['chain_max']) for r in rows], [2, -1])
            for row in rows:
                self.assertEqual(row['status'], 'ok')
                self.assertEqual(int(row['build_threads']), 32)
                self.assertAlmostEqual(float(row['load_without_verification_seconds']), 1.9)
                stages = ('index_load', 'cache_selection', 'cache_read', 'remap', 'forest', 'encode',
                          'compact', 'state', 'verify', 'cache_other', 'cache_return', 'load_other')
                self.assertAlmostEqual(sum(float(row[s + '_load_percent']) for s in stages), 100)
            calls = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(calls), 3)
            self.assertTrue(all('--cache_budget_probe_only' in c['rest'] and '--verify_cache' in c['rest']
                                for c in calls[1:]))
            # Older binaries cannot silently generate a successful zero-time report.
            binary.write_text(SAMPLE_FIXTURE)
            result = subprocess.run(command, stdout=subprocess.DEVNULL)
            self.assertNotEqual(result.returncode, 0)
            rows = list(csv.DictReader((root / 'out/summary.csv').read_text().splitlines()))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['status'] == 'failed' for r in rows))
            # The new wrapper runs only the unbounded case after one shared sample.
            binary.write_text(FIXTURE)
            subprocess.run(command + ['--chains', '-1'], check=True, stdout=subprocess.DEVNULL)
            rows = list(csv.DictReader((root / 'out/summary.csv').read_text().splitlines()))
            self.assertEqual([int(r['chain_max']) for r in rows], [-1])
            self.assertEqual(int(rows[0]['fallback_root_candidates']), 1)
            subprocess.run(command + ['--chains', '-1', '--fallback-root-candidates', '0'],
                           check=True, stdout=subprocess.DEVNULL)
            rows = list(csv.DictReader((root / 'out/summary.csv').read_text().splitlines()))
            self.assertEqual(int(rows[0]['fallback_root_candidates']), 0)


if __name__ == '__main__':
    unittest.main()
