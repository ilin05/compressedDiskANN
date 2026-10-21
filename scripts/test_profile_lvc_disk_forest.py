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

FIXTURE = FAKE_SEARCH + r'''
threads=int(rest[rest.index('--lvc_build_threads')+1])
print('LVC_BUILD_TIMING nodes={} chain_max={} mode=2 roots={} rounds=3 fallback_nodes=2 build_threads={} root_selection_seconds=0.01 propagation_seconds=0.02 fallback_seconds=0.03 finalize_seconds=0.001 forest_seconds=0.062 encode_seconds=0.04'.format(n,ch,n//100,threads))
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
            binary.write_text(FAKE_SEARCH)
            result = subprocess.run(command, stdout=subprocess.DEVNULL)
            self.assertNotEqual(result.returncode, 0)
            rows = list(csv.DictReader((root / 'out/summary.csv').read_text().splitlines()))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['status'] == 'failed' for r in rows))


if __name__ == '__main__':
    unittest.main()
