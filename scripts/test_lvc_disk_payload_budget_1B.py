#!/usr/bin/env python3
"""Full wrapper flow with fixtures; no BIGANN performance measurements."""
import csv
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from lvc_disk_payload_budget import CASES
from test_profile_lvc_disk_forest import FIXTURE


# Executed at the end of each fake production process. The previous group must
# already have been atomically committed before this next process was launched.
CHECK_PREVIOUS = r'''
tags=['raw','alp','dexor_ch2','gorilla_ch2','elf_ch2','dexor_chunbounded','gorilla_chunbounded','elf_chunbounded']
tag=codec if not diff else codec+('_ch2' if ch==2 else '_chunbounded')
ordinal=tags.index(tag)
if ordinal:
 out=Path(a.result_path).parent
 previous=list(csv.DictReader((out/(tags[ordinal-1]+'_results.csv')).open()))
 summary=list(csv.DictReader((out/'summary.csv').open()))
 assert len(previous)==8 and summary[-1]['case']==tags[ordinal-1]
'''


class FullBudgetWrapperTests(unittest.TestCase):
    def test_order_threads_rounds_phases_persistence_and_resume(self):
        self.exercise_wrapper(k=1)

    def test_k10_wrapper_recall_columns_and_resume(self):
        self.exercise_wrapper(k=10)

    def exercise_wrapper(self, k):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'apps').mkdir()
            binary = root / 'apps/search_disk_index'
            fixture = FIXTURE.replace('rounds=3 fallback_nodes', 'rounds=8 fallback_nodes')
            fixture = fixture.replace("'Recall@1'", "'Recall@{}'".format(k))
            # Exercise the progress-line prefix that occurs in real raw/ALP logs.
            fixture = fixture.replace("print('LVC_CACHE_LOAD_TIMING ", "print('Loading the cache list into memory..LVC_CACHE_LOAD_TIMING ")
            binary.write_text(fixture + CHECK_PREVIOUS)
            binary.chmod(0o755)
            index = root / 'index'
            Path(str(index) + '_disk.index').write_bytes(b'disk')
            Path(str(index) + '_pq_pivots.bin').write_bytes(b'pivots')
            # Sparse file: validates the real 1B header/file-size check without
            # allocating or reading 32 GB of test data.
            with Path(str(index) + '_pq_compressed.bin').open('wb') as handle:
                handle.write(struct.pack('<II', 1000000000, 32)); handle.truncate(32000000008)
            query, truth = root / 'query', root / 'gt'
            query.write_bytes(b'query'); truth.write_bytes(b'gt')
            environment = dict(os.environ)
            for key in ('ROUNDS', 'BUDGET_MIB', 'PLANNING_BUDGET_MIB', 'SAMPLE_NODES'):
                environment.pop(key, None)
            environment.update(SKIP_BUILD='1', BUILD_DIR=str(root), INDEX_PREFIX=str(index),
                               QUERY_FILE=str(query), GT_FILE=str(truth), OUTPUT_DIR=str(root / 'out'),
                               LVC_BUILD_THREADS='32', RESUME='1', VERIFY_CACHE='0', PERF_W='2')
            wrapper = Path(__file__).with_name('test_lvc_disk_payload_budget_bigann_1B' +
                                               ('_k10' if k == 10 else '') + '.sh')
            command = ['bash', str(wrapper)]
            first = subprocess.run(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(first.returncode, 0, first.stdout)
            calls = [json.loads(line) for line in (root / 'apps/calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(calls), 9)
            self.assertTrue(calls[0]['sample'])
            self.assertEqual([(c['codec'], int(c['chain']) if c['codec'] not in ('raw', 'alp') else 0)
                              for c in calls[1:]], list(CASES))
            for call in calls:
                for flag, value in (('-T', '1'), ('-R', '3'), ('--lvc_build_threads', '32'),
                                    ('--lvc_fallback_root_candidates', '1'), ('-K', str(k))):
                    self.assertEqual(call['rest'][call['rest'].index(flag) + 1], value)
                self.assertNotIn('--cache_budget_probe_only', call['rest'])
            rows = list(csv.DictReader((root / 'out/summary.csv').read_text().splitlines()))
            self.assertEqual(len(rows), 64)
            self.assertEqual([int(r['L']) for r in rows[:8]],
                             [1, 2, 3, 4, 5, 6, 8, 10] if k == 1 else [10, 11, 12, 13, 14, 16, 18, 20])
            for row in rows:
                self.assertNotEqual(row['status'], 'failed')
                self.assertEqual(int(row['k']), k)
                self.assertEqual(row['recall_at_{}_percent'.format(k)], '1')
                if k == 10:
                    self.assertNotIn('recall_at_1_percent', row)
                self.assertEqual(int(row['rounds']), 3)
                self.assertEqual(int(row['budget_bytes']), 20 * 2**30)
                self.assertEqual(float(row['cache_read_seconds']), 0.1)
                if row['codec'] not in ('raw', 'alp'):
                    self.assertEqual(int(row['propagation_rounds']), 8)
                    self.assertEqual(float(row['fallback_root_search_seconds']), 0.01)
            second = subprocess.run(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertEqual(len((root / 'apps/calls.jsonl').read_text().splitlines()), 9)
            if k == 10:
                stamp = root / 'out/raw.done.json'
                signature = json.loads(stamp.read_text())
                self.assertEqual(signature['k'], 10)
                signature['k'] = 1
                stamp.write_text(json.dumps(signature))
                third = subprocess.run(command, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
                self.assertEqual(third.returncode, 0, third.stdout)
                self.assertEqual(len((root / 'apps/calls.jsonl').read_text().splitlines()), 10)


if __name__ == '__main__':
    unittest.main()
