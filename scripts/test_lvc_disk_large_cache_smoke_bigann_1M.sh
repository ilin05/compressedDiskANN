#!/usr/bin/env bash
# Small native test of the explicit-case launcher, footprint counters and payloads.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
export BUILD_DIR="$build" DATA_DIR="$data"
export OUTPUT_DIR="${OUTPUT_DIR:-$repo/scripts/lvc_large_cache_results/smoke_1M_$(date -u +%Y%m%dT%H%M%SZ)}"
export INDEX_PREFIX="${INDEX_PREFIX:-$data/disk_index_bigann_1M_R32_L50_A1.2}"
export GT_FILE="${GT_FILE:-$data/bigann_1M_gt.bin}" EXPECTED_POINTS=1000000
export CASES="dexor:1024 alp:2048 alp:4096" NUM_ROUNDS=2 L_VALUES="1 2" BEAMWIDTH=1 VERIFY_CACHE=1
bash "$repo/scripts/test_lvc_disk_large_cache_bigann_1B.sh"
[[ ${DRY_RUN:-0} != 1 ]] || exit 0
python3 - "$OUTPUT_DIR" <<'PY'
import csv
import json
import os
from pathlib import Path
import sys
out = Path(sys.argv[1])
with (out / 'footprint.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
assert [(r['codec'], int(r['nodes'])) for r in rows] == [('dexor', 1024), ('alp', 2048), ('alp', 4096)]
for row in rows:
    assert row['status'] == row['footprint_status'] == 'ok', row
    total = sum(int(row[k]) for k in ('adjacency_bytes', 'payload_bytes', 'record_metadata_bytes', 'state_cache_bytes'))
    assert total == int(row['cached_data_bytes'])
    assert total + int(row['lookup_bytes']) == int(row['cache_bytes'])
with (out / 'summary.csv').open(newline='') as stream:
    searches = list(csv.DictReader(stream))
assert len(searches) == 6
assert all(r['status'] == 'ok' and r['measured_rounds'] == '1' for r in searches)
if os.environ.get('MEMORY_MONITOR') == '1':
    for row in rows:
        case = out / row['case']
        done = json.loads((case / 'done.json').read_text())
        attempt = case / ('attempt_' + str(done['rows'][0]['attempt']))
        samples = list(csv.DictReader((attempt / 'memory_samples.csv').open()))
        assert samples and all(s['pid'] for s in samples)
        assert len(list(csv.DictReader((attempt / 'memory_rounds.csv').open()))) == 4
        assert json.loads((attempt / 'memory_summary.json').read_text())['memory_samples'] == len(samples)
    print('PASS process memory/swap samples and per-round diagnostic outputs')
print('PASS three explicit cases, resident-footprint sums, native payload verification, complete query rounds')
print('Results:', out)
PY
