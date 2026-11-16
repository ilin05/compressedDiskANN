#!/usr/bin/env bash
# Verify measurement controls do not change real BIGANN-1M search results.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
out=${OUTPUT_DIR:-"$repo/scripts/lvc_disk_dexor_stability_results/smoke_1M_$(date -u +%Y%m%dT%H%M%SZ)"}
export BUILD_DIR="$build" DATA_DIR="$data"
export INDEX_PREFIX="${INDEX_PREFIX:-$data/disk_index_bigann_1M_R32_L50_A1.2}"
export GT_FILE="${GT_FILE:-$data/bigann_1M_gt.bin}"
export CACHE_NODES_LIST="${CACHE_NODES:-4096}" L_VALUES="1 4 8" ROUNDS=2 WARMUP_ROUNDS=1
# One in-flight read avoids completion-order differences in the bitwise A/B check.
export BEAMWIDTH=1
for profile in 0 1; do
    OUTPUT_DIR="$out/profile${profile}" LVC_DECODE_PROFILE="$profile" \
        bash "$repo/scripts/test_lvc_disk_dexor_stability_bigann_1B.sh"
done
python3 - "$out" "$CACHE_NODES_LIST" <<'PY'
import csv, sys
from pathlib import Path
out, nodes = Path(sys.argv[1]), sys.argv[2]
cases = [out/('profile'+str(p))/('dexor_C'+nodes) for p in (0, 1)]
files = sorted(cases[0].glob('search*.bin'))
assert len(files) == 24, len(files)  # 3 L * (3 passes + 1 compatibility pair) * 2
for file in files:
    assert file.read_bytes() == (cases[1]/file.name).read_bytes(), file.name
for profile, case in enumerate(cases):
    with (case/'rounds.csv').open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 9
    assert sum(row['phase'] == 'warmup' for row in rows) == 3
    assert all(int(row['lvc_decode_profile']) == profile for row in rows)
    calls = sum(int(row['decode_calls']) for row in rows)
    assert (calls == 0 if profile == 0 else calls > 0), (profile, calls)
    with (case/'summary.csv').open() as source:
        summary = list(csv.DictReader(source))
    assert all(row['complete'] == 'True' and row['completed_rounds'] == '2' for row in summary)
    for row in summary:
        measured = [r for r in rows if r['L'] == row['L'] and r['phase'] == 'measurement']
        expected = sum(float(r['mean_us']) for r in measured)/2
        assert abs(float(row['mean_us']) - expected) < 1e-6
print('PASS profile on/off: bitwise results, warmup exclusion and decode counters')
print('Results:', out)
PY
