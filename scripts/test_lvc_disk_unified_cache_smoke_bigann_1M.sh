#!/usr/bin/env bash
# Verify all five modes with parallel loading before running the 1B matrix.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
out=${OUTPUT_DIR:-"$repo/scripts/lvc_unified_cache_results/smoke_1M_$(date -u +%Y%m%dT%H%M%SZ)"}
nodes=${CACHE_NODES:-1024}
export BUILD_DIR="$build" DATA_DIR="$data" OUTPUT_DIR="$out"
export INDEX_PREFIX="${INDEX_PREFIX:-$data/disk_index_bigann_1M_R32_L50_A1.2}"
export GT_FILE="${GT_FILE:-$data/bigann_1M_gt.bin}"
export EXPECTED_POINTS=1000000 CACHE_NODES="$nodes" NUM_ROUNDS=2 L_VALUES=10 BEAMWIDTH=1 VERIFY_CACHE=1
bash "$repo/scripts/test_lvc_disk_unified_cache_bigann_1B.sh"
python3 - "$out" "$nodes" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1]); nodes=sys.argv[2]
raw=out/('raw_'+nodes)/'attempt_1'
for codec in ('dexor','gorilla','elf'):
    target=out/(codec+'_'+nodes)/'attempt_1'
    for suffix in ('idx_uint32.bin','dists_float.bin'):
        name='search_L10_R2_'+suffix
        a=(raw/name).read_bytes(); b=(target/name).read_bytes()
        if a!=b:
            offset=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
            raise SystemExit('FAIL %s %s byte=%d query=%d'%(codec,suffix,offset,max(0,(offset-8)//4)))
        print('PASS',codec,suffix,'exactly matches raw')
print('PASS ALP per-vector serial/parallel byte check and reconstruction tolerance were verified by the runner')
print('Results:',out)
PY
