#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
index=${INDEX_PREFIX:-"$build/data/bigann/disk_index_bigann_1B_R32_L50_A1.2"}
out=${OUTPUT_DIR:-"$repo/scripts/cache_lookup_results/bigann_1B"}
maximum=${MAX_NODES:-500000000}
limit=${LIMIT_BYTES:-34359738368}
mkdir -p "$out"
if [[ ${SKIP_BUILD:-0} != 1 ]]; then
    cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target test_cache_lookup search_disk_index --parallel "${BUILD_THREADS:-4}"
fi
"$build/apps/test_cache_lookup" --index-prefix "$index" \
    --ids-file "${IDS_FILE:-$out/cache_ids_${maximum}.bin}" \
    --max-nodes "$maximum" --csv "$out/lookup_curve.csv" \
    --limit-bytes "$limit" 2>&1 | tee "$out/probe.log"
python3 "$repo/scripts/lookup_capacity_curve.py" --curve "$out/lookup_curve.csv" \
    --limit-bytes "$limit" --require-limit-exceeded | tee "$out/summary.json"
echo "BIGANN1B lookup growth threshold recorded; results: $out"
