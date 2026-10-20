#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
index=${INDEX_PREFIX:-"$data/disk_index_bigann_1M_R32_L50_A1.2"}
out=${OUTPUT_DIR:-"$repo/scripts/lvc_disk_payload_budget_results/bigann_1M"}
if [[ ${SKIP_BUILD:-0} != 1 ]]; then
    cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target search_disk_index --parallel "${BUILD_THREADS:-4}"
fi
extra=()
[[ ${RESUME:-1} != 1 ]] || extra+=(--resume)
python3 "$repo/scripts/lvc_disk_payload_budget.py" \
    --binary "$build/apps/search_disk_index" --index-prefix "$index" \
    --query "${QUERY_FILE:-$data/bigann_query.fbin}" --gt "${GT_FILE:-$data/bigann_1M_gt.bin}" \
    --output-dir "$out" --expected-points 1000000 --sample-nodes "${SAMPLE_NODES:-1000}" \
    --budget-mib "${BUDGET_MIB:-20}" --planning-budget-mib "${PLANNING_BUDGET_MIB:-19.5}" \
    --threads 1 --L 1 2 3 4 5 6 8 10 --rounds "${ROUNDS:-10}" --beamwidth 1 \
    --state-cache-ratio "${STATE_CACHE_RATIO:-0.01}" --verify-cache --check-results "${extra[@]}"
