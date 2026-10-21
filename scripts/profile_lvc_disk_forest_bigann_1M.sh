#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
out=${OUTPUT_DIR:-"$repo/scripts/lvc_disk_forest_profile_results/bigann_1M"}
if [[ ${SKIP_BUILD:-0} != 1 ]]; then
    cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target search_disk_index --parallel "${BUILD_THREADS:-4}"
fi
# Build parallelism above is unrelated to the measured single-threaded runs.
export OMP_NUM_THREADS=${LVC_BUILD_THREADS:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 "$repo/scripts/profile_lvc_disk_forest.py" \
    --binary "$build/apps/search_disk_index" \
    --index-prefix "${INDEX_PREFIX:-$data/disk_index_bigann_1M_R32_L50_A1.2}" \
    --query "${QUERY_FILE:-$data/bigann_query.fbin}" --output-dir "$out" \
    --sample-nodes "${SAMPLE_NODES:-1000}" --build-threads "${LVC_BUILD_THREADS:-1}" \
    --budget-mib "${BUDGET_MIB:-20}" --planning-budget-mib "${PLANNING_BUDGET_MIB:-19.5}"
