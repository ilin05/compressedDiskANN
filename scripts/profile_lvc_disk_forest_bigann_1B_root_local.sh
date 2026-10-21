#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
out=${OUTPUT_DIR:-"$repo/scripts/lvc_disk_forest_profile_results/bigann_1B_1GiB_root_local"}
lvc_build_threads=${LVC_BUILD_THREADS:-32}
if [[ ${SKIP_BUILD:-0} != 1 ]]; then
    cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target search_disk_index --parallel "${BUILD_THREADS:-4}"
fi
export OMP_NUM_THREADS="$lvc_build_threads" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 "$repo/scripts/profile_lvc_disk_forest.py" \
    --binary "$build/apps/search_disk_index" \
    --index-prefix "${INDEX_PREFIX:-$data/disk_index_bigann_1B_R32_L50_A1.2}" \
    --query "${QUERY_FILE:-$data/bigann_query.fbin}" --output-dir "$out" \
    --expected-points 1000000000 --sample-nodes "${SAMPLE_NODES:-1000000}" \
    --build-threads "$lvc_build_threads" --chains -1 --fallback-root-candidates 1 \
    --budget-mib "${BUDGET_MIB:-1024}" --planning-budget-mib "${PLANNING_BUDGET_MIB:-998.4}"
