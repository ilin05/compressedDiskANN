#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
out=${OUTPUT_DIR:-"$repo/scripts/lvc_disk_payload_budget_results/bigann_1B_dexor_unbounded_4GiB"}
lvc_build_threads=${LVC_BUILD_THREADS:-32}
if [[ ${SKIP_BUILD:-0} != 1 ]]; then
    cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target search_disk_index --parallel "${BUILD_THREADS:-4}"
fi
export OMP_NUM_THREADS="$lvc_build_threads" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
extra=()
[[ ${VERIFY_CACHE:-0} == 1 ]] || extra+=(--skip-verify-cache)
python3 "$repo/scripts/profile_lvc_disk_forest.py" \
    --binary "$build/apps/search_disk_index" \
    --index-prefix "${INDEX_PREFIX:-$data/disk_index_bigann_1B_R32_L50_A1.2}" \
    --query "${QUERY_FILE:-$data/bigann_query.fbin}" \
    --gt "${GT_FILE:-$data/bigann_1B_gt.bin}" --output-dir "$out" \
    --expected-points 1000000000 --sample-nodes "${SAMPLE_NODES:-1000000}" \
    --build-threads "$lvc_build_threads" --chains -1 --fallback-root-candidates 1 \
    --budget-mib "${BUDGET_MIB:-4096}" --planning-budget-mib "${PLANNING_BUDGET_MIB:-3993.6}" \
    --search --L 1 5 10 --rounds 3 --beamwidth "${PERF_W:-2}" "${extra[@]}"
