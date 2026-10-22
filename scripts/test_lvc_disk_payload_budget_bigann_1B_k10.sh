#!/usr/bin/env bash
# Same payload budget and load configuration as the K=1 experiment.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
index=${INDEX_PREFIX:-"$data/disk_index_bigann_1B_R32_L50_A1.2"}
out=${OUTPUT_DIR:-"$repo/scripts/lvc_disk_payload_budget_results/bigann_1B_k10"}
lvc_build_threads=${LVC_BUILD_THREADS:-32}
if [[ ${SKIP_BUILD:-0} != 1 ]]; then
    cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target search_disk_index --parallel "${BUILD_THREADS:-4}"
fi
export OMP_NUM_THREADS="$lvc_build_threads" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
extra=()
[[ ${RESUME:-1} != 1 ]] || extra+=(--resume)
[[ ${VERIFY_CACHE:-0} != 1 ]] || extra+=(--verify-cache)
python3 "$repo/scripts/lvc_disk_payload_budget.py" \
    --binary "$build/apps/search_disk_index" --index-prefix "$index" \
    --query "${QUERY_FILE:-$data/bigann_query.fbin}" --gt "${GT_FILE:-$data/bigann_1B_gt.bin}" \
    --output-dir "$out" --expected-points 1000000000 --sample-nodes "${SAMPLE_NODES:-1000000}" \
    --budget-mib "${BUDGET_MIB:-20480}" --planning-budget-mib "${PLANNING_BUDGET_MIB:-19968}" \
    --threads 1 --build-threads "$lvc_build_threads" --fallback-root-candidates 1 --require-phase-timings \
    --k 10 --L 10 11 12 13 14 16 18 20 --rounds "${ROUNDS:-3}" --beamwidth "${PERF_W:-2}" \
    --state-cache-ratio "${STATE_CACHE_RATIO:-0.01}" "${extra[@]}"
