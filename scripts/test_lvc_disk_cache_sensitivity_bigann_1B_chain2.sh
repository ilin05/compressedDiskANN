#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
index=${INDEX_PREFIX:-"$data/disk_index_bigann_1B_R32_L50_A1.2"}
out=${OUTPUT_DIR:-"$repo/scripts/lvc_disk_cache_sensitivity_results/bigann_1B_chain2"}
load_threads=${LVC_BUILD_THREADS:-32}

# In this order: 10M x3, 25M x3, Gorilla 50M, DeXOR/Elf 100M. Serial execution.
read -r -a cases <<< "${CASES:-dexor:10000000 gorilla:10000000 elf:10000000 dexor:25000000 gorilla:25000000 elf:25000000 gorilla:50000000 dexor:100000000 elf:100000000}"
read -r -a levels <<< "${L_VALUES:-1 2 3 4 5 6 8 10}"

if [[ ${SKIP_BUILD:-0} != 1 && ${DRY_RUN:-0} != 1 ]]; then
    cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target search_disk_index --parallel "${BUILD_THREADS:-4}"
fi
export OMP_NUM_THREADS="$load_threads" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
extra=()
[[ ${RESUME:-1} != 1 ]] || extra+=(--resume)
[[ ${VERIFY_CACHE:-0} != 1 ]] || extra+=(--verify-cache)
[[ ${DRY_RUN:-0} != 1 ]] || extra+=(--dry-run)

python3 "$repo/scripts/lvc_disk_cache_sensitivity.py" \
    --binary "$build/apps/search_disk_index" --index-prefix "$index" \
    --query "${QUERY_FILE:-$data/bigann_query.fbin}" --gt "${GT_FILE:-$data/bigann_1B_gt.bin}" \
    --output-dir "$out" --expected-points "${EXPECTED_POINTS:-1000000000}" \
    --cases "${cases[@]}" --build-threads "$load_threads" \
    --L "${levels[@]}" --rounds "${NUM_ROUNDS:-3}" --beamwidth "${BEAMWIDTH:-2}" \
    --state-cache-ratio "${STATE_CACHE_RATIO:-0.01}" "${extra[@]}"
