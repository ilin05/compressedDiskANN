#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
out=${OUTPUT_DIR:-"$repo/scripts/lvc_unified_cache_results/bigann_1B_$(date -u +%Y%m%dT%H%M%SZ)"}
read -r -a nodes <<< "${CACHE_NODES:-10000000 25000000 50000000}"
read -r -a codecs <<< "${CODECS:-raw alp dexor gorilla elf}"
read -r -a levels <<< "${L_VALUES:-1 2 3 4 5 6 8 10}"
extra=()
[[ ${RESUME:-0} != 1 ]] || extra+=(--resume)
[[ ${VERIFY_CACHE:-0} != 1 ]] || extra+=(--verify-cache)
[[ ${DRY_RUN:-0} != 1 ]] || extra+=(--dry-run)
if [[ ${SKIP_BUILD:-0} != 1 && ${DRY_RUN:-0} != 1 ]]; then
    cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target search_disk_index --parallel "${BUILD_THREADS:-4}"
fi
export OMP_NUM_THREADS="${LOAD_THREADS:-32}" OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python3 "$repo/scripts/lvc_disk_unified_cache.py" \
    --binary "$build/apps/search_disk_index" --index "${INDEX_PREFIX:-$data/disk_index_bigann_1B_R32_L50_A1.2}" \
    --query "${QUERY_FILE:-$data/bigann_query.fbin}" --gt "${GT_FILE:-$data/bigann_1B_gt.bin}" \
    --output "$out" --expected-points "${EXPECTED_POINTS:-1000000000}" \
    --nodes "${nodes[@]}" --codecs "${codecs[@]}" --L "${levels[@]}" \
    --load-threads "${LOAD_THREADS:-32}" --rounds "${NUM_ROUNDS:-10}" --beamwidth "${BEAMWIDTH:-2}" "${extra[@]}"
