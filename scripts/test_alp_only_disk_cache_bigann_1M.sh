#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${BUILD_DIR:-$repo/build}"
data="${DATA_DIR:-$build/data/bigann}"
index="${INDEX_PREFIX:-$data/disk_index_bigann_1M_R32_L50_A1.2}"
query="${QUERY_FILE:-$data/bigann_query.fbin}"
gt="${GT_FILE:-$data/bigann_1M_gt.bin}"
out="${OUTPUT_DIR:-$repo/scripts/alp_only_results/bigann_1M}"
for file in "${index}_disk.index" "${index}_pq_compressed.bin" "$query" "$gt"; do
    [[ -s "$file" ]] || { echo "Missing input: $file" >&2; exit 1; }
done
if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    [[ -f "$build/CMakeCache.txt" ]] || cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target search_disk_index --parallel "${BUILD_THREADS:-4}"
fi
read -r -a threads <<< "${THREAD_COUNTS:-1}"
read -r -a beams <<< "${BEAM_WIDTHS:-1}"
read -r -a ks <<< "${K_VALUES:-1}"
extra=()
[[ "${CHECK_CAPS:-0}" != 1 ]] || extra+=(--cap-check)
python3 "$repo/scripts/test_alp_only_disk_cache.py" \
    --binary "$build/apps/search_disk_index" --index-prefix "$index" \
    --query "$query" --gt "$gt" --output "$out" \
    --cache-nodes "${CACHE_NODES:-10000}" --rounds "${ROUNDS:-1}" \
    --threads "${threads[@]}" --beamwidths "${beams[@]}" --k "${ks[@]}" "${extra[@]}"
