#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${BUILD_DIR:-$repo/build}"
data="${DATA_DIR:-$build/data/bigann}"
index="${INDEX_PREFIX:-$data/disk_index_bigann_1B_R32_L50_A1.2}"
query="${QUERY_FILE:-$data/bigann_query.fbin}"
gt="${GT_FILE:-$data/bigann_1B_gt.bin}"
out="${OUTPUT_DIR:-$repo/scripts/alp_only_results/bigann_1B}"
for file in "${index}_disk.index" "${index}_pq_compressed.bin" "$query" "$gt"; do
    [[ -s "$file" ]] || { echo "Missing input: $file" >&2; exit 1; }
done
if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    [[ -f "$build/CMakeCache.txt" ]] || cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target search_disk_index --parallel "${BUILD_THREADS:-4}"
fi
mkdir -p "$out"
extra=()
[[ "${VERIFY_CACHE:-0}" != 1 ]] || extra+=(--verify_cache)
for k in 1 10; do
    if [[ "$k" == 1 ]]; then levels=(1 2 3 4 5 6 8 10); else levels=(10 11 12 13 14 16 18 20); fi
    echo "Running ALP-only BIGANN1B K=$k cache_nodes=${CACHE_NODES:-400000000}"
    "$build/apps/search_disk_index" \
        --data_type float --dist_fn l2 --index_path_prefix "$index" \
        --query_file "$query" --gt_file "$gt" --cache_codec alp \
        -K "$k" -L "${levels[@]}" -T "${THREADS:-1}" -W "${BEAM_WIDTH:-2}" -R "${ROUNDS:-100}" \
        --result_path "$out/alp_K${k}" --num_nodes_to_cache "${CACHE_NODES:-400000000}" \
        "${extra[@]}" > "$out/alp_K${k}.log" 2>&1
done
echo "ALP-only results: $out"
