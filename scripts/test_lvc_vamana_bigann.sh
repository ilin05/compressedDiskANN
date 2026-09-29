#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$DISKANN_DIR/build}"
DATASET="${DATASET:-$BUILD_DIR/data/bigann/bigann_1M.fbin}"
QUERY_FILE="${QUERY_FILE:-$BUILD_DIR/data/bigann/bigann_query.fbin}"
GRAPH_PREFIX="${GRAPH_PREFIX:-$BUILD_DIR/data/bigann/memory_index_bigann_1M_R32_L50_A1.2}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$BUILD_DIR/data/bigann/lvc_d2_vamana_1M}"
GT_FILE="${GT_FILE:-null}"
NUM_THREADS="${NUM_THREADS:-1}"
SEARCH_L="${SEARCH_L:-50}"
K="${K:-10}"
ROUNDS="${ROUNDS:-3}"
CXX="${CXX:-g++}"

mkdir -p "$BUILD_DIR/apps" "$(dirname "$OUTPUT_PREFIX")"
if [[ ! -f "$DATASET" || ! -f "$QUERY_FILE" ]]; then
  echo "Missing DATASET or QUERY_FILE" >&2
  exit 1
fi
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S "$DISKANN_DIR" -B "$BUILD_DIR"
fi
cmake --build "$BUILD_DIR" --target build_memory_index search_memory_index --parallel 4

if [[ ! -f "$GRAPH_PREFIX" ]]; then
  "$BUILD_DIR/apps/build_memory_index" --data_type float --dist_fn l2 \
    --data_path "$DATASET" --index_path_prefix "$GRAPH_PREFIX" \
    -R 32 -L 50 -T "$NUM_THREADS" --alpha 1.2 --build_PQ_bytes 0
fi

for app in convert_lvc_vamana verify_lvc_vamana; do
  "$CXX" -std=c++17 -O2 -I "$DISKANN_DIR/include" \
    "$DISKANN_DIR/apps/$app.cpp" \
    "$DISKANN_DIR/include/lvc_codec/encoding_algorithms/elf/elf64_utils.cpp" \
    -o "$BUILD_DIR/apps/$app"
done

"$BUILD_DIR/apps/convert_lvc_vamana" "$GRAPH_PREFIX" "$DATASET" "$OUTPUT_PREFIX"

for codec in raw dexor gorilla elf; do
  prefix="${OUTPUT_PREFIX}_${codec}"
  "$BUILD_DIR/apps/verify_lvc_vamana" "$DATASET" "${prefix}.data"
  "$BUILD_DIR/apps/search_memory_index" \
    --data_type float --dist_fn l2 --index_path_prefix "$prefix" \
    --query_file "$QUERY_FILE" --gt_file "$GT_FILE" \
    -K "$K" -L "$SEARCH_L" -T "$NUM_THREADS" -R "$ROUNDS" \
    --use_pq_dist 0 --pq_exact_rerank_ratio 0 \
    --result_path "${prefix}_results" > "${prefix}_search.log" 2>&1
  tail -n 5 "${prefix}_search.log"
done

python3 - "$OUTPUT_PREFIX" "$SEARCH_L" "$K" "$NUM_THREADS" <<'PY'
import csv
import pathlib
import struct
import sys

prefix, search_l, k, threads = sys.argv[1:]
raw_prefix = pathlib.Path(prefix + '_raw')
raw_ids = (raw_prefix.parent / f'{raw_prefix.name}_results_{search_l}_idx_uint32.bin').read_bytes()
raw_dist = (raw_prefix.parent / f'{raw_prefix.name}_results_{search_l}_dists_float.bin').read_bytes()
for codec in ('raw', 'dexor', 'gorilla', 'elf'):
    p = pathlib.Path(prefix + '_' + codec)
    ids = (p.parent / f'{p.name}_results_{search_l}_idx_uint32.bin').read_bytes()
    dist = (p.parent / f'{p.name}_results_{search_l}_dists_float.bin').read_bytes()
    if ids != raw_ids or dist != raw_dist:
        kind, actual, expected = ('IDs', ids, raw_ids) if ids != raw_ids else ('L2 bits', dist, raw_dist)
        first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                     min(len(actual), len(expected)))
        if first < 8 or first + 4 > min(len(actual), len(expected)):
            raise SystemExit(f'FAIL {codec}: {kind} differ from raw at byte {first}')
        item = (first - 8) // 4
        fmt = '<I' if kind == 'IDs' else '<f'
        position = 8 + item * 4
        observed = struct.unpack_from(fmt, actual, position)[0]
        reference = struct.unpack_from(fmt, expected, position)[0]
        raise SystemExit(f'FAIL {codec}: {kind} query={item // int(k)} rank={item % int(k)} '
                         f'raw={reference} decoded={observed}')
    csv_path = pathlib.Path(f'{p}_K{k}_T{threads}_search_result.csv')
    with csv_path.open(newline='') as file:
        row = next(csv.DictReader(file))
    print(f'PASS {codec}: exact same IDs and L2 bits; '
          f'QPS={row["QPS"]}, mean_us={row["Mean Latency (mus)"]}, '
          f'p999_us={row["99.9 Latency"]}, '
          f'graph_bytes={p.stat().st_size}, data_bytes={pathlib.Path(str(p) + ".data").stat().st_size}')
PY
