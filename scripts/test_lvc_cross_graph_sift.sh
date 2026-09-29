#!/usr/bin/env bash
set -euo pipefail

# SIFT pilot: the same fvecs/ivecs inputs feed HNSW and Vamana.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "${HNSW_DIR:-}" ]]; then
  if [[ -d "$DISKANN_DIR/../hnswlib_cpp_py" ]]; then
    HNSW_DIR="$(cd "$DISKANN_DIR/../hnswlib_cpp_py" && pwd)"
  else
    HNSW_DIR="$HOME/HNSWCW/compressedhnsw"
  fi
fi
if [[ ! -f "$HNSW_DIR/CMakeLists.txt" ]]; then
  echo "HNSW_DIR is not a CMake project: $HNSW_DIR" >&2
  exit 1
fi
HNSW_DATA_DIR="${HNSW_DATA_DIR:-$HNSW_DIR/datasets/hdf5files}"
DATASET="${DATASET:-sift-128-euclidean}"
OUTPUT_DIR="${OUTPUT_DIR:-$DISKANN_DIR/build/data/sift_cross_graph}"
DISKANN_BUILD_DIR="${DISKANN_BUILD_DIR:-$DISKANN_DIR/build}"
HNSW_BUILD_DIR="${HNSW_BUILD_DIR:-$HNSW_DIR/build_lvc_sift}"
BUILD_THREADS="${BUILD_THREADS:-4}"
SEARCH_THREADS="${SEARCH_THREADS:-1}"
ROUNDS="${ROUNDS:-3}"
CXX="${CXX:-g++}"

for kind in train.fvecs test.fvecs neighbors.ivecs; do
  file="$HNSW_DATA_DIR/${DATASET}_$kind"
  if [[ ! -f "$file" ]]; then
    echo "Missing SIFT input: $file" >&2
    exit 1
  fi
done
mkdir -p "$OUTPUT_DIR" "$DISKANN_BUILD_DIR/apps"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
HNSW_DATA_DIR="$(cd "$HNSW_DATA_DIR" && pwd)"

# Check source dimensions/counts and row headers before using the existing converters.
python3 - "$HNSW_DATA_DIR" "$DATASET" <<'PY'
import pathlib, struct, sys
base, name = pathlib.Path(sys.argv[1]), sys.argv[2]
shapes = {}
for suffix in ('train.fvecs', 'test.fvecs', 'neighbors.ivecs'):
    path = base / f'{name}_{suffix}'
    with path.open('rb') as f:
        header = f.read(4)
        if len(header) != 4:
            raise SystemExit(f'Empty input: {path}')
        dim, = struct.unpack('<i', header)
        if dim <= 0 or (path.stat().st_size % (4 + 4 * dim)):
            raise SystemExit(f'Invalid row width: {path}')
        count = path.stat().st_size // (4 + 4 * dim)
        f.seek(0)
        for row in range(count):
            actual, = struct.unpack('<i', f.read(4))
            if actual != dim:
                raise SystemExit(f'Inconsistent dimension: {path} row={row} dim={actual}')
            f.seek(4 * dim, 1)
    shapes[suffix] = (count, dim)
if shapes['train.fvecs'][1] != shapes['test.fvecs'][1]:
    raise SystemExit('Train/query dimensions differ')
if shapes['test.fvecs'][0] != shapes['neighbors.ivecs'][0]:
    raise SystemExit('Query/ground-truth counts differ')
if shapes['neighbors.ivecs'][1] < 10:
    raise SystemExit('Ground truth needs at least 10 neighbors')
print('SIFT input shapes:', shapes)
PY

if [[ ! -f "$DISKANN_BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S "$DISKANN_DIR" -B "$DISKANN_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "$DISKANN_BUILD_DIR" \
  --target fvecs_to_bin ivecs_to_bin build_memory_index search_memory_index \
  --parallel "$BUILD_THREADS"
if [[ ! -f "$HNSW_BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S "$HNSW_DIR" -B "$HNSW_BUILD_DIR" -DHNSWLIB_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "$HNSW_BUILD_DIR" \
  --target test_build_compressed_hnsw_ablation test_search_compressed_hnsw_ablation \
  --parallel "$BUILD_THREADS"

echo "Converting SIFT inputs to DiskANN bin format"
train="$OUTPUT_DIR/${DATASET}_train.fbin"
query="$OUTPUT_DIR/${DATASET}_test.fbin"
gt="$OUTPUT_DIR/${DATASET}_neighbors.bin"
"$DISKANN_BUILD_DIR/apps/utils/fvecs_to_bin" float "$HNSW_DATA_DIR/${DATASET}_train.fvecs" "$train"
"$DISKANN_BUILD_DIR/apps/utils/fvecs_to_bin" float "$HNSW_DATA_DIR/${DATASET}_test.fvecs" "$query"
"$DISKANN_BUILD_DIR/apps/utils/ivecs_to_bin" "$HNSW_DATA_DIR/${DATASET}_neighbors.ivecs" "$gt"
python3 - "$HNSW_DATA_DIR" "$DATASET" "$train" "$query" "$gt" <<'PY'
import pathlib, struct, sys
base, name = pathlib.Path(sys.argv[1]), sys.argv[2]
for suffix, target in zip(('train.fvecs', 'test.fvecs', 'neighbors.ivecs'), map(pathlib.Path, sys.argv[3:])):
    source = base / f'{name}_{suffix}'
    with target.open('rb') as converted:
        n, d = struct.unpack('<II', converted.read(8))
    if target.stat().st_size != 8 + n * d * 4:
        raise SystemExit(f'Invalid converted file: {target}')
    with source.open('rb') as f, target.open('rb') as converted:
        for row in (0, n // 2, n - 1):
            f.seek(row * (4 + 4 * d) + 4)
            expected = f.read(4 * d)
            converted.seek(8 + row * 4 * d)
            actual = converted.read(4 * d)
            if actual != expected:
                raise SystemExit(f'Conversion mismatch: {target} row={row}')
    print(f'Converted {suffix}: n={n} dim={d} bytes={target.stat().st_size}')
PY

# HNSW ablation uses F64 vectors, M=16, efConstruction=200, 1% roots by level,
# chain_max=2, 1% decoding cache, and TLS ratio 0.2.
echo "Building and searching HNSW indexes"
(
  cd "$OUTPUT_DIR"
  "$HNSW_BUILD_DIR/test_build_compressed_hnsw_ablation" \
    --base_dir "$HNSW_DATA_DIR" --dataset "$DATASET" \
    --algorithm DeXOR Gorilla Elf --chain_max 2 --threads "$BUILD_THREADS" \
    --M 16 --ef_construction 200 --output_csv hnsw_build.csv \
    > hnsw_build.log 2>&1
  for k in 1 10; do
    "$HNSW_BUILD_DIR/test_search_compressed_hnsw_ablation" \
      --base_dir "$HNSW_DATA_DIR" --dataset "$DATASET" \
      --algorithm DeXOR Gorilla Elf --chain_max 2 --threads "$SEARCH_THREADS" \
      --k "$k" --ef 20 50 100 --use_cache 1 --use_tls 1 --tls_ratio 0.2 \
      --num_rounds "$ROUNDS" --output_csv "hnsw_search_k${k}.csv" \
      > "hnsw_search_k${k}.log" 2>&1
  done
)

# Vamana builds PQ along with the graph. PQDataStore writes the filenames
# expected by search_memory_index, so generate_pq is not used here.
echo "Building Vamana graph and PQ codes"
graph="$OUTPUT_DIR/vamana_R32_L50_A1.2"
"$DISKANN_BUILD_DIR/apps/build_memory_index" \
  --data_type float --dist_fn l2 --data_path "$train" --index_path_prefix "$graph" \
  -R 32 -L 50 -T "$BUILD_THREADS" --alpha 1.2 --build_PQ_bytes 16 \
  > "$OUTPUT_DIR/vamana_build.log" 2>&1
for suffix in pq16_pivots.bin pq16_compressed.bin; do
  [[ -s "${train}${suffix}" ]] || { echo "Missing PQ file: ${train}${suffix}" >&2; exit 1; }
done

for app in convert_lvc_vamana verify_lvc_vamana; do
  "$CXX" -std=c++17 -O2 -I "$DISKANN_DIR/include" \
    "$DISKANN_DIR/apps/$app.cpp" \
    "$DISKANN_DIR/include/lvc_codec/encoding_algorithms/elf/elf64_utils.cpp" \
    -o "$DISKANN_BUILD_DIR/apps/$app"
done
prefix="$OUTPUT_DIR/vamana_lvc"
echo "Compressing and searching Vamana indexes"
"$DISKANN_BUILD_DIR/apps/convert_lvc_vamana" "$graph" "$train" "$prefix" \
  | tee "$OUTPUT_DIR/vamana_compression.log"
for codec in raw dexor gorilla elf; do
  "$DISKANN_BUILD_DIR/apps/verify_lvc_vamana" "$train" "${prefix}_${codec}.data" \
    > "$OUTPUT_DIR/vamana_verify_${codec}.log" 2>&1
  for k in 1 10; do
    "$DISKANN_BUILD_DIR/apps/search_memory_index" \
      --data_type float --dist_fn l2 --index_path_prefix "${prefix}_${codec}" \
      --query_file "$query" --gt_file "$gt" \
      -K "$k" -L 20 50 100 -T "$SEARCH_THREADS" -R "$ROUNDS" \
      --use_pq_dist 1 --pq_bytes 16 --pq_prefix "$train" \
      --pq_exact_rerank_ratio 0.2 --lvc_state_cache_ratio 0.01 \
      --result_path "$OUTPUT_DIR/vamana_${codec}_k${k}" \
      > "$OUTPUT_DIR/vamana_${codec}_k${k}.log" 2>&1
    cp "${prefix}_${codec}_K${k}_T${SEARCH_THREADS}_search_result.csv" \
      "$OUTPUT_DIR/vamana_${codec}_k${k}.csv"
  done
done

python3 - "$OUTPUT_DIR" <<'PY'
import csv, pathlib, re, sys
out = pathlib.Path(sys.argv[1])
for k in (1, 10):
    for L in (20, 50, 100):
        for kind in ('idx_uint32.bin', 'dists_float.bin'):
            raw = (out / f'vamana_raw_k{k}_{L}_{kind}').read_bytes()
            for codec in ('dexor', 'gorilla', 'elf'):
                candidate = (out / f'vamana_{codec}_k{k}_{L}_{kind}').read_bytes()
                if candidate != raw:
                    first = next((i for i, pair in enumerate(zip(candidate, raw))
                                  if pair[0] != pair[1]), min(len(candidate), len(raw)))
                    raise SystemExit(f'FAIL Vamana {codec} K={k} L={L} {kind} byte={first}')
print('PASS Vamana: all codec IDs and L2 bits match raw for every K/L')
print('\nHNSW compression (baseline: F64):')
for row in csv.DictReader((out / 'hnsw_build.csv').open()):
    print(row['Algorithm'], 'data_ratio=', row['DataCompressionRatio'],
          'index_ratio=', row['IndexCompressionRatio'],
          'build_s=', row['BuildTime(s)'], 'compress_s=', row['CompressTime(s)'])
print('\nVamana compression (baseline: F32):')
for line in (out / 'vamana_compression.log').read_text().splitlines():
    if re.match(r'^(forest|raw|dexor|gorilla|elf)\b', line):
        print(line)
rows = []
for k in (1, 10):
    for row in csv.DictReader((out / f'hnsw_search_k{k}.csv').open()):
        rows.append(dict(graph='HNSW', codec=row['Algorithm'], k=k,
                         search_width=row['ef'], recall=row['Recall'], qps=row['QPS'],
                         mean_us=row['TimePerQuery(us)'], p99_us=row['P99_Latency(us)'],
                         metric='p99'))
    for codec in ('raw', 'dexor', 'gorilla', 'elf'):
        for row in csv.DictReader((out / f'vamana_{codec}_k{k}.csv').open()):
            rows.append(dict(graph='Vamana', codec=codec, k=k,
                             search_width=row['Ls'], recall=row[f'Recall@{k}'],
                             qps=row['QPS'], mean_us=row['Mean Latency (mus)'],
                             p99_us=row['99.9 Latency'], metric='p999'))
with (out / 'pilot_search_summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=('graph', 'codec', 'k', 'search_width',
                                           'recall', 'qps', 'mean_us', 'p99_us', 'metric'))
    writer.writeheader()
    writer.writerows(rows)
print(f'Wrote {len(rows)} retrieval rows to {out / "pilot_search_summary.csv"}')
PY
echo "SIFT pilot results: $OUTPUT_DIR"
