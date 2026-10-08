#!/usr/bin/env bash
set -euo pipefail

# One cross-graph dataset: the same fvecs/ivecs inputs feed HNSW and Vamana.
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
OUTPUT_DIR="${OUTPUT_DIR:-$DISKANN_DIR/build/data/cross_graph_${DATASET}}"
DISKANN_BUILD_DIR="${DISKANN_BUILD_DIR:-$DISKANN_DIR/build}"
HNSW_BUILD_DIR="${HNSW_BUILD_DIR:-$HNSW_DIR/build_lvc_sift}"
BUILD_THREADS="${BUILD_THREADS:-4}"
SEARCH_THREADS="${SEARCH_THREADS:-1}"
ROUNDS="${ROUNDS:-3}"
CHAIN_SWEEP="2 4 8 -1"
CXX="${CXX:-g++}"

for kind in train.fvecs test.fvecs neighbors.ivecs; do
  file="$HNSW_DATA_DIR/${DATASET}_$kind"
  if [[ ! -f "$file" ]]; then
    echo "Missing cross-graph input: $file" >&2
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
print('Cross-graph input shapes:', shapes)
PY

# Angular ground truth is cosine based. Unit normalization makes squared L2
# rank vectors in the same order; both graph implementations use these F32 files.
PREPROCESSING="original_f32"
if [[ "$DATASET" == *angular* ]]; then
  source_dir="$HNSW_DATA_DIR"
  HNSW_DATA_DIR="$OUTPUT_DIR/normalized_inputs"
  mkdir -p "$HNSW_DATA_DIR"
  python3 - "$source_dir" "$HNSW_DATA_DIR" "$DATASET" <<'PY'
import math, pathlib, shutil, struct, sys
source, target, name = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
for suffix in ('train.fvecs', 'test.fvecs'):
    source_file, target_file = source / f'{name}_{suffix}', target / f'{name}_{suffix}'
    with source_file.open('rb') as src, target_file.open('wb') as dst:
        while header := src.read(4):
            if len(header) != 4: raise SystemExit(f'Truncated header: {source_file}')
            dim, = struct.unpack('<i', header)
            values = struct.unpack('<%df' % dim, src.read(dim * 4))
            norm = math.sqrt(sum(value * value for value in values))
            if norm == 0 or not math.isfinite(norm):
                raise SystemExit(f'Invalid angular vector norm: {source_file}')
            dst.write(header)
            dst.write(struct.pack('<%df' % dim, *(value / norm for value in values)))
shutil.copyfile(source / f'{name}_neighbors.ivecs', target / f'{name}_neighbors.ivecs')
print(f'Unit-normalized F32 train/query inputs: {target}')
PY
  PREPROCESSING="unit_normalized_f32"
fi

DIM="$(python3 - "$HNSW_DATA_DIR/${DATASET}_train.fvecs" <<'PY'
import struct, sys
with open(sys.argv[1], 'rb') as f:
    print(struct.unpack('<i', f.read(4))[0])
PY
)"
if (( DIM <= 0 || DIM % 8 != 0 )); then
  echo "Dimension must be a positive multiple of 8 for dim/8 PQ: $DIM" >&2
  exit 1
fi
PQ_BYTES=$((DIM / 8))
python3 - "$OUTPUT_DIR/config.json" "$DATASET" "$DIM" "$PQ_BYTES" "$PREPROCESSING" <<'PY'
import json, pathlib, sys
path, dataset, dim, pq_bytes, preprocessing = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps(dict(dataset=dataset, vectors='full_train',
    dimension=int(dim), input_precision='F32', preprocessing=preprocessing,
    root_policy='top_1_percent_degree', chain_max=[2,4,8,-1],
    decoding_cache_ratio=0.01, pq_bytes=int(pq_bytes), two_level_alpha=0.2,
    search_chain_max=2, payload_ratio_basis='raw_f32_bytes/compressed_vector_bytes',
    index_ratio_basis='raw_f32_vector_plus_graph/compressed_vector_plus_graph_and_metadata',
    index_ratio_excludes_pq=True, index_ratio_excludes_decoding_cache=True,
    reference_l2='Euclidean distance to parent; summary excludes roots',
    average_depth='Number of parent edges averaged over all nodes; roots have depth zero',
    deep_vamana_index_ratio='Estimated metadata; deep records are compression-only'), indent=2) + '\n')
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

echo "Converting cross-graph inputs to DiskANN bin format"
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

# Both paths use the same F32 source values and 1% highest-degree roots.
# HNSW stores those exact F32 values in F64 slots; ratios below use F32 bytes.
echo "Building and searching HNSW indexes"
(
  cd "$OUTPUT_DIR"
  "$HNSW_BUILD_DIR/test_build_compressed_hnsw_ablation" \
    --base_dir "$HNSW_DATA_DIR" --dataset "$DATASET" \
    --algorithm DeXOR Gorilla Elf --chain_max $CHAIN_SWEEP --root_policy degree --f32_lossless --reuse_graph --threads "$BUILD_THREADS" \
    --M 16 --ef_construction 200 --output_csv hnsw_build.csv \
    > hnsw_build.log 2>&1
  for k in 1 10; do
    "$HNSW_BUILD_DIR/test_search_compressed_hnsw_ablation" \
      --base_dir "$HNSW_DATA_DIR" --dataset "$DATASET" \
      --algorithm DeXOR Gorilla Elf --chain_max 2 --root_policy degree --threads "$SEARCH_THREADS" \
      --k "$k" --ef 20 50 100 --use_cache 1 --use_tls 0 1 --tls_ratio 0.2 --no_early_stop \
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
  -R 32 -L 50 -T "$BUILD_THREADS" --alpha 1.2 --build_PQ_bytes "$PQ_BYTES" \
  > "$OUTPUT_DIR/vamana_build.log" 2>&1
for suffix in "pq${PQ_BYTES}_pivots.bin" "pq${PQ_BYTES}_compressed.bin"; do
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
for chain in $CHAIN_SWEEP; do
  [[ "$chain" == 2 ]] && continue
  "$DISKANN_BUILD_DIR/apps/convert_lvc_vamana" "$graph" "$train" "${prefix}_ch${chain}" \
    --chain_max "$chain" --compression_only \
    > "$OUTPUT_DIR/vamana_compression_ch${chain}.log"
done
for codec in raw dexor gorilla elf; do
  "$DISKANN_BUILD_DIR/apps/verify_lvc_vamana" "$train" "${prefix}_${codec}.data" \
    > "$OUTPUT_DIR/vamana_verify_${codec}.log" 2>&1
  for k in 1 10; do
    for tls in 0 1; do
    "$DISKANN_BUILD_DIR/apps/search_memory_index" \
      --data_type float --dist_fn l2 --index_path_prefix "${prefix}_${codec}" \
      --query_file "$query" --gt_file "$gt" \
      -K "$k" -L 20 50 100 -T "$SEARCH_THREADS" -R "$ROUNDS" \
      --use_pq_dist "$tls" --pq_bytes "$PQ_BYTES" --pq_prefix "$train" \
      --pq_exact_rerank_ratio 0.2 --lvc_state_cache_ratio 0.01 \
      --result_path "$OUTPUT_DIR/vamana_${codec}_k${k}_tls${tls}" \
      > "$OUTPUT_DIR/vamana_${codec}_k${k}_tls${tls}.log" 2>&1
    cp "${prefix}_${codec}_K${k}_T${SEARCH_THREADS}_search_result.csv" \
      "$OUTPUT_DIR/vamana_${codec}_k${k}_tls${tls}.csv"
    done
  done
done

python3 - "$OUTPUT_DIR" "$DATASET" <<'PY'
import csv, pathlib, re, sys
out, dataset = pathlib.Path(sys.argv[1]), sys.argv[2]
for k in (1, 10):
  for tls in (0, 1):
    for L in (20, 50, 100):
        for kind in ('idx_uint32.bin', 'dists_float.bin'):
            raw = (out / f'vamana_raw_k{k}_tls{tls}_{L}_{kind}').read_bytes()
            for codec in ('dexor', 'gorilla', 'elf'):
                candidate = (out / f'vamana_{codec}_k{k}_tls{tls}_{L}_{kind}').read_bytes()
                if candidate != raw:
                    first = next((i for i, pair in enumerate(zip(candidate, raw))
                                  if pair[0] != pair[1]), min(len(candidate), len(raw)))
                    raise SystemExit(f'FAIL Vamana {codec} K={k} L={L} {kind} byte={first}')
print('PASS Vamana: all codec IDs and L2 bits match raw for every K/L')
print('\nHNSW compression (common F32 baseline):')
for row in csv.DictReader((out / 'hnsw_build.csv').open()):
    print(row['Algorithm'], 'chain=', row['ChainMaxLength'], 'data_ratio_f32=', row['PayloadRatioF32'],
          'index_ratio_f32=', row['FullIndexRatioF32'],
          'build_s=', row['BuildTime(s)'], 'compress_s=', row['CompressTime(s)'])
print('\nVamana compression (baseline: F32):')
for line in (out / 'vamana_compression.log').read_text().splitlines():
    if re.match(r'^(forest|raw|dexor|gorilla|elf)\b', line):
        print(line)
rows = []
for k in (1, 10):
    for row in csv.DictReader((out / f'hnsw_search_k{k}.csv').open()):
        rows.append(dict(dataset=dataset, graph='HNSW', codec=row['Algorithm'].lower(),
                         chain_max=2, k=k, tls=row['UseTwoLevelSearch'],
                         alpha=row['TlsRatio'],
                         search_width=row['ef'], recall=row['Recall'], qps=row['QPS'],
                         mean_us=row['TimePerQuery(us)'], tail_us=row['P99_Latency(us)'],
                         tail_quantile='0.99'))
    for codec in ('raw', 'dexor', 'gorilla', 'elf'):
      for tls in (0, 1):
        for row in csv.DictReader((out / f'vamana_{codec}_k{k}_tls{tls}.csv').open()):
            rows.append(dict(dataset=dataset, graph='Vamana', codec=codec,
                             chain_max=2, k=k, tls=tls, alpha=0.2 if tls else 0.0,
                             search_width=row['Ls'], recall=row[f'Recall@{k}'],
                             qps=row['QPS'], mean_us=row['Mean Latency (mus)'],
                             tail_us=row['99.9 Latency'], tail_quantile='0.999'))
with (out / 'retrieval_ch2.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=('dataset', 'graph', 'codec', 'chain_max', 'k',
                                           'tls', 'alpha', 'search_width', 'recall',
                                           'qps', 'mean_us', 'tail_us', 'tail_quantile'))
    writer.writeheader()
    writer.writerows(rows)
print(f'Wrote {len(rows)} retrieval rows to {out / "retrieval_ch2.csv"}')
PY
echo "Cross-graph results: $OUTPUT_DIR"
