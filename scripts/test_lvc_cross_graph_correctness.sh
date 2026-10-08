#!/usr/bin/env bash
set -euo pipefail

# Small deterministic F32 test. No external dataset or ground truth is needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "${HNSW_DIR:-}" ]]; then
  if [[ -d "$DISKANN_DIR/../hnswlib_cpp_py" ]]; then
    HNSW_DIR="$(cd "$DISKANN_DIR/../hnswlib_cpp_py" && pwd)"
  else
    HNSW_DIR="$HOME/HNSWCW/compressedhnsw"
  fi
fi
OUT="${OUT:-$DISKANN_DIR/build/data/cross_graph_correctness}"
DBUILD="${DISKANN_BUILD_DIR:-$DISKANN_DIR/build}"
HBUILD="${HNSW_BUILD_DIR:-$HNSW_DIR/build_lvc_cross_graph}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

python3 - "$OUT" <<'PY'
import pathlib, random, struct, sys
out = pathlib.Path(sys.argv[1]); rng = random.Random(20261008)
n, dim = 512, 128
vectors = []
with (out / 'cross_graph_tiny_train.fvecs').open('wb') as f:
    for i in range(n):
        values = [float((i // 32) * 0.25 + (j % 13) * 0.01 + rng.random() * 0.001)
                  for j in range(dim)]
        vectors.append(values)
        f.write(struct.pack('<i', dim)); f.write(struct.pack('<%df' % dim, *values))
with (out / 'cross_graph_tiny_test.fvecs').open('wb') as f, \
     (out / 'cross_graph_tiny_neighbors.ivecs').open('wb') as gt:
    for i in range(10):
        f.write(struct.pack('<i', dim)); f.write(struct.pack('<%df' % dim, *vectors[i]))
        gt.write(struct.pack('<iI', 1, i))
print(f'Generated F32 vectors={n} dim={dim}')
PY

cmake -S "$DISKANN_DIR" -B "$DBUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$DBUILD" --target fvecs_to_bin ivecs_to_bin build_memory_index search_memory_index --parallel "${BUILD_THREADS:-4}"
cmake -S "$HNSW_DIR" -B "$HBUILD" -DHNSWLIB_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$HBUILD" --target test_build_compressed_hnsw_ablation test_search_compressed_hnsw_ablation --parallel "${BUILD_THREADS:-4}"
g++ -std=c++17 -O2 -I "$DISKANN_DIR/include" \
  "$DISKANN_DIR/apps/convert_lvc_vamana.cpp" \
  "$DISKANN_DIR/include/lvc_codec/encoding_algorithms/elf/elf64_utils.cpp" \
  -o "$OUT/convert_lvc_vamana"
g++ -std=c++17 -O2 -I "$DISKANN_DIR/include" \
  "$DISKANN_DIR/apps/verify_lvc_vamana.cpp" \
  "$DISKANN_DIR/include/lvc_codec/encoding_algorithms/elf/elf64_utils.cpp" \
  -o "$OUT/verify_lvc_vamana"

"$DBUILD/apps/utils/fvecs_to_bin" float "$OUT/cross_graph_tiny_train.fvecs" "$OUT/train.fbin"
"$DBUILD/apps/utils/fvecs_to_bin" float "$OUT/cross_graph_tiny_test.fvecs" "$OUT/test.fbin"
"$DBUILD/apps/utils/ivecs_to_bin" "$OUT/cross_graph_tiny_neighbors.ivecs" "$OUT/gt.bin"
"$DBUILD/apps/build_memory_index" --data_type float --dist_fn l2 \
  --data_path "$OUT/train.fbin" --index_path_prefix "$OUT/vamana_graph" \
  -R 16 -L 32 -T "${BUILD_THREADS:-4}" --alpha 1.2 --build_PQ_bytes 16 \
  > "$OUT/vamana_build.log" 2>&1

(
  cd "$OUT"
  "$HBUILD/test_build_compressed_hnsw_ablation" --base_dir "$OUT" \
    --dataset cross_graph_tiny --algorithm DeXOR Gorilla Elf \
    --chain_max 2 4 8 -1 --root_policy degree --f32_lossless --verify_vectors --reuse_graph \
    --threads "${BUILD_THREADS:-4}" --output_csv hnsw_build.csv \
    > hnsw_build.log 2>&1
  "$HBUILD/test_search_compressed_hnsw_ablation" --base_dir "$OUT" \
    --dataset cross_graph_tiny --algorithm DeXOR Gorilla Elf \
    --chain_max 2 --root_policy degree --use_cache 1 --use_tls 0 1 \
    --tls_ratio 0.2 --k 1 --ef 20 50 --threads 1 --output_csv hnsw_search.csv \
    > hnsw_search.log 2>&1
)
for chain in 2 4 8 -1; do
  prefix="$OUT/vamana_ch${chain}"
  if [[ "$chain" == 2 ]]; then
    "$OUT/convert_lvc_vamana" "$OUT/vamana_graph" "$OUT/train.fbin" "$prefix" \
      > "$OUT/vamana_ch${chain}.log"
    for codec in raw dexor gorilla elf; do
      "$OUT/verify_lvc_vamana" "$OUT/train.fbin" "${prefix}_${codec}.data" \
        > "$OUT/verify_${codec}.log"
    done
  else
    "$OUT/convert_lvc_vamana" "$OUT/vamana_graph" "$OUT/train.fbin" "$prefix" \
      --chain_max "$chain" --compression_only > "$OUT/vamana_ch${chain}.log"
  fi
done

for tls in 0 1; do
  for codec in raw dexor gorilla elf; do
    "$DBUILD/apps/search_memory_index" --data_type float --dist_fn l2 \
      --index_path_prefix "$OUT/vamana_ch2_${codec}" \
      --query_file "$OUT/test.fbin" --gt_file "$OUT/gt.bin" \
      -K 1 -L 20 50 -T 1 -R 1 --use_pq_dist "$tls" --pq_bytes 16 \
      --pq_prefix "$OUT/train.fbin" --pq_exact_rerank_ratio 0.2 \
      --lvc_state_cache_ratio 0.01 --result_path "$OUT/${codec}_tls${tls}" \
      > "$OUT/search_${codec}_tls${tls}.log" 2>&1
  done
done

python3 - "$OUT" <<'PY'
import csv, pathlib, sys
out = pathlib.Path(sys.argv[1]); n = 512; roots = max(1, n // 100)
forest_by_group = {}
for path in sorted(out.glob('*.forest.csv')) + sorted(out.glob('*_forest.csv')):
    rows = list(csv.DictReader(path.open()))
    if len(rows) != n: raise SystemExit(f'{path}: wrong node count')
    hnsw = 'parent_internal_id' in rows[0]
    parents = [int(row['parent_internal_id' if hnsw else 'parent'])
               if row['parent_internal_id' if hnsw else 'parent'] else -1 for row in rows]
    depths = [int(row['depth']) for row in rows]
    if parents.count(-1) != roots: raise SystemExit(f'{path}: wrong root count')
    degrees = [int(row['level0_degree' if hnsw else 'degree']) for row in rows]
    expected_roots = set(sorted(range(n), key=lambda i: (-degrees[i], i))[:roots])
    actual_roots = {i for i, parent in enumerate(parents) if parent == -1}
    if actual_roots != expected_roots: raise SystemExit(f'{path}: roots are not top degree')
    if hnsw:
        chain = next(part for part in path.name.split('_') if part.startswith('ch'))
        group = ('hnsw', chain)
    else:
        group = ('vamana', path.name.split('_forest.csv')[0])
    previous = forest_by_group.setdefault(group, parents)
    if previous != parents: raise SystemExit(f'{path}: codec changed predecessor forest')
    for i in range(n):
        seen = set(); cursor = i
        while cursor >= 0:
            if cursor in seen or cursor >= n: raise SystemExit(f'{path}: cycle at {i}')
            seen.add(cursor); cursor = parents[cursor]
        if len(seen) - 1 != depths[i]: raise SystemExit(f'{path}: depth mismatch at {i}')
    print(f'PASS forest {path.name}: roots={roots} max_depth={max(depths)}')
for row in csv.DictReader((out / 'hnsw_build.csv').open()):
    if int(row['Roots']) != roots or float(row['PayloadRatioF32']) <= 0:
        raise SystemExit('HNSW compression statistics invalid')
    if int(row['ChainMaxLength']) == 2 and int(row['MaxDepth']) > 1:
        raise SystemExit('HNSW chain=2 exceeded depth 1')
for path in out.glob('vamana_ch*_compression.csv'):
    for row in csv.DictReader(path.open()):
        if int(row['roots']) != roots or float(row['payload_ratio_f32']) <= 0:
            raise SystemExit(f'{path}: Vamana compression statistics invalid')
search = list(csv.DictReader((out / 'hnsw_search.csv').open()))
if len(search) < 6 or {row['UseTwoLevelSearch'] for row in search} != {'0', '1'}:
    raise SystemExit('HNSW two-level on/off search missing')
for tls in (0, 1):
    for L in (20, 50):
        for kind in ('idx_uint32.bin', 'dists_float.bin'):
            raw = (out / f'raw_tls{tls}_{L}_{kind}').read_bytes()
            for codec in ('dexor', 'gorilla', 'elf'):
                if (out / f'{codec}_tls{tls}_{L}_{kind}').read_bytes() != raw:
                    raise SystemExit(f'Vamana {codec} TLS={tls} L={L} {kind} differs from raw')
print('PASS cross-graph F32 lossless codecs, forest, depth limits, and byte statistics')
PY
