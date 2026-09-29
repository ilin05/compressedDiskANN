#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$DISKANN_DIR/build}"
DATA_DIR="${DATA_DIR:-$BUILD_DIR/data/bigann}"
VAMANA_PREFIX="${VAMANA_PREFIX:-$DATA_DIR/lvc_d2_vamana_1M}"
DISK_PREFIX="${DISK_PREFIX:-$DATA_DIR/disk_index_bigann_1M_R32_L50_A1.2}"
QUERY_FILE="${QUERY_FILE:-$DATA_DIR/bigann_query.fbin}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$DATA_DIR/lvc_state_cache_1M}"
GT_FILE="${GT_FILE:-null}"
THREADS="${THREADS:-1}"
K="${K:-10}"
L="${L:-50}"
CACHE_NODES="${CACHE_NODES:-10000}"
PQ_PREFIX="${PQ_PREFIX:-}"
PQ_BYTES="${PQ_BYTES:-16}"
PQ_EXACT_RERANK_RATIO="${PQ_EXACT_RERANK_RATIO:-0.2}"

mkdir -p "$(dirname "$OUTPUT_PREFIX")"
cmake --build "$BUILD_DIR" --target search_memory_index search_disk_index --parallel 4

pq_args=(--use_pq_dist 0 --pq_exact_rerank_ratio 0)
if [[ -n "$PQ_PREFIX" ]]; then
  pq_args=(--use_pq_dist 1 --pq_prefix "$PQ_PREFIX" --pq_bytes "$PQ_BYTES"
           --pq_exact_rerank_ratio "$PQ_EXACT_RERANK_RATIO")
fi
echo "Vamana PQ mode: $([[ -n "$PQ_PREFIX" ]] && echo enabled || echo disabled)"

for codec in dexor gorilla elf; do
  for ratio in 0 0.01; do
    label=off
    [[ "$ratio" == 0 ]] || label=on
    result="${OUTPUT_PREFIX}_vamana_${codec}_${label}"
    "$BUILD_DIR/apps/search_memory_index" \
      --data_type float --dist_fn l2 --index_path_prefix "${VAMANA_PREFIX}_${codec}" \
      --query_file "$QUERY_FILE" --gt_file "$GT_FILE" \
      -K "$K" -L "$L" -T "$THREADS" -R 1 \
      "${pq_args[@]}" --lvc_state_cache_ratio "$ratio" \
      --result_path "$result" > "${result}.log" 2>&1
    cp "${VAMANA_PREFIX}_${codec}_K${K}_T${THREADS}_search_result.csv" "${result}.csv"
    echo "Vamana $codec $label: $(grep -m1 'LVC Vamana state cache' "${result}.log" || true)"
  done
done

for codec in dexor gorilla elf; do
  for ratio in 0 0.01; do
    label=off
    [[ "$ratio" == 0 ]] || label=on
    result="${OUTPUT_PREFIX}_disk_${codec}_${label}"
    "$BUILD_DIR/apps/search_disk_index" \
      --data_type float --dist_fn l2 --index_path_prefix "$DISK_PREFIX" \
      --query_file "$QUERY_FILE" --gt_file "$GT_FILE" \
      -K "$K" -L "$L" -W 1 -T "$THREADS" \
      --num_nodes_to_cache "$CACHE_NODES" --cache_codec "$codec" \
      --verify_cache --lvc_state_cache_ratio "$ratio" \
      --result_path "$result" > "${result}.log" 2>&1
    cp "${DISK_PREFIX}_K${K}_T${THREADS}_C${CACHE_NODES}_search_result.csv" "${result}.csv"
    grep -E 'LVC disk cache mode|LVC disk cache PASS|LVC decode L=' "${result}.log" | tail -n 3
  done
done

python3 - "$OUTPUT_PREFIX" "$L" <<'PY'
import pathlib
import csv
import re
import sys

prefix, search_l = sys.argv[1:]
for path in ('vamana', 'disk'):
    for codec in ('dexor', 'gorilla', 'elf'):
        names = [f'{prefix}_{path}_{codec}_{state}' for state in ('off', 'on')]
        for suffix in ('idx_uint32.bin', 'dists_float.bin'):
            before = pathlib.Path(f'{names[0]}_{search_l}_{suffix}').read_bytes()
            after = pathlib.Path(f'{names[1]}_{search_l}_{suffix}').read_bytes()
            if before != after:
                position = next((i for i, pair in enumerate(zip(before, after))
                                 if pair[0] != pair[1]), min(len(before), len(after)))
                raise SystemExit(f'FAIL {path} {codec} {suffix}: first differing byte {position}')
        if path == 'disk':
            logs = [pathlib.Path(f'{name}.log').read_text() for name in names]
            selected = re.search(r'LVC disk cache mode=.*?roots=(\d+).*?state_cache_roots=(\d+)', logs[1])
            if selected is None or selected.group(1) != selected.group(2):
                raise SystemExit(f'FAIL disk {codec}: default state cache differs from forest roots')
            metrics = [re.findall(r'LVC decode L=.*?replay_records=(\d+) state_hits=(\d+)', log)
                       for log in logs]
            if not all(metrics):
                raise SystemExit(f'FAIL disk {codec}: decode metrics missing')
            replay_off, hits_off = map(int, metrics[0][-1])
            replay_on, hits_on = map(int, metrics[1][-1])
            if hits_off != 0 or hits_on <= 0 or replay_on >= replay_off:
                raise SystemExit(f'FAIL disk {codec}: state cache did not reduce replay records')
            print(f'PASS disk {codec}: exact results; replay {replay_off}->{replay_on}; hits={hits_on}')
        else:
            log = pathlib.Path(f'{names[1]}.log').read_text()
            selected = re.search(r'LVC Vamana forest roots=(\d+) state_cache_roots=(\d+)', log)
            if selected is None or selected.group(1) != selected.group(2):
                raise SystemExit(f'FAIL Vamana {codec}: default state cache differs from forest roots')
            print(f'PASS Vamana {codec}: exact IDs and L2 bits with state cache off/on')
        for name in names:
            with pathlib.Path(f'{name}.csv').open(newline='') as file:
                row = next(csv.DictReader(file))
            recall = next((row[key] for key in row if key.startswith('Recall@')), None)
            print(f'{path} {codec} {name.rsplit("_", 1)[-1]} QPS={row["QPS"]} '
                  f'mean_us={row["Mean Latency (mus)"]} '
                  f'recall={recall if recall else "n/a"}')
PY
