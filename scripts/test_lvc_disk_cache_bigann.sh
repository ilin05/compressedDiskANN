#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$DISKANN_DIR/build}"
DATA_DIR="${DATA_DIR:-$BUILD_DIR/data/bigann}"
INDEX_PREFIX="${INDEX_PREFIX:-$DATA_DIR/disk_index_bigann_1M_R32_L50_A1.2}"
QUERY_FILE="${QUERY_FILE:-$DATA_DIR/bigann_query.fbin}"
GT_FILE="${GT_FILE:-null}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$DATA_DIR/lvc_d3_disk_cache_1M}"
CACHE_NODES="${CACHE_NODES:-10000}"
THREADS="${THREADS:-1 4}"
K="${K:-10}"
L="${L:-50}"
CHECK_W=1
PERF_W="${PERF_W:-${W:-2}}"
RUN_PERF="${RUN_PERF:-1}"

mkdir -p "$(dirname "$OUTPUT_PREFIX")"
cmake --build "$BUILD_DIR" --target search_disk_index --parallel 4

run_search() {
  local phase="$1" codec="$2" thread_count="$3" beamwidth="$4"
  local result_prefix="${OUTPUT_PREFIX}_${phase}_${codec}_T${thread_count}"
  local csv_path="${INDEX_PREFIX}_K${K}_T${thread_count}_C${CACHE_NODES}_search_result.csv"
  local verify_args=()
  if [[ "$phase" == check && "$codec" != raw ]]; then verify_args+=(--verify_cache); fi
  "$BUILD_DIR/apps/search_disk_index" \
    --data_type float --dist_fn l2 --index_path_prefix "$INDEX_PREFIX" \
    --query_file "$QUERY_FILE" --gt_file "$GT_FILE" \
    -K "$K" -L "$L" -W "$beamwidth" -T "$thread_count" \
    --num_nodes_to_cache "$CACHE_NODES" --cache_codec "$codec" \
    --result_path "$result_prefix" "${verify_args[@]}" \
    > "${result_prefix}.log" 2>&1
  cp "$csv_path" "${result_prefix}.csv"
  grep -E 'LVC disk cache mode|LVC disk cache PASS|Peak RSS' "${result_prefix}.log" | tail -n 5 || true
}

for thread_count in $THREADS; do
  echo "Correctness: threads=$thread_count beamwidth=$CHECK_W"
  for codec in raw dexor gorilla elf; do
    run_search check "$codec" "$thread_count" "$CHECK_W"
  done

  python3 - "${OUTPUT_PREFIX}_check" "$thread_count" "$L" "$K" <<'PY'
import pathlib
import csv
import re
import struct
import sys

prefix, threads, search_l, k = sys.argv[1:]
raw_prefix = f'{prefix}_raw_T{threads}'
raw_log = pathlib.Path(raw_prefix + '.log').read_text()
raw_cache = re.search(r'Cache node list: count=(\d+) hash=(\d+)', raw_log)
if raw_cache is None:
    raise SystemExit('FAIL raw: cache node list hash missing')
for codec in ('dexor', 'gorilla', 'elf'):
    log = pathlib.Path(f'{prefix}_{codec}_T{threads}.log').read_text()
    match = re.search(r'Cache node list: count=(\d+) hash=(\d+)', log)
    if match is None or match.groups() != raw_cache.groups():
        raise SystemExit(f'FAIL {codec} T={threads}: cache node list differs from raw')
for suffix in ('idx_uint32.bin', 'dists_float.bin'):
    raw = pathlib.Path(f'{raw_prefix}_{search_l}_{suffix}').read_bytes()
    for codec in ('dexor', 'gorilla', 'elf'):
        path = pathlib.Path(f'{prefix}_{codec}_T{threads}_{search_l}_{suffix}')
        actual = path.read_bytes()
        if actual != raw:
            offset = next((i for i, pair in enumerate(zip(actual, raw)) if pair[0] != pair[1]),
                          min(len(actual), len(raw)))
            item = (offset - 8) // 4
            fmt = '<I' if suffix.startswith('idx') else '<f'
            if offset >= 8 and 8 + 4 * (item + 1) <= min(len(actual), len(raw)):
                expected_value = struct.unpack_from(fmt, raw, 8 + 4 * item)[0]
                actual_value = struct.unpack_from(fmt, actual, 8 + 4 * item)[0]
                raise SystemExit(f'FAIL {codec} T={threads} {suffix} query={item // int(k)} '
                                 f'rank={item % int(k)} raw={expected_value} decoded={actual_value}')
            raise SystemExit(f'FAIL {codec} T={threads} {suffix} byte={offset}')
        print(f'PASS {codec} T={threads} {suffix}: exact match with raw cache')
for codec in ('raw', 'dexor', 'gorilla', 'elf'):
    path = pathlib.Path(f'{prefix}_{codec}_T{threads}.csv')
    with path.open(newline='') as file:
        row = next(csv.DictReader(file))
    print(f'{codec} T={threads} QPS={row["QPS"]} '
          f'mean_us={row["Mean Latency (mus)"]} p999_us={row["99.9 Latency"]} '
          f'mean_IOs={row["Mean IOs"]}')
PY

  if [[ "$RUN_PERF" == 1 ]]; then
    echo "Performance: threads=$thread_count beamwidth=$PERF_W"
    for codec in raw dexor gorilla elf; do
      run_search perf "$codec" "$thread_count" "$PERF_W"
    done
    python3 - "${OUTPUT_PREFIX}_perf" "$thread_count" <<'PY'
import csv
import pathlib
import sys

prefix, threads = sys.argv[1:]
for codec in ('raw', 'dexor', 'gorilla', 'elf'):
    with pathlib.Path(f'{prefix}_{codec}_T{threads}.csv').open(newline='') as file:
        row = next(csv.DictReader(file))
    recall_column = next((name for name in row if name.startswith('Recall@')), None)
    recall = f' recall={row[recall_column]}' if recall_column and row[recall_column] else ''
    print(f'{codec} T={threads} QPS={row["QPS"]} '
          f'mean_us={row["Mean Latency (mus)"]} p999_us={row["99.9 Latency"]} '
          f'mean_IOs={row["Mean IOs"]}{recall}')
PY
  fi
done
