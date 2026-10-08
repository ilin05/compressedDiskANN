#!/usr/bin/env bash
set -euo pipefail

# 32 GiB covers cached graph, node lookups, vector payload and LVC records.
# PQ codes and the 1% decoding state cache are reported outside this cap.
# The default boundary resolution is 100,000 nodes to limit repeated 1B loads;
# set NODE_GRANULARITY=1 for an adjacent-node boundary search.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$DISKANN_DIR/build}"
DATA_DIR="${DATA_DIR:-$BUILD_DIR/data/bigann}"
INDEX_PREFIX="${INDEX_PREFIX:-$DATA_DIR/disk_index_bigann_1B_R32_L50_A1.2}"
QUERY_FILE="${QUERY_FILE:-$DATA_DIR/bigann_query.fbin}"
GT_FILE="${GT_FILE:-$DATA_DIR/bigann_1B_gt.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/lvc_disk_budget_results/bigann_1B}"
PILOT_SUMMARY="${PILOT_SUMMARY:-$SCRIPT_DIR/lvc_disk_budget_results/bigann_1M/summary.csv}"
BUDGET_MIB="${BUDGET_MIB:-32768}"

for path in "${INDEX_PREFIX}_disk.index" "${INDEX_PREFIX}_pq_compressed.bin" \
            "${INDEX_PREFIX}_pq_pivots.bin" "$QUERY_FILE" "$GT_FILE"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
cmake --build "$BUILD_DIR" --target search_disk_index --parallel "${BUILD_THREADS:-4}"

pilot_args=()
if [[ -s "$PILOT_SUMMARY" ]]; then
  pilot_args+=(--pilot-summary "$PILOT_SUMMARY" --pilot-summary-points 1000000)
  echo "Using bigann1M pilot counts from $PILOT_SUMMARY"
else
  echo "No bigann1M summary found; using a 1M-node pilot for each codec"
fi
resume_args=()
if [[ "${RESUME_PROBES:-1}" == 1 ]]; then resume_args+=(--resume-probes); fi

python3 "$SCRIPT_DIR/lvc_disk_memory_budget.py" \
  --binary "$BUILD_DIR/apps/search_disk_index" \
  --index-prefix "$INDEX_PREFIX" --query "$QUERY_FILE" --gt "$GT_FILE" \
  --output-dir "$OUTPUT_DIR" --budget-mib "$BUDGET_MIB" \
  --expected-points 1000000000 --expected-pq-bytes 32 \
  --threads "${THREADS:-1}" --k 1 --L 1 2 3 4 5 6 8 10 --rounds 10 \
  --performance-beamwidth "${PERF_W:-2}" \
  --pilot-nodes "${PILOT_NODES:-1000000}" \
  --node-granularity "${NODE_GRANULARITY:-100000}" \
  --max-probes "${MAX_PROBES:-32}" \
  "${pilot_args[@]}" "${resume_args[@]}"
