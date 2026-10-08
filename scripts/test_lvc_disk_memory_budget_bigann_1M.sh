#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$DISKANN_DIR/build}"
DATA_DIR="${DATA_DIR:-$BUILD_DIR/data/bigann}"
INDEX_PREFIX="${INDEX_PREFIX:-$DATA_DIR/disk_index_bigann_1M_R32_L50_A1.2}"
QUERY_FILE="${QUERY_FILE:-$DATA_DIR/bigann_query.fbin}"
GT_FILE="${GT_FILE:-$DATA_DIR/bigann_1M_gt.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/lvc_disk_budget_results/bigann_1M}"
BUDGET_MIB="${BUDGET_MIB:-32}"

for path in "${INDEX_PREFIX}_disk.index" "${INDEX_PREFIX}_pq_compressed.bin" "$QUERY_FILE" "$GT_FILE"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
cmake --build "$BUILD_DIR" --target search_disk_index --parallel "${BUILD_THREADS:-4}"
python3 "$SCRIPT_DIR/lvc_disk_memory_budget.py" \
  --binary "$BUILD_DIR/apps/search_disk_index" \
  --index-prefix "$INDEX_PREFIX" --query "$QUERY_FILE" --gt "$GT_FILE" \
  --output-dir "$OUTPUT_DIR" --budget-mib "$BUDGET_MIB" \
  --threads "${THREADS:-1}" --k 1 --L 1 2 3 4 5 6 8 10 --rounds 10 \
  --performance-beamwidth "${PERF_W:-2}" \
  --pilot-nodes "${PILOT_NODES:-8192}" --max-probes "${MAX_PROBES:-32}"
