#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../build" && pwd)"
DATA_DIR="$BUILD_DIR/data/bigann"

INDEX_PREFIX="$DATA_DIR/disk_index_bigann_1B_R32_L50_A1.2"
QUERY_FILE="$DATA_DIR/bigann_query.fbin"
GT_FILE="$DATA_DIR/bigann_1B_gt.bin"

THREADS="${THREADS:-1}"
NUM_ROUNDS="${NUM_ROUNDS:-10}"

# Override from the environment if needed, e.g.:
# CACHE_NODE_COUNTS="0 50000000 100000000" bash search_compressed_disk_index_bigann_1B_cache_sensitivity.sh
# CACHE_NODE_COUNTS="${CACHE_NODE_COUNTS:-0 10000000 25000000 50000000 100000000 200000000 400000000}"
CACHE_NODE_COUNTS="${CACHE_NODE_COUNTS:-0  2500000 5000000 10000000 20000000 50000000 100000000}"

for CACHE_NODES in $CACHE_NODE_COUNTS; do
  echo "[BigANN-1B] Search with num_nodes_to_cache=${CACHE_NODES}"

  "$BUILD_DIR/apps/search_disk_index" \
    --data_type float \
    --dist_fn l2 \
    --index_path_prefix "$INDEX_PREFIX" \
    --query_file "$QUERY_FILE" \
    --gt_file "$GT_FILE" \
    -K 1 -L 1 2 3 4 5 6 8 10 -T "$THREADS" -R "$NUM_ROUNDS" \
    --result_path "$DATA_DIR/res_cache_${CACHE_NODES}_K1" \
    --num_nodes_to_cache "$CACHE_NODES"

  "$BUILD_DIR/apps/search_disk_index" \
    --data_type float \
    --dist_fn l2 \
    --index_path_prefix "$INDEX_PREFIX" \
    --query_file "$QUERY_FILE" \
    --gt_file "$GT_FILE" \
    -K 10 -L 10 11 12 13 14 16 18 20 -T "$THREADS" -R "$NUM_ROUNDS" \
    --result_path "$DATA_DIR/res_cache_${CACHE_NODES}_K10" \
    --num_nodes_to_cache "$CACHE_NODES"
done

echo "Done. CSV files are written next to the index prefix with _C<num_nodes_to_cache>_ in the file name."