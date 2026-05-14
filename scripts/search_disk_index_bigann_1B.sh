#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../build" && pwd)"
DATA_DIR="$BUILD_DIR/data/bigann"

"$BUILD_DIR/apps/search_disk_index" \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix "$DATA_DIR/disk_index_bigann_1B_R32_L50_A1.2" \
  --query_file "$DATA_DIR/bigann_query.fbin" \
  --gt_file "$DATA_DIR/bigann_1B_gt.bin" \
  -K 1 -L 1 2 3 4 5 6 8 10 -T 1 -R 100 \
  --result_path "$DATA_DIR/res" \
  --num_nodes_to_cache 100000000

"$BUILD_DIR/apps/search_disk_index" \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix "$DATA_DIR/disk_index_bigann_1B_R32_L50_A1.2" \
  --query_file "$DATA_DIR/bigann_query.fbin" \
  --gt_file "$DATA_DIR/bigann_1B_gt.bin" \
  -K 10 -L 10 11 12 13 14 16 18 20 -T 1 -R 100 \
  --result_path "$DATA_DIR/res" \
  --num_nodes_to_cache 100000000