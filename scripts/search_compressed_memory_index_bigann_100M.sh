#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../build" && pwd)"
DATA_DIR="$BUILD_DIR/data/bigann"

"$BUILD_DIR/apps/search_memory_index" \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix "$DATA_DIR/compressed_memory_index_bigann_100M_R32_L50_A1.2" \
  --gt_file "$DATA_DIR/bigann_100M_gt.bin" \
  --query_file "$DATA_DIR/bigann_query.fbin" \
  -K 1 -L 2 4 6 8 10 -T 1 \
  --result_path "$DATA_DIR/res" \
  --use_pq_dist 1 \
  --pq_bytes 32 \
  --pq_prefix "$DATA_DIR/bigann_100M.fbin" \
  --pq_exact_rerank_ratio 0.2