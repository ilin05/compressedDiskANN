#!/bin/bash

set -e

echo "Running all BigANN search tests"

echo "==================================="
echo "1. Running Search Tests for Compressed Memory Index on BigANN 100M"
echo "==================================="
bash search_compressed_memory_index_bigann_100M.sh

echo "==================================="
echo "2. Running Search Tests for Compressed Memory Index on BigANN 500M"
echo "==================================="
bash search_compressed_memory_index_bigann_500M.sh

echo "==================================="
echo "3. Running Search Tests for Compressed Disk Index on BigANN 1B"
echo "==================================="
bash search_compressed_disk_index_bigann_1B.sh

# usage: nohup bash test_all_bigann_search.sh > compressed_diskann_bigann_search_all_output.log 2>&1 &