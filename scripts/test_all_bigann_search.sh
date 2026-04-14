#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running all BigANN search tests"

echo "==================================="
echo "1. Running Search Tests for Memory Index on BigANN 100M"
echo "==================================="
bash "$SCRIPT_DIR/search_memory_index_bigann_100M.sh"

echo "==================================="
echo "2. Running Search Tests for Disk Index on BigANN 500M"
echo "==================================="
bash "$SCRIPT_DIR/search_disk_index_bigann_500M.sh"

echo "==================================="
echo "3. Running Search Tests for Disk Index on BigANN 1B"
echo "==================================="
bash "$SCRIPT_DIR/search_disk_index_bigann_1B.sh"