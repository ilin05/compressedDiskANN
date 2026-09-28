#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$DISKANN_DIR/.." && pwd)"
HNSW_DIR="${HNSW_DIR:-$WORKSPACE_DIR/hnswlib_cpp_py}"
BUILD_DIR="${BUILD_DIR:-$DISKANN_DIR/build}"
DATASET="${DATASET:-$BUILD_DIR/data/bigann/bigann_1M.fbin}"
QUERY_FILE="${QUERY_FILE:-$BUILD_DIR/data/bigann/bigann_query.fbin}"
RESULT_CSV="${RESULT_CSV:-$BUILD_DIR/data/bigann/lvc_d0_f32_codec_results.csv}"
SAMPLE_COUNT="${SAMPLE_COUNT:-4096}"
ID_START="${ID_START:-0}"
CHAIN_LENGTH="${CHAIN_LENGTH:-4}"
SEED="${SEED:-20260928}"
CXX="${CXX:-g++}"

if [[ ! -f "$DATASET" ]]; then
  echo "Dataset not found: $DATASET" >&2
  exit 1
fi
if [[ ! -f "$HNSW_DIR/hnswlib/compressed_codecs.h" ]]; then
  echo "HNSW codecs not found: $HNSW_DIR (set HNSW_DIR to their checkout)" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR/apps" "$(dirname "$RESULT_CSV")"
"$CXX" -std=c++17 -O2 -I "$HNSW_DIR" \
  "$DISKANN_DIR/apps/test_lvc_f32_codecs.cpp" \
  "$HNSW_DIR/encoding_algorithms/elf/elf64_utils.cpp" \
  -o "$BUILD_DIR/apps/test_lvc_f32_codecs"

ARGS=(--input "$DATASET" --count "$SAMPLE_COUNT" --id-start "$ID_START"
      --chain-length "$CHAIN_LENGTH" --seed "$SEED" --csv "$RESULT_CSV")
if [[ -f "$QUERY_FILE" ]]; then
  ARGS+=(--query "$QUERY_FILE")
fi
"$BUILD_DIR/apps/test_lvc_f32_codecs" "${ARGS[@]}"
echo "Results: $RESULT_CSV"
