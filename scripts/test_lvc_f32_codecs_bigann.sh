#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
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
if [[ ! -f "$DISKANN_DIR/include/lvc_codec/hnswlib/compressed_codecs.h" ]]; then
  echo "Vendored LVC codecs not found in DiskANN/include/lvc_codec" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR/apps" "$(dirname "$RESULT_CSV")"
"$CXX" -std=c++17 -O2 -I "$DISKANN_DIR/include" \
  "$DISKANN_DIR/apps/test_lvc_f32_codecs.cpp" \
  "$DISKANN_DIR/include/lvc_codec/encoding_algorithms/elf/elf64_utils.cpp" \
  -o "$BUILD_DIR/apps/test_lvc_f32_codecs"

ARGS=(--input "$DATASET" --count "$SAMPLE_COUNT" --id-start "$ID_START"
      --chain-length "$CHAIN_LENGTH" --seed "$SEED" --csv "$RESULT_CSV")
if [[ -f "$QUERY_FILE" ]]; then
  ARGS+=(--query "$QUERY_FILE")
fi
"$BUILD_DIR/apps/test_lvc_f32_codecs" "${ARGS[@]}"
echo "Results: $RESULT_CSV"
