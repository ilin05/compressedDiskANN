#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISKANN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$DISKANN_DIR/build}"
CXX="${CXX:-g++}"

mkdir -p "$BUILD_DIR/apps"
"$CXX" -std=c++17 -O2 -I "$DISKANN_DIR/include" \
  "$DISKANN_DIR/apps/test_lvc_forest.cpp" \
  "$DISKANN_DIR/include/lvc_codec/encoding_algorithms/elf/elf64_utils.cpp" \
  -o "$BUILD_DIR/apps/test_lvc_forest"

"$BUILD_DIR/apps/test_lvc_forest" "$@"
