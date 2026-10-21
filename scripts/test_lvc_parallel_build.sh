#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
mkdir -p "$build/tests"
"${CXX:-g++}" -std=c++17 -Ofast -fopenmp -I "$repo/include" \
    "$repo/tests/lvc_parallel_build_test.cpp" \
    "$repo/include/lvc_codec/encoding_algorithms/elf/elf64_utils.cpp" \
    -o "$build/tests/lvc_parallel_build_test"
OMP_NUM_THREADS=${LVC_BUILD_THREADS:-32} "$build/tests/lvc_parallel_build_test" "${LVC_BUILD_THREADS:-32}"
