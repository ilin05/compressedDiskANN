#!/usr/bin/env bash
# Sequential full-cache recheck with periodic process/host memory diagnostics.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export OUTPUT_DIR="${OUTPUT_DIR:-$repo/scripts/lvc_large_cache_results/bigann_1B_swap_recheck_$(date -u +%Y%m%dT%H%M%SZ)}"
export CASES="${CASES:-alp:400000000 dexor:100000000}"
export MEMORY_MONITOR=1 MEMORY_SAMPLE_SECONDS="${MEMORY_SAMPLE_SECONDS:-5}"
exec bash "$repo/scripts/test_lvc_disk_large_cache_bigann_1B.sh"
