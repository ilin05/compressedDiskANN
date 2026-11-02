#!/usr/bin/env bash
# Run serially to avoid competing for memory and SSD I/O.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${OUTPUT_DIR:-"$script_dir/lvc_disk_cache_sensitivity_recheck_results/bigann_1B_dexor_C10M_25M_$(date -u +%Y%m%dT%H%M%SZ)"}
for size in 10M 25M; do
    echo "Starting DeXOR ${size} recheck"
    OUTPUT_DIR="$out/bigann_1B_dexor_C${size}" \
        bash "$script_dir/test_lvc_disk_dexor_${size}_recheck_bigann_1B.sh"
done
echo "Both rechecks completed: $out"
