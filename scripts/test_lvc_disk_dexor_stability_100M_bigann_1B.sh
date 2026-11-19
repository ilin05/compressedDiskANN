#!/usr/bin/env bash
# Run only the 100M-node case. A single load serves both 50-round sweeps.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CACHE_NODES_LIST=100000000 ROUNDS=100 CHECKPOINT_ROUNDS=50
export OUTPUT_DIR="${OUTPUT_DIR:-$script_dir/lvc_disk_dexor_stability_results/bigann_1B_dexor_C100M_R100_$(date -u +%Y%m%dT%H%M%SZ)}"
exec bash "$script_dir/test_lvc_disk_dexor_stability_bigann_1B.sh"
