#!/usr/bin/env bash
# Same query order, ten rounds and diagnostics as the 100M recheck.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CACHE_NODES=25000000
exec bash "$script_dir/test_lvc_disk_dexor_100M_recheck_bigann_1B.sh"
