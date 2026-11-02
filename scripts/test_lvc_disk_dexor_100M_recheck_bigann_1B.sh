#!/usr/bin/env bash
# One index/cache load, then ten complete query-set passes for every L.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
index=${INDEX_PREFIX:-"$data/disk_index_bigann_1B_R32_L50_A1.2"}
query=${QUERY_FILE:-"$data/bigann_query.fbin"}
gt=${GT_FILE:-"$data/bigann_1B_gt.bin"}
cache_nodes=${CACHE_NODES:-100000000}
case "$cache_nodes" in
    10000000) cache_label=10M ;;
    25000000) cache_label=25M ;;
    100000000) cache_label=100M ;;
    *) echo "CACHE_NODES must be 10000000, 25000000 or 100000000." >&2; exit 1 ;;
esac
out=${OUTPUT_DIR:-"$repo/scripts/lvc_disk_cache_sensitivity_recheck_results/bigann_1B_dexor_C${cache_label}_$(date -u +%Y%m%dT%H%M%SZ)"}
load_threads=${LVC_BUILD_THREADS:-32}
interval=${MONITOR_INTERVAL:-5}
mkdir -p "$out"
out=$(cd -- "$out" && pwd)
if [[ -e "$out/rounds.csv" || -e "$out/command.sh" ]]; then
    echo "Existing run in $out; choose a new OUTPUT_DIR to preserve its diagnostics." >&2
    exit 1
fi

if [[ ${SKIP_BUILD:-0} != 1 && ${DRY_RUN:-0} != 1 ]]; then
    cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build" --target search_disk_index --parallel "${BUILD_THREADS:-4}"
fi
export OMP_NUM_THREADS="$load_threads" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
command=("$build/apps/search_disk_index"
    --data_type float --dist_fn l2 --index_path_prefix "$index"
    --query_file "$query" --gt_file "$gt" --cache_codec dexor
    --num_nodes_to_cache "$cache_nodes" --lvc_chain_max 2
    --lvc_build_threads "$load_threads" --lvc_fallback_root_candidates 1
    --lvc_state_cache_ratio 0.01 --cache_budget_include_state
    -K 1 -L 1 2 3 4 5 6 8 10 -T 1 -W 2 -R 10
    --result_path "$out/dexor_ch2_C${cache_label}" --per_round_csv "$out/rounds.csv")
printf '#!/usr/bin/env bash\n' > "$out/command.sh"
printf 'export OMP_NUM_THREADS=%q OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n' "$load_threads" >> "$out/command.sh"
printf '%q ' "${command[@]}" >> "$out/command.sh"
printf '\n' >> "$out/command.sh"
echo "Results: $out"
if [[ ${DRY_RUN:-0} == 1 ]]; then
    cat "$out/command.sh"
    exit 0
fi
for input in "${index}_disk.index" "${index}_pq_compressed.bin" "${index}_pq_pivots.bin" "$query" "$gt"; do
    [[ -r "$input" ]] || { echo "Missing input: $input" >&2; exit 1; }
done
"$build/apps/search_disk_index" --help > "$out/binary_help.txt" 2>&1
if ! grep -q -- '--per_round_csv' "$out/binary_help.txt"; then
    echo "Binary lacks --per_round_csv; rebuild search_disk_index with the updated source." >&2
    exit 1
fi
{
    date -u
    uname -a
    git -C "$repo" rev-parse HEAD
    free -k
    cat /proc/sys/vm/swappiness
} > "$out/host_before.txt" 2>&1

monitors=()
search_pid=""
cleanup() {
    for monitor in "${monitors[@]}"; do kill "$monitor" 2>/dev/null || true; done
    for monitor in "${monitors[@]}"; do wait "$monitor" 2>/dev/null || true; done
    if [[ -n "$search_pid" ]]; then kill "$search_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${command[@]}" > "$out/search.log" 2>&1 &
search_pid=$!
echo "$search_pid" > "$out/search.pid"
echo "Search PID: $search_pid; progress: tail -f '$out/search.log'"
if command -v pidstat >/dev/null 2>&1; then
    LC_ALL=C pidstat -u -r -d -w -p "$search_pid" "$interval" > "$out/pidstat.log" 2>&1 &
    monitors+=("$!")
fi
if command -v vmstat >/dev/null 2>&1; then
    LC_ALL=C vmstat -t "$interval" > "$out/vmstat.log" 2>&1 &
    monitors+=("$!")
fi
status=0
wait "$search_pid" || status=$?
search_pid=""
printf '%s\n' "$status" > "$out/exit_code.txt"
{ date -u; free -k; } > "$out/host_after.txt" 2>&1
analysis_status=0
if [[ -s "$out/rounds.csv" ]]; then
    python3 "$repo/scripts/analyze_lvc_disk_rounds.py" "$out/rounds.csv" \
        > "$out/analysis.log" 2>&1 || analysis_status=$?
    cat "$out/analysis.log"
else
    echo "No completed rounds; inspect $out/search.log" >&2
    analysis_status=1
fi
[[ "$status" == 0 ]] || exit "$status"
exit "$analysis_status"
