#!/usr/bin/env bash
# Fixed-order DeXOR cache sensitivity: unprofiled search, warmup + 50 measured passes/L.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD_DIR:-"$repo/build"}
data=${DATA_DIR:-"$build/data/bigann"}
index=${INDEX_PREFIX:-"$data/disk_index_bigann_1B_R32_L50_A1.2"}
query=${QUERY_FILE:-"$data/bigann_query.fbin"}
gt=${GT_FILE:-"$data/bigann_1B_gt.bin"}
out=${OUTPUT_DIR:-"$repo/scripts/lvc_disk_dexor_stability_results/bigann_1B_$(date -u +%Y%m%dT%H%M%SZ)"}
rounds=${ROUNDS:-50}
warmups=${WARMUP_ROUNDS:-3}
checkpoint_rounds=${CHECKPOINT_ROUNDS:-0}
load_threads=${LVC_BUILD_THREADS:-32}
profile=${LVC_DECODE_PROFILE:-0}
beamwidth=${BEAMWIDTH:-2}
read -r -a counts <<< "${CACHE_NODES_LIST:-10000000 25000000 50000000}"
read -r -a levels <<< "${L_VALUES:-1 2 3 4 5 6 8 10}"
# Same query CPU across cache sizes. Override with a quiet, allowed physical CPU.
search_cpu=${SEARCH_CPU:-$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')}
python3 - "$rounds" "$warmups" "$load_threads" "$profile" "$search_cpu" "${counts[*]}" "${levels[*]}" "$beamwidth" "$checkpoint_rounds" <<'PY'
import os, sys
r, w, t, p, cpu = map(int, sys.argv[1:6])
counts, levels = [list(map(int, value.split())) for value in sys.argv[6:8]]
if not 0 < int(sys.argv[8]) < 2**32:
    raise SystemExit('BEAMWIDTH must be a positive integer')
if not (0 < r < 2**32 and 0 <= w < 2**32-r and t > 0 and p in (0, 1)):
    raise SystemExit('Invalid rounds, warmup, thread count or profile setting')
if not 0 <= int(sys.argv[9]) < r:
    raise SystemExit('CHECKPOINT_ROUNDS must be 0 or a positive integer less than ROUNDS')
if not counts or len(set(counts)) != len(counts) or min(counts) < 1 or max(counts) > 500000000:
    raise SystemExit('CACHE_NODES_LIST must contain distinct counts in [1, 500000000]')
if not levels or len(set(levels)) != len(levels) or min(levels) < 1:
    raise SystemExit('L_VALUES must contain distinct positive values')
if cpu != -1 and (cpu not in os.sched_getaffinity(0) or cpu >= 1024):
    raise SystemExit('SEARCH_CPU is not available (use -1 to leave it unbound)')
for key in ('CPUPROFILE', 'HEAPPROFILE', 'HEAPCHECK'):
    if os.environ.get(key):
        raise SystemExit('Unset profiling environment variable: '+key)
if 'profiler' in os.environ.get('LD_PRELOAD', '').lower():
    raise SystemExit('Remove profiler from LD_PRELOAD before benchmarking')
PY
mkdir -p "$out"
out=$(cd -- "$out" && pwd)
[[ ! -e "$out/run_config.txt" ]] || { echo "Existing run: choose a fresh OUTPUT_DIR" >&2; exit 1; }
printf 'rounds=%s\nwarmup_rounds=%s\ncache_nodes=%s\nL=%s\nsearch_cpu=%s\nlvc_decode_profile=%s\n' \
    "$rounds" "$warmups" "${counts[*]}" "${levels[*]}" "$search_cpu" "$profile" > "$out/run_config.txt"
checkpoint_args=()
if (( checkpoint_rounds > 0 )); then
    checkpoint_args=(--round_checkpoint "$checkpoint_rounds")
    printf 'checkpoint_rounds=%s\n' "$checkpoint_rounds" >> "$out/run_config.txt"
fi
# No active spinning by idle construction workers; ordinary QueryStats remain enabled.
export OMP_NUM_THREADS="$load_threads" OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
export OMP_PROC_BIND=FALSE
unset GOMP_CPU_AFFINITY OMP_PLACES
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
# A per-user lock prevents copies of this launcher from competing.
exec 9>"${TMPDIR:-/tmp}/lvc_dexor_stability_${UID}.lock"
flock -n 9 || { echo "Another DeXOR stability run is active for this user" >&2; exit 1; }
if [[ ${SKIP_BUILD:-0} != 1 && ${DRY_RUN:-0} != 1 ]]; then
    cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release > "$out/build.log" 2>&1
    cmake --build "$build" --target search_disk_index --parallel "${BUILD_THREADS:-4}" >> "$out/build.log" 2>&1
fi
if [[ ${DRY_RUN:-0} != 1 ]]; then
    for file in "${index}_disk.index" "${index}_pq_compressed.bin" "${index}_pq_pivots.bin" "$query" "$gt"; do
        [[ -r "$file" ]] || { echo "Missing input: $file" >&2; exit 1; }
    done
    "$build/apps/search_disk_index" --help > "$out/binary_help.txt" 2>&1
    for flag in lvc_decode_profile warmup_rounds defer_round_results search_cpu per_round_csv; do
        grep -q -- "--$flag" "$out/binary_help.txt" || { echo "Rebuild updated binary: missing --$flag" >&2; exit 1; }
    done
    if (( checkpoint_rounds > 0 )) && ! grep -q -- '--round_checkpoint' "$out/binary_help.txt"; then
        echo 'Rebuild updated binary: missing --round_checkpoint' >&2
        exit 1
    fi
    if [[ ${ALLOW_OTHER_SEARCH:-0} != 1 ]]; then
        python3 - <<'PY'
from pathlib import Path
found = []
for path in Path('/proc').glob('[0-9]*/comm'):
    try:
        if path.read_text().strip().startswith(('search_disk_ind', 'search_memory_i')):
            found.append(path.parent.name)
    except OSError:
        pass
if found:
    raise SystemExit('Other search processes detected (PIDs '+','.join(found)+'); run benchmarks serially')
PY
    fi
fi
snapshot() {
    # Snapshots happen between cases only, never a background sampler during queries.
    {
        date -u
        uname -a
        git -C "$repo" rev-parse HEAD
        git -C "$repo" diff --stat
        lscpu
        free -k
        cat /proc/meminfo /proc/vmstat /proc/pressure/memory /proc/pressure/io
        ps -eo pid,comm,pcpu,pmem --sort=-pcpu | head -n 20 || true
        for item in /sys/devices/system/cpu/cpu"$search_cpu"/cpufreq/scaling_governor \
                    /sys/kernel/mm/transparent_hugepage/enabled /proc/sys/vm/swappiness; do
            if [[ -r "$item" ]]; then echo "$item"; cat "$item"; fi
        done
        printenv | LC_ALL=C sort | grep -E '^(OMP_|GOMP_|KMP_|MKL_|OPENBLAS_|BLIS_|LD_PRELOAD=|MALLOC_|TCMALLOC_)' || true
    } > "$1" 2>&1
}
search_pid=""
cleanup() {
    if [[ -n "$search_pid" ]]; then
        kill "$search_pid" 2>/dev/null || true
        wait "$search_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for nodes in "${counts[@]}"; do
    case_dir="$out/dexor_C${nodes}"
    mkdir "$case_dir"
    command=("$build/apps/search_disk_index"
        --data_type float --dist_fn l2 --index_path_prefix "$index"
        --query_file "$query" --gt_file "$gt" --cache_codec dexor
        --num_nodes_to_cache "$nodes" --lvc_chain_max 2
        --lvc_build_threads "$load_threads" --cache_load_threads 1 --lvc_fallback_root_candidates 1
        --lvc_state_cache_ratio 0.01 --cache_budget_include_state
        --lvc_decode_profile "$profile" --warmup_rounds "$warmups" --defer_round_results
        --search_cpu "$search_cpu" -K 1 -L "${levels[@]}" -T 1 -W "$beamwidth" -R "$rounds"
        --result_path "$case_dir/search" --per_round_csv "$case_dir/rounds.csv" "${checkpoint_args[@]}")
    {
        printf '#!/usr/bin/env bash\n'
        printf 'export OMP_NUM_THREADS=%q OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0\n' "$load_threads"
        printf 'export OMP_PROC_BIND=FALSE\nunset GOMP_CPU_AFFINITY OMP_PLACES\n'
        printf 'export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1\n'
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "$case_dir/command.sh"
    echo "DeXOR C=$nodes: warmup=$warmups, measured rounds=$rounds/L, CPU=$search_cpu, profiling=$profile"
    if [[ ${DRY_RUN:-0} == 1 ]]; then cat "$case_dir/command.sh"; continue; fi
    snapshot "$case_dir/host_before.txt"
    # Preserve binary/source and input identities, without hashing the 32-GB PQ matrix.
    sha256sum "$build/apps/search_disk_index" "$repo/apps/search_disk_index.cpp" \
        "$repo/src/pq_flash_index.cpp" "$repo/include/pq_flash_index.h" \
        "$repo/src/linux_aligned_file_reader.cpp" > "$case_dir/code_sha256.txt"
    stat --printf='%n size=%s mtime=%y\n' "${index}_disk.index" "${index}_pq_compressed.bin" \
        "${index}_pq_pivots.bin" "$query" "$gt" > "$case_dir/input_identity.txt"
    "${command[@]}" > "$case_dir/search.log" 2>&1 &
    search_pid=$!
    echo "$search_pid" > "$case_dir/search.pid"
    status=0
    wait "$search_pid" || status=$?
    search_pid=""
    printf '%s\n' "$status" > "$case_dir/exit_code.txt"
    snapshot "$case_dir/host_after.txt"
    analysis_status=0
    if [[ -s "$case_dir/rounds.csv" ]]; then
        python3 "$repo/scripts/analyze_lvc_disk_rounds.py" "$case_dir/rounds.csv" \
            --expected-rounds "$rounds" --L "${levels[@]}" > "$case_dir/analysis.log" 2>&1 || analysis_status=$?
        cat "$case_dir/analysis.log"
    else
        echo "No completed rounds; inspect $case_dir/search.log" >&2
        analysis_status=1
    fi
    python3 - "$out" <<'PY'
import csv, re, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for path in sorted(out.glob('dexor_C*/summary.csv')):
    footprint = {}
    log = (path.parent/'search.log').read_text()
    match = re.search(r'^CACHE_BUDGET (.*)$', log, re.MULTILINE)
    if match:
        values = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', match.group(1)))
        footprint = dict(graph_bytes=values['graph_bytes']-values['graph_lookup_bytes'],
                         lookup_bytes=values['graph_lookup_bytes'],
                         payload_bytes=values['vector_payload_bytes'],
                         metadata_bytes=values['vector_metadata_bytes'],
                         state_cache_bytes=values['state_cache_bytes'])
        footprint['cached_footprint_without_lookup_bytes'] = sum(
            value for key, value in footprint.items() if key != 'lookup_bytes')
        footprint['cached_footprint_with_lookup_bytes'] = values['total_bytes']
        with (path.parent/'footprint.csv').open('w', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=list(footprint))
            writer.writeheader()
            writer.writerow(footprint)
    with path.open(newline='') as source:
        rows.extend(dict(cache_nodes=int(path.parent.name.split('C')[1]), **footprint, **row)
                    for row in csv.DictReader(source))
if rows:
    with (out/'summary.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
PY
    [[ "$status" == 0 ]] || exit "$status"
    [[ "$analysis_status" == 0 ]] || exit "$analysis_status"
done
echo "Results: $out"
