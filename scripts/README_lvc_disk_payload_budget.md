# BIGANN cache budget excluding PQ and node lookups

Run the small correctness test first, then the billion-scale experiment:

```bash
bash scripts/test_lvc_disk_payload_budget_bigann_1M.sh
nohup bash scripts/test_lvc_disk_payload_budget_bigann_1B.sh > test_lvc_disk_payload_budget_bigann_1B.log 2>&1 &
```

Existing sampled/full-cache-budget scripts are unchanged. These new scripts use
MiB/GiB (powers of 1024): 1M nominal **20 MiB**, planning **19.5 MiB**;
1B nominal **20 GiB**, planning **19.5 GiB**.

## What is counted

```
budget = fixed adjacency slots + vector payload capacity
       + differential record metadata capacity + decoding state cache resident estimate
       = CACHE_BUDGET.total_bytes - CACHE_BUDGET.graph_lookup_bytes
```

Both node lookup hash tables and PQ are excluded, but their byte sizes and RSS
are reported. State cache includes its auxiliary map estimate; only the two
node-to-neighborhood/coordinate maps are excluded. Temporary BFS/forest/raw-vector
buffers and allocator overhead are not part of this structural steady-state
budget. This is not an equal-total-RAM comparison or a peak-RSS limit.

The shared sample is 1K nodes on 1M and 1M nodes on 1B. Payload, metadata and
state-cache sample averages are extrapolated; adjacency uses its exact fixed
per-node stride. Lookup capacities do NOT enter selection. A binary search in
arithmetic selects the maximum predicted node count under the planning budget,
subject to the production 50% cache cap. Native BFS(N) and sorted insertion are
used; the lookup-only ID list/curve is not required. Sample forests are shared
across the differential codecs for each chain condition; final forests are
rebuilt for each case's selected node set.

## Order and one-attempt policy

After shared sample profiling:

1. Raw DiskANN.
2. ALP.
3. DeXOR, Gorilla, Elf with chain_max=2.
4. DeXOR, Gorilla, Elf with unbounded chains.

Each case executes once. There is no hard cache cap passed to the search binary,
no capacity correction and no automatic retry. A successfully searched case is
recorded as `ok` or `over_budget`; over-budget results retain all performance
measurements. A crash/validation failure is recorded as `failed` (with whatever
measurements are available), then later cases still run. The driver exits
nonzero after all cases if any failed. Nominal budget excess alone is not a
failure.

Completed attempts, including failures, are reused with RESUME=1 (default) when
input/config/binary signatures match. To rerun, use RESUME=0 or remove only the
selected case's `.done.json`. Binary changes invalidate prior signatures.

Both scripts use K=1, L=1,2,3,4,5,6,8,10 and 1% root state cache. The 1M script
keeps 10 rounds and one build/search thread. The 1B script now defaults to
**32 LVC build threads, one query thread, and 3 rounds**. Build thread count is
explicitly passed independently from query `-T`; propagation and cache reading
retain their existing implementation. No extra candidate-filtering optimization is enabled; DiskANN's native PQ
navigation remains. 1M uses beamwidth=1 and checks all differential cache vectors
bitwise plus all codecs' final IDs/L2 bits against Raw. 1B uses beamwidth=2 and
does not require identical result bits across asynchronous search executions.

## Results

Default folders: `scripts/lvc_disk_payload_budget_results/bigann_1M` and
`bigann_1B`.

- `sample.csv`: measured shared sample bytes and costs.
- `node_plans.csv`: planning arithmetic, written before each case.
- `raw_results.csv`, `alp_results.csv`, `dexor_ch2_results.csv`, etc.: each case
  saved immediately, with one row per L (or a failure row if no metrics exist).
- `summary.csv`: cumulative rows, refreshed after each case.
- `CASE_attempt0.log`, `.timing.json`: live subprocess log and wall time.
- `CASE_search_summary.csv`: original search metrics from C++.
- `CASE.done.json`, `run_status.json`: configuration signatures and completion.
- `config.json`: budget, codec order, build/search threads, search parameters,
  input paths and fallback policy.

Each result records nominal/planning/actual budget, budget utilization,
within-budget flag, planned/actual nodes, every counted component, excluded
lookup bytes/buckets, PQ file bytes, RSS, forest stats, Recall@1 (percentage),
QPS, mean/p999 latency, mean IOs and timings:

| Field | Scope |
|---|---|
| index_load_seconds | Disk index load, PQ/centroids/scratch initialization; excludes query/GT reads |
| cache_selection_seconds | BFS, sort and cache-list hash |
| cache_data_load_seconds | Cache loading, lookup/adjacency setup, forest, encoding, state cache and optional verification |
| cache_load_seconds | Selection plus cache data load |
| forest_build_seconds | build_forest only, including root selection/propagation/fallback; Raw/ALP=0 |
| total_load_seconds | Index load through cache load completion; includes forest; excludes queries |
| process_wall_seconds | Entire subprocess, including queries and output |

Forest time is a subset of cache load, not an additional term to add to total.
The shared sample profiling is a separate preparation cost in sample timing/log;
it is not silently charged once per final codec. Loading, forest construction
and allocator high-water RSS may still be large/slow: excluding lookup from the
budget does not eliminate those costs. The current unbounded fallback policy
is documented below.

## Current 20 GiB BIGANN-1B full run

```bash
nohup bash scripts/test_lvc_disk_payload_budget_bigann_1B.sh \
  > test_lvc_disk_payload_budget_bigann_1B_output.log 2>&1 &
```

The script rebuilds `search_disk_index` by default, samples 1M cached nodes once,
then executes exactly this order:

```text
raw
alp
dexor_ch2
gorilla_ch2
elf_ch2
dexor_chunbounded
gorilla_chunbounded
elf_chunbounded
```

Each production group loads once and runs all eight L values for three rounds:
`-K 1 -L 1 2 3 4 5 6 8 10 -T 1 -R 3 -W 2`. The nominal budget is 20 GiB;
planning uses 19.5 GiB. Initial roots are degree-selected 1%; query state cache
defaults to 1%. Raw/ALP do not use differential forest/state caches.

Unbounded differential fallback uses the current **nearest-root tree k=1**
policy, explicitly passed to both sampling and production. Bounded chain=2
retains its root-child policy. Root state reuse and parallel fallback/encoding
are enabled by the current backend. Other scripts keep their existing behavior
unless supplied the new parameters.

After every group, `CASE_results.csv` and cumulative `summary.csv` are written
with atomic replacement, before the next group starts. A normal complete run
contains 64 summary rows (8 groups x 8 L values). Original application reports,
result binaries, full logs and subprocess timing JSON are retained. Failed
groups save a failure row or available partial rows and execution continues;
over-budget groups still search and are marked `over_budget`, without retries.

The 1B wrapper requires the current phase/policy metadata, so a stale binary
cannot silently substitute older forest logic. Completed matching attempts are
reused with `RESUME=1`; changed build threads, policy, search settings or binary
invalidate the corresponding signatures. Use `RESUME=0` for a fresh run.

Additional CSV columns include `build_threads`, `fallback_root_candidates`,
`root_selection_seconds`, `propagation_seconds`, `fallback_seconds`,
`finalize_seconds`, `encode_seconds`, nearest-root/bucket search times and
distance counts, cache-read/remap/state-cache/verification times, and
`load_without_verification_seconds`. `propagation_rounds` is the forest iteration
count; **`rounds` always means query repetitions** (3 by default on 1B).
Load timing repeats across a group's L rows and should be counted once per group.

By default `VERIFY_CACHE=0` avoids a full extra decoding pass during performance
measurement. Set `VERIFY_CACHE=1` to include it, timed separately. Set
`LVC_BUILD_THREADS=8` to adjust the build maximum; query threads remain 1.

Local fixture checks (no ANN performance measurements):

```bash
python3 scripts/test_lvc_disk_payload_budget.py
python3 scripts/test_lvc_disk_payload_budget_1B.py
```

Environment overrides: BUILD_DIR, DATA_DIR, INDEX_PREFIX, QUERY_FILE, GT_FILE,
OUTPUT_DIR, SAMPLE_NODES, BUDGET_MIB, PLANNING_BUDGET_MIB, STATE_CACHE_RATIO,
ROUNDS, BUILD_THREADS, SKIP_BUILD, RESUME; 1B also LVC_BUILD_THREADS, PERF_W and VERIFY_CACHE.
