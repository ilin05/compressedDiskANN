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

Both scripts use K=1, L=1,2,3,4,5,6,8,10, 10 rounds, one thread, and 1% root state
cache. No extra candidate-filtering optimization is enabled; DiskANN's native PQ
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
budget does not eliminate those costs. No forest algorithm has been changed.

Environment overrides: BUILD_DIR, DATA_DIR, INDEX_PREFIX, QUERY_FILE, GT_FILE,
OUTPUT_DIR, SAMPLE_NODES, BUDGET_MIB, PLANNING_BUDGET_MIB, STATE_CACHE_RATIO,
ROUNDS, BUILD_THREADS, SKIP_BUILD, RESUME; 1B also PERF_W and VERIFY_CACHE.
