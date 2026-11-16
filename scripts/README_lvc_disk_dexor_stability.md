# DeXOR cache sensitivity: 50 measured rounds without decode profiling

Run after syncing the changed C++ sources and scripts to the **apply_differential**
checkout. This benchmarks DiskANN+DeXOR on that branch; it does not create a pure
`test_bigann` Raw DiskANN baseline.

```bash
nohup bash scripts/test_lvc_disk_dexor_stability_bigann_1B.sh \
  > test_lvc_disk_dexor_stability_bigann_1B_output.log 2>&1 &
```

Defaults: cache sizes **10M then 25M**, chain_max=2, degree-selected roots 1%,
root-state cache 1%, construction threads=32, query threads=1, K=1, W=2,
L=1,2,3,4,5,6,8,10. Each cache size loads once. For each L, the **entire query
file in its original order** runs three warmup passes, followed by 50 measured
passes. L order remains ascending, matching earlier rechecks. No additional LVC
two-level candidate filtering is introduced; DiskANN's normal PQ routing remains.

Choose an otherwise idle allowed physical CPU explicitly when possible:

```bash
SEARCH_CPU=12 CACHE_NODES_LIST="10000000 25000000 50000000 100000000" \
ROUNDS=50 WARMUP_ROUNDS=3 \
nohup bash scripts/test_lvc_disk_dexor_stability_bigann_1B.sh \
  > test_lvc_disk_dexor_stability_bigann_1B_output.log 2>&1 &
```

`SEARCH_CPU` defaults to the first allowed CPU, consistently across cache sizes;
this does not guarantee that CPU or its SMT sibling is idle. `SEARCH_CPU=-1`
disables binding. Only the query thread is bound, **after** parallel construction;
construction still uses its normal CPU allocation. NUMA placement is not changed.
The launcher refuses concurrent visible search processes unless
`ALLOW_OTHER_SEARCH=1`, and uses a per-user lock for copies of this launcher.
It does not stop unrelated jobs, change CPU governors, disable swap, or drop caches.

## What was found in the earlier experiments

- The 10M/25M launcher delegates to the 100M script. It fixes R=10, has no
  explicit full-query warmup, and runs `pidstat` and `vmstat` throughout the case.
  Neither old launcher invokes `perf`. Sampling monitors alone do not establish
  the cause of the previously observed variance.
- `decode_cache_vector` originally timed **every differential decode**, updated
  counters for bytes/records/cache hits, and accumulated decode microseconds in
  `QueryStats`. That measurement work was inside timed queries, even when no
  external profiler was attached. The new `--lvc_decode_profile 0` skips the
  per-decode clocks and counter collection; the default for old commands remains 1.
- Old per-round ID/distance files were written between passes. These writes were
  outside the query timer but could compete with subsequent queries for storage
  service. `--defer_round_results` now retains these small arrays in memory and
  writes them only after **all** L/round queries complete. With 10,000 queries,
  K=1, eight L values and 53 passes/L, their raw arrays occupy about 32.35 MiB.
  A crash preserves completed CSV rows but not deferred result bins.
- The new launcher does not start `perf`, `pidstat`, `vmstat`, `iostat` or any
  background monitoring process. It rejects common CPU/heap profiling environment
  variables. It cannot prevent another user from attaching a profiler later.
- Ordinary DiskANN `QueryStats` (per-query latency, I/O counts and normal I/O/CPU
  timers) remain enabled to measure latency and Mean I/Os. Therefore this is
  **detailed decode profiling disabled**, not a claim of zero instrumentation.
- Idle OpenMP construction workers use passive waiting (`OMP_WAIT_POLICY=PASSIVE`,
  `GOMP_SPINCOUNT=0`); BLAS threads=1. Release build is requested. Host snapshots,
  source/binary hashes and input size/mtime are recorded outside query runs.
  Inherited OpenMP placement/affinity overrides are cleared; explicit query CPU
  binding happens after construction.
- The current branch uses io_uring and attempts SQPOLL with fallback. The reader
  now logs `IO_BACKEND io_uring sqpoll=0/1` when registering a context, so backend
  differences between runs are visible. Query T=1 is not a promise of one total
  CPU, since SQPOLL may use a kernel worker.

## Outputs and interpretation

Under `scripts/lvc_disk_dexor_stability_results/bigann_1B_<UTC>/`:

- `dexor_C<nodes>/rounds.csv`: 424 rows by default, including 24 warmup rows
  (`phase=warmup`) and 400 measured rows (`phase=measurement`). Includes per-pass
  recall, QPS, mean/p999 query latency, Mean I/Os, RSS, query-phase process fault
  and context-switch deltas, start timestamp and CPU before/after. The disabled
  detailed decode fields are **zero because profiling is off**, not because no
  decoding occurred; `lvc_decode_profile=0` makes this explicit.
- Swap diagnostics are boundary snapshots, outside the query-set timer:
  `process_swap_kib` is this process's current swap residency;
  `system_swap_in_pages/out_pages` are **system-wide** deltas during the pass.
  Nonzero system swap cannot by itself attribute paging to this process.
  No per-query `/proc` reads are added.
- `summary.csv` per case and consolidated at the parent: measured-round means,
  median, range, CV, aggregate QPS, fault/swap totals and completeness.
  `aggregate_qps=sum(queries)/sum(wall_seconds)`; `qps_mean` is arithmetic mean
  of per-round QPS. `mean_us` is the mean of per-pass mean query latencies.
- `blocks.csv`: rounds 1–10, 11–20, ..., 41–50 summarized separately. Inspect
  differences between blocks for time drift; 50 adjacent query passes share one
  load and are not 50 independently built indexes.
- `warmup_rounds.csv` and `rounds_flagged.csv`: warmups retained separately and
  measured rounds flagged for unusually slow latency. No measured rows are
  automatically excluded, including measured round 1. A secondary mean excluding
  measured round 1 is labeled diagnostic only.
- `search_L<L>_R<round>_{idx_uint32,dists_float}.bin`: every measured pass;
  `search_L<L>_warmup_R<round>_...`: every warmup pass. Query IDs/distances remain
  available for correctness and recall checks.
- `footprint.csv`: stable cache components (adjacency, payload, metadata, state
  cache), lookup bytes and sums with/without lookup. These exclude the PQ matrix
  and temporary allocations, and are also copied into the parent summary.
- `search.log`: cache hash, `CACHE_BUDGET` block sizes, forest/encoding/load
  timings, actual io_uring mode, ordinary search summaries.
- `command.sh`, `host_before/after.txt`, `code_sha256.txt`, `input_identity.txt`,
  exit status and analysis log.

To analyze again without loading the index:

```bash
python3 scripts/analyze_lvc_disk_rounds.py /path/to/case/rounds.csv --expected-rounds 50
```

Fifty rounds improve visibility, not a guarantee of low variance. Increasing
cache nodes can reduce I/O while increasing decode work, memory latency or paging.
Use measured means/medians and blocks, together with identical recall/I/O results,
major faults and swap diagnostics. Do not delete slow rounds just to obtain a
monotone cache-size/latency curve. Warmup results describe startup separately.
If 100M still swaps, reduce memory pressure or run it separately on a quiet host.

## Controls and a small correctness check

`ROUNDS`, `WARMUP_ROUNDS`, `CACHE_NODES_LIST`, `L_VALUES`, `SEARCH_CPU`,
`BEAMWIDTH`, `LVC_BUILD_THREADS`, `BUILD_THREADS`, `BUILD_DIR`, `DATA_DIR`, `INDEX_PREFIX`,
`QUERY_FILE`, `GT_FILE`, `OUTPUT_DIR`, `SKIP_BUILD` are configurable.
`LVC_DECODE_PROFILE=1` is available for a separate on/off diagnostic; do not mix
its performance numbers with the default unprofiled results.

```bash
DRY_RUN=1 OUTPUT_DIR=/tmp/dexor_stability_preview \
  bash scripts/test_lvc_disk_dexor_stability_bigann_1B.sh

# On the server with an existing BIGANN-1M disk index and GT:
bash scripts/test_lvc_disk_dexor_stability_smoke_bigann_1M.sh
```

The smoke script runs profiling off/on with identical small cache/query settings
(W=1 to avoid asynchronous completion-order differences)
and compares every round's IDs/distances bitwise, including the final compatibility
bins. It checks warmup exclusion and zero versus positive decode profiling counters.
It requires real input files, not a synthetic benchmark substitute.
