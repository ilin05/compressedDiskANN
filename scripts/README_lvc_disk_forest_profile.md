# BIGANN-1M DeXOR forest/encoding profile

Run from the DiskANN repository on the server:

```bash
bash scripts/profile_lvc_disk_forest_bigann_1M.sh
```

The default inputs are the existing
`build/data/bigann/disk_index_bigann_1M_R32_L50_A1.2` index and
`build/data/bigann/bigann_query.fbin`. No full-precision source fbin or GT is
required: cached vectors and adjacency are read from the existing disk index.

Defaults:

- Single thread (`-T 1`, OMP/BLAS/MKL thread counts all set to 1).
- Shared 1,000-node sample using the existing budget sampler. This sampler
  evaluates all codecs, but the measured production cases are only DeXOR
  `chain_max=2` and `chain_max=-1` (unbounded).
- Nominal budget 20 MiB, planning budget 19.5 MiB. Counted components are
  adjacency slots, vector payload, record metadata and decoding state cache.
  PQ and the node lookup tables are excluded, matching the payload-budget suite.
- Roots 1%, decoding state cache 1%; each chain configuration estimates its own
  node count. This profiles the existing budget experiment, not equal-node-count
  scaling between chains.
- One construction per case, no retries. An over-budget result is saved as
  `over_budget`; its timings remain usable.
- Every cached vector is verified bitwise. Query search is skipped with
  `--cache_budget_probe_only`; no recall/QPS experiment is run here.

Results are written to
`scripts/lvc_disk_forest_profile_results/bigann_1M/`:

- `summary.csv`: two production cases, timings and percentage breakdown.
- `dexor_ch2_timings.csv`, `dexor_chunbounded_timings.csv`: saved immediately
  after each case.
- `dexor_ch2.log`, `dexor_chunbounded.log`: complete production logs, including
  `LVC_BUILD_TIMING`, forest statistics, payload ratios and bitwise verification.
- `sample.csv`, `sample.log`: separate budget sampling results.
- `*.timing.json`: subprocess commands, exit status and wall time.
- `config.json`: configuration and budget definition.

## Timing boundaries

Timers use `steady_clock`/wall time and run only at stage boundaries; no timers
are added inside node, coordinate or distance loops. Timing instrumentation does
not affect parent selection; the selected fallback policy is described below.

| Column | Included work |
|---|---|
| `root_selection_seconds` | Forest/root scratch allocation, degree sort and root initialization |
| `propagation_seconds` | All propagation rounds; deep-chain visited initialization |
| `fallback_seconds` | All fallback candidate scans and assignments, including the outer scan |
| `finalize_seconds` | Final parent/depth materialization for chain=2, and forest validation |
| `forest_seconds` | Complete `build_forest` call, including small timing overhead |
| `encode_seconds` | Complete RecordStore build: validation/copy, ordering, parent-state replay, encoding, patch validation and record appends |
| `forest_plus_encode_seconds` | `forest_seconds + encode_seconds`; denominator for all stage percentages |
| `process_wall_seconds` | Entire measured subprocess: index/PQ load, BFS, cache I/O/remapping, forest, encoding, state cache, full verification and cleanup |

Encoding excludes the subsequent `shrink_to_fit` and root state-cache creation.
Encoding now includes a temporary canonical root-state cache used by all child
encoders. It is released when `RecordStore::build` returns; it is separate from
the retained query state cache. Records are encoded by depth in batches of 4096,
then appended in the original depth/ID order. This preserves serial bytes while
bounding temporary record memory. Peak construction memory can exceed the final
payload budget, which does not constrain temporary state/raw-vector buffers.
Sampling is a separate subprocess and is excluded from production timing rows.
The difference between whole-process time and forest+encoding is **not** pure I/O.

The script rebuilds `search_disk_index` by default. For subsequent runs with
the updated binary already built:

```bash
SKIP_BUILD=1 OUTPUT_DIR=/path/to/new/profile_run \
  bash scripts/profile_lvc_disk_forest_bigann_1M.sh
```

Other supported environment variables are `BUILD_DIR`, `BUILD_THREADS`,
`DATA_DIR`, `INDEX_PREFIX`, `QUERY_FILE`, `SAMPLE_NODES`, `BUDGET_MIB`, and
`PLANNING_BUDGET_MIB`, and `LVC_BUILD_THREADS` (default 1 for the 1M wrapper).
Build parallelism does not change measured run threads.
Each invocation runs afresh and overwrites results in its output directory;
use different `OUTPUT_DIR` values to retain repeats. A binary without the new
timing instrumentation is reported as a failed case, never as zero time.

## Parallel BIGANN-1B / 1 GiB profile

First run the small, dataset-independent correctness regression:

```bash
bash scripts/test_lvc_parallel_build.sh
```

It checks chains 2/4/8/unbounded, serial vs 32-thread forests, all three codec
payloads against the original serial encoding, distance ties, disconnected
fallback nodes, batch boundaries, bitwise recovery and encoding worker errors.
No extra CMake target is needed; the script compiles the standalone test.

Then profile the existing BIGANN-1B disk index:

```bash
nohup bash scripts/profile_lvc_disk_forest_bigann_1B.sh \
  > profile_lvc_disk_forest_bigann_1B_output.log 2>&1 &
```

Defaults: 1 GiB nominal / 0.975 GiB planning budget, 1M sampled nodes, DeXOR
chains 2 and unbounded, root/state ratios 1%, up to 32 LVC build threads.
The same budget exclusions and full cache bitwise checks as above apply. No
query search is run. Output is
`scripts/lvc_disk_forest_profile_results/bigann_1B_1GiB/`.
The original 1B PQ table (~32 GB) must still be loaded in each subprocess;
1 GiB refers to the cache budget, not process RSS.

Change `lvc_build_threads` in the 1B shell script or set it at launch:

```bash
LVC_BUILD_THREADS=8 bash scripts/profile_lvc_disk_forest_bigann_1B.sh
```

`--lvc_build_threads` is independent of query `-T` (kept at 1 in the profiler).
The CLI defaults to 32; other library callers retain a one-thread default.
The maximum is configurable; OpenMP runtime limits can yield fewer workers.
Propagation remains serial. Chain=2 fallback uses an ascending-ID root list and
parallelizes independent unresolved nodes. Bounded deep chains and the legacy
unbounded policy keep assignment order serial, parallelizing candidate distances
with deterministic distance/ID tie breaking using one persistent team. The new
default unbounded policy is nearest-root-local, described below.

Additional `summary.csv` timing columns:

- `index_load_seconds`: index/PQ load and thread-context setup.
- `cache_selection_seconds`: BFS, cache ID sort and hash.
- `cache_read_seconds`: allocation, disk reads, lookup inserts and vector copies.
- `remap_seconds`: construction of cached-node local adjacency lists.
- `compact_seconds`: compressed payload shrink-to-fit.
- `state_seconds`: retained query root-state cache creation.
- `verify_seconds`: full cache bitwise correctness check.
- `cache_other_seconds`: forest statistics/logging and local cleanup inside the
  measured cache function, excluding the named stages.
- `cache_return_seconds`: function-return cleanup and caller timing overhead.
- `load_other_seconds`: remaining caller overhead between measured load stages.
- `total_load_seconds`: complete index load plus cache selection/construction,
  including verification but excluding process startup and final destruction.
- `load_without_verification_seconds`: total load minus correctness checking.
- `process_other_seconds`: subprocess wall time minus total load, including
  query-file loading, startup and final destruction; it is not pure I/O.

Each stage has a corresponding `*_load_percent`, relative to total load.
The disjoint load breakdown sums index_load, cache_selection, cache_read,
remap, **forest**, encode, compact, state, verify, cache_other, cache_return,
load_other. Root selection/propagation/fallback/finalize percentages are
subdivisions of forest; do not add them a second time. Existing `*_percent`
columns still use forest+encoding as their denominator.

For a thread scaling comparison, use separate `OUTPUT_DIR` paths and identical
input/sample settings with `LVC_BUILD_THREADS=1` and `32`. Both runs use the
optimized root-state reuse, so this measures thread scaling, not the benefit of
root-state reuse alone. These profiles report measured results, not a guaranteed
32x speedup. An over-budget attempt is recorded without an automatic retry.

## Nearest-root-local unbounded fallback (k=1)

For the DeXOR BIGANN-1B unbounded-only test, run:

```bash
bash scripts/test_lvc_parallel_build.sh
nohup bash scripts/profile_lvc_disk_forest_bigann_1B_root_local.sh \
  > profile_lvc_disk_forest_bigann_1B_root_local_output.log 2>&1 &
```

Default settings: 1 GiB cache budget, 0.975 GiB planning budget, 1M-node sample,
32 build threads, degree-selected 1% initial roots, 1% retained root state cache.
The production run tests only DeXOR `chain_max=-1`, verifies every cache vector
bitwise, and exits before query search. Sampling still uses the shared sampler
for all codecs and both chains. Both sampling and production explicitly use
the same new policy; samples are recomputed on each invocation.

Results use a separate directory, preserving previous global-fallback results:

```text
scripts/lvc_disk_forest_profile_results/bigann_1B_1GiB_root_local/
```

The algorithm changes only unbounded fallback:

1. Keep the existing initial roots and graph-neighbor propagation.
2. Group propagated nodes by their predecessor-tree root.
3. Find the exact nearest initial root for each unresolved node; equal distances
   select the smallest root ID.
4. Within that root's tree, select the nearest already assigned member. This
   includes earlier fallback nodes in the same bucket. Equal predecessor
   distances select the smallest node ID.
5. Append the newly assigned node to its root bucket.

Root search is parallel across unresolved nodes. Tree search is parallel across
root buckets, with ascending node ID order inside each bucket. Different buckets
are independent, giving the same forest as serial ascending-ID execution of
this policy. Only assigned nodes become parents, preventing cycles.

This is predecessor-tree membership, **not** reachability along outgoing cache
graph edges. It changes the former global nearest-predecessor guarantee, so
compression ratio, reference distances and depths may change. Query decoding
and full F32 lossless restoration retain their existing semantics.

`search_disk_index --lvc_fallback_root_candidates 1` is now the DiskANN default;
`0` selects the original global fallback for comparisons. This first version
supports only 0 or 1 and ignores the setting for bounded chains. The low-level
`lvc::build_forest` default remains 0 to preserve existing Vamana callers;
DiskANN sample/load callers explicitly pass their selected policy.

New report fields (also present in `LVC_BUILD_TIMING`):

- `fallback_root_candidates`: policy, checked against sample metadata.
- `fallback_bucket_seconds`: tree membership grouping and pending-node grouping.
- `fallback_root_search_seconds`: exact nearest-root search.
- `fallback_tree_search_seconds`: sequential search inside each tree, parallel
  across trees.
- `fallback_root_distances`, `fallback_tree_distances`: actual number of
  candidate L2 evaluations in successful root/tree searches.
- `fallback_max_tree_nodes`: largest final root-tree membership count.
- `fallback_mean_tree_candidates`: tree distance evaluations / fallback nodes.
- `payload_ratio_f32`, `average_depth`, `max_depth`, `forest_hash`: production
  compression/depth statistics for comparison with prior runs.

The three fallback sub-stage times are part of `fallback_seconds`, not additional
load stages. Candidate counts avoid per-distance timing calls. Root search costs
`F * R` distances; tree search depends on bucket sizes and load skew. There is
no guaranteed bound on maximum bucket size. Transient membership lists and raw
vectors are outside the retained cache budget.

The standalone correctness test includes an independent serial k=1 oracle,
one/32-thread equality, exact candidate counters, a case where k=1 must reject
the globally nearest predecessor, root-distance ties, and lossless encoding for
all three codecs. Python driver fixtures check unbounded-only execution and
policy propagation; they are not ANN performance measurements.
