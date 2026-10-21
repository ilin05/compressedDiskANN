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
are added inside node, coordinate or distance loops. The forest algorithm is
unchanged.

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
`PLANNING_BUDGET_MIB`. Build parallelism does not change measured run threads.
Each invocation runs afresh and overwrites results in its output directory;
use different `OUTPUT_DIR` values to retain repeats. A binary without the new
timing instrumentation is reported as a failed case, never as zero time.
