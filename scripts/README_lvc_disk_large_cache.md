# BIGANN-1B large-cache footprint and query tests

Run from the DiskANN repository:

```bash
OUTPUT_DIR="$PWD/scripts/lvc_large_cache_results/bigann_1B_large" \
  nohup bash scripts/test_lvc_disk_large_cache_bigann_1B.sh \
  > test_lvc_disk_large_cache_bigann_1B_output.log 2>&1 &
```

This reuses the unified cache runner and executes exactly three cases in order:
DeXOR 100M (`chain_max=2`, 1% degree roots, 1% decode-state cache), ALP 200M,
and ALP 400M. It does not run the Cartesian product of codecs and node counts.
Cache reads and ALP/differential encoding use 32 threads. Search uses
`K=1, L=1,2,3,4,5,6,8,10, T=1, W=2, R=10`. Native disk PQ navigation is
preserved; no additional LVC candidate filtering or cache-budget cap is enabled.
Each case fully loads the selected cache, records its footprint, and queries the
full query file. No automatic node-count reduction is performed.

With an existing BIGANN-1M disk index, first validate the launcher and accounting:

```bash
bash scripts/test_lvc_disk_large_cache_smoke_bigann_1M.sh
```

The smoke test uses DeXOR 1024 / ALP 2048 / ALP 4096 cached nodes, two query
rounds, `L=1,2`, and native payload verification. Its footprint sums and explicit
case ordering are checked automatically; its verification timings are not
performance measurements.

## Outputs

Each case writes its own `<case>/summary.csv` and `<case>/footprint.csv`.
The root `summary.csv`, `footprint.csv`, and `status.json` are updated atomically
after every case, including failures. Successful cases also have `done.json`.
Logs/commands and native per-round outputs live in `<case>/attempt_<n>/`.
`rounds.csv` and per-round ID/distance bins retain all ten rounds for every L;
the summary uses rounds 2--10 uniformly, matching the earlier unified tests.

The new `cached_data_bytes` field is the exact sum of recorded component counters:

```
adjacency_bytes + payload_bytes + record_metadata_bytes + state_cache_bytes
```

`cached_data_gib` divides it by 2^30. It excludes lookup and global PQ.
`vector_record_bytes` is payload plus record metadata only. Individual counters,
lookup bucket counts, PQ code/table bytes, entry-point bytes, and
`accounted_index_bytes` (including lookup/PQ/state) are retained separately.
These are accounted resident structures after loading, not process RSS or peak
loading memory. Current/peak RSS, page faults and load/encoding timings are also
retained in the query summary. The native log prints footprint counters before
query execution; the runner validates and exports them after process exit. If a
query fails after counters were emitted, `status=failed` can coexist with
`footprint_status=ok`, preserving the completed footprint measurement. A failure
before the counters exist has `footprint_status=unavailable`.

ALP 400M still needs the complete lookup and cache to fit in actual server memory.
Excluding lookup from the plotted footprint does not remove its allocation. If a
case fails, its logs are preserved and subsequent cases continue; the final exit
code is nonzero. No timings or footprint values are estimated for missing runs.

## Resume and overrides

```bash
OUTPUT_DIR="$PWD/scripts/lvc_large_cache_results/bigann_1B_large" \
  RESUME=1 SKIP_BUILD=1 \
  bash scripts/test_lvc_disk_large_cache_bigann_1B.sh
```

Resume checks executable/data/config identities, skips successful cases, and
places retries in new attempt directories. Do not rebuild between resuming cases
unless you choose a new output directory. Set `SKIP_BUILD=1` only when the updated
binary is already built. The wrapper builds `search_disk_index` by default.

Preview commands without reading data or building:

```bash
DRY_RUN=1 bash scripts/test_lvc_disk_large_cache_bigann_1B.sh
```

Run a subset, or execute separate cases concurrently using distinct output dirs
only when server memory allows:

```bash
CASES="dexor:100000000" OUTPUT_DIR=/path/to/dexor100m \
  bash scripts/test_lvc_disk_large_cache_bigann_1B.sh
```

Other overrides are `BUILD_DIR`, `DATA_DIR`, `INDEX_PREFIX`, `QUERY_FILE`,
`GT_FILE`, `LOAD_THREADS`, `NUM_ROUNDS`, `BEAMWIDTH`, `L_VALUES`,
`EXPECTED_POINTS`, `BUILD_THREADS`, and `VERIFY_CACHE`.
`VERIFY_CACHE=1` adds payload verification overhead and is for correctness checks,
not the normal performance run. Existing GT/recall semantics are unchanged.

## ALP 400M / DeXOR 100M recheck with swap diagnostics

Use a new output directory; completed cases in an old directory would be reused:

```bash
OUTPUT_DIR="$PWD/scripts/lvc_large_cache_results/bigann_1B_swap_recheck" \
  nohup bash scripts/test_lvc_disk_large_cache_swap_recheck_bigann_1B.sh \
  > test_lvc_disk_large_cache_swap_recheck_bigann_1B_output.log 2>&1 &
```

The default order is ALP 400M, then DeXOR 100M, running sequentially. It uses the
same native load/search settings as the full-cache test. Both cases include the
existing footprint accounting and all query rounds. `CASES` can select only one
case. `MEMORY_SAMPLE_SECONDS=5` is the default; no sudo is needed.

Each `attempt_<n>` additionally contains:

- `memory_samples.csv`: flushed periodic snapshots through loading and queries.
  Target-process fields include VmRSS, VmHWM, VmSize, VmData, VmSwap, RssAnon,
  RssFile, thread count, and cumulative minor/major faults. Host fields include
  MemAvailable, SwapFree, SwapCached, and cumulative swap-in/out page counts.
- `memory_summary.json`: sampled peak RSS/Swap, minimum available host memory,
  sampling errors, and host swap-in/out byte deltas. These fields are also added
  to the case/root `summary.csv` (case-wide, repeated for each L).
- `memory_rounds.csv`: native per-round performance/fault counts joined to
  snapshots inside that round. Before/after snapshots are included separately
  for context; a round shorter than the interval may have zero inside snapshots.

Process VmSwap and major-fault counts describe the target; host swap-in/out can
include other processes. Periodic peaks can miss activity between snapshots.
The monitor does not reset swap, drop caches, trim the allocator, or alter node
counts/search behavior. It does not walk smaps on each sample. Monitoring has
some overhead, so use the same interval for comparisons. Native nonzero exits
retain sampled diagnostics, and the runner continues to the next case.

Optional BIGANN-1M validation:

```bash
MEMORY_MONITOR=1 MEMORY_SAMPLE_SECONDS=0.2 \
  bash scripts/test_lvc_disk_large_cache_smoke_bigann_1M.sh
```

For resuming the diagnostic run, set the same `OUTPUT_DIR` plus
`RESUME=1 SKIP_BUILD=1`. Do not change the code/binary or sampling configuration
between attempts within that output directory. ALP 400M fitting in an older
binary does not guarantee the current lookup layout and allocation peak fit;
the diagnostic launcher intentionally preserves the requested full cache.
