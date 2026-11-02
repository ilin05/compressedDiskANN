# BIGANN-1B DeXOR 100M cache recheck

Run from the DiskANN repository after updating `apps/search_disk_index.cpp`:

```bash
nohup bash scripts/test_lvc_disk_dexor_100M_recheck_bigann_1B.sh \
  > test_lvc_disk_dexor_100M_recheck.log 2>&1 &
```

The script builds `search_disk_index`, loads the index/cache **once**, and runs
the **entire query file** ten times for each L, in order:
`L=1,2,3,4,5,6,8,10`, then `round=1,...,10` within each L.
There is no extra warmup or automatic exclusion of the first round.
The default is DeXOR, 100,000,000 BFS cache nodes, chain_max=2, degree roots 1%,
state cache 1%, load threads=32, query threads=1, K=1, beamwidth=2.
There is no additional LVC candidate filtering or hard memory cap.

Each run creates a fresh timestamped directory under
`scripts/lvc_disk_cache_sensitivity_recheck_results/`. Supported path overrides:
`BUILD_DIR`, `DATA_DIR`, `INDEX_PREFIX`, `QUERY_FILE`, `GT_FILE`, `OUTPUT_DIR`.
`LVC_BUILD_THREADS=32`, `BUILD_THREADS=4`, `MONITOR_INTERVAL=5` can be adjusted.
Use `SKIP_BUILD=1` only after rebuilding the updated binary.
`DRY_RUN=1` writes/prints the command without accessing the dataset or loading it.
An existing `OUTPUT_DIR` containing a command or round CSV is never overwritten.

Outputs:

- `rounds.csv`: 80 rows on success, one row per complete query-set pass,
  flushed immediately. Includes recall **from that round's IDs**, QPS, wall
  time, mean/p999 query latency, I/O statistics, timed CPU stages in microseconds,
  decoding totals, RSS, query-phase process minor/major faults, user/system CPU
  seconds and a start timestamp. `decode_us` is a total over the query set;
  `mean_cpu_us` is elapsed stage timing, not process CPU utilization.
- `dexor_ch2_C100M_L<L>_R<round>_{idx_uint32,dists_float}.bin`: every round's
  IDs/distances, in DiskANN bin format (two uint32 header dimensions + entries).
- `summary.csv`: all-round mean/median/min/max QPS, variation, latency, recall
  range, faults and completeness. `aggregate_qps = sum(queries)/sum(wall_seconds)`
  differs from arithmetic mean QPS when rounds vary.
- `rounds_flagged.csv`: all original rows with first-round and slow-outlier flags.
  Flags use the per-L upper fence Q3+1.5*IQR for wall time/query and mean query
  latency separately. **No rows are removed.** The mean excluding round one is
  explicitly a secondary diagnostic; the primary mean includes all ten rounds.
- `search.log`, `command.sh`, `binary_help.txt`, host snapshots and exit status.
- `pidstat.log` / `vmstat.log` when the corresponding tools are available.
  These lightweight monitors run during load and search, with no sudo/perf.
- The application's original averaged CSV and result files are also retained.

If the process fails after completing some rounds, those rows remain available;
the analyzer writes partial summaries and returns a nonzero status. To analyze
again without loading the index:

```bash
python3 scripts/analyze_lvc_disk_rounds.py /path/to/run/rounds.csv
```

Compare the L=8 median, aggregate QPS, round range and flags against the previous
100M result. Correlate slow rounds' timestamps with faults, CPU time and system
monitor logs before deciding whether any round should be excluded from a later
reported statistic. Preserve the full-run results alongside any such statistic.
