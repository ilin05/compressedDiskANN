# BIGANN-1B differential cache sensitivity (chain=2)

Run from the DiskANN repository on `apply_differential`:

```bash
nohup bash scripts/test_lvc_disk_cache_sensitivity_bigann_1B_chain2.sh \
  > test_lvc_disk_cache_sensitivity_bigann_1B_chain2_output.log 2>&1 &
```

The nine cases run **serially**, in this order:

1. 10M nodes: DeXOR, Gorilla, Elf.
2. 25M nodes: DeXOR, Gorilla, Elf.
3. 50M nodes: Gorilla.
4. 100M nodes: DeXOR, Elf.

Each case loads the index/cache once, builds a degree-rooted forest (initial
roots 1%, chain maximum 2 records = at most 1 predecessor edge), encodes the
cache, and searches all `L=1,2,3,4,5,6,8,10` with `K=1, T=1, W=2, R=3`.
Load/build threads default to 32; decoding state cache is 1% of cached nodes.
Extra LVC candidate filtering is off. DiskANN's native PQ navigation remains.
Only the in-memory cache is compressed; disk payloads/adjacency stay unchanged.

There is **no 20GiB cap, sampling, capacity estimation, or automatic node-count
reduction**. Every case must load exactly its requested node count. Each process
uses the existing deterministic BFS selection (`shuffle=false`); at equal node
counts the runner checks both the selected-node hash and forest hash across
codecs. A mismatch is saved as a failure. This checks the current experiment;
it does not retroactively establish matching cache populations for older runs.

The 100M cases exceed the previously measured cache populations. Run them alone
on the server, after reviewing the smaller cases' memory consumption. Total RSS
includes PQ, node lookups, state caches, and retained allocations. Forest
construction can have a higher peak than the final cache footprint.

## Outputs and resume

Default output directory:
`scripts/lvc_disk_cache_sensitivity_results/bigann_1B_chain2/`.

- `summary.csv`: one row per case/L; a successful full run has 72 rows.
- `case_summary.csv`: nine case rows, with load/build/encoding/wall times and
  memory statistics. Do not sum repeated per-process load times over L rows.
  RSS here is from the first L; `summary.csv` retains the RSS for every L.
- `<codec>_ch2_C<N>_results.csv`: saved immediately after that case completes,
  including an explicit failure row if loading/searching/parsing failed.
- `<codec>_ch2_C<N>_search_summary.csv`: original C++ search summary.
- `.log`, `.timing.json`: process log, executed command, exit status, wall time.
- `.done.json`: successful-case input/command signature; `config.json` and
  `run_status.json`: plan and progress. A directory lock prevents concurrent
  writers to the same output directory.

The aggregate CSVs are refreshed after every case. Rerunning the command reuses
successful cases only when input paths/sizes/timestamps, executable, parser,
runner, and all case parameters match. Failed or interrupted cases are retried.
Avoid rebuilding the binary before a resume, because a changed binary timestamp
invalidates reuse:

```bash
SKIP_BUILD=1 RESUME=1 nohup bash scripts/test_lvc_disk_cache_sensitivity_bigann_1B_chain2.sh \
  > test_lvc_disk_cache_sensitivity_bigann_1B_chain2_resume.log 2>&1 &
```

`payload_budget_bytes` is adjacency + vector payload + record metadata + state
cache; it **excludes PQ and lookup allocations**. It is an observed footprint,
not a budget limit. `lookup_bytes` and `cache_with_lookup_bytes` are reported
separately. For RSS–performance comparisons use
`current_rss_after_search_gib` (total current process RSS), with peak RSS as a
separate construction-memory measurement. Recall values are percentages.
Forest/decode correctness is not inferred from recall: `VERIFY_CACHE=1` adds
the existing full cached-vector bitwise verification (off for timing runs).

## Overrides / preview

```bash
# Print all nine commands without building, reading datasets, or launching searches.
DRY_RUN=1 bash scripts/test_lvc_disk_cache_sensitivity_bigann_1B_chain2.sh

# Optional small BIGANN-1M smoke run on the server, separate from the 1B results.
EXPECTED_POINTS=1000000 \
INDEX_PREFIX="$PWD/build/data/bigann/disk_index_bigann_1M_R32_L50_A1.2" \
GT_FILE="$PWD/build/data/bigann/bigann_1M_gt.bin" \
CASES="dexor:1000 gorilla:1000 elf:1000" VERIFY_CACHE=1 \
OUTPUT_DIR="$PWD/scripts/lvc_disk_cache_sensitivity_results/bigann_1M_smoke" \
bash scripts/test_lvc_disk_cache_sensitivity_bigann_1B_chain2.sh
```

Adjust the smoke GT path to the actual 1M truthset. It must match the query
file. Other overrides: `BUILD_DIR`, `DATA_DIR`, `QUERY_FILE`, `OUTPUT_DIR`,
`LVC_BUILD_THREADS`, `NUM_ROUNDS`, `BEAMWIDTH`, `STATE_CACHE_RATIO`, `L_VALUES`.
`CASES` accepts space-separated `codec:count` pairs. Changing the selected cases
changes the aggregate CSV's scope; per-case files remain available.
The default nine-case run is estimated at about 12–20 hours, with 24 hours
reserved; 100M forest costs are extrapolations rather than measured timings.
