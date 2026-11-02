# Unified BIGANN-1B cache-node sensitivity

After updating and rebuilding the DiskANN source, run:

```bash
nohup bash scripts/test_lvc_disk_unified_cache_bigann_1B.sh > unified_cache.log 2>&1 &
```

This serially runs Raw, ALP, DeXOR, Gorilla and Elf, each with 10M, 25M and
50M cache nodes. Each case loads once with 32 cache-read threads; ALP encoding
and differential forest/encoding use 32 threads; query search uses one thread.
Raw has no codec encoding. BFS cache-node selection and final cache-map merging
remain sequential. Differential forests use 1% degree roots and chain_max=2;
decoding state cache is 1%. The SSD index and native PQ navigation are unchanged.
No additional LVC candidate filtering or hard memory cap is applied.

The full query file is evaluated in L order 1,2,3,4,5,6,8,10, with K=1,
beamwidth=2 and ten rounds per L. Each L's first round is warm-up for the
summary; rounds 2--10 are averaged. `rounds.csv` retains all ten rounds and
per-round ID/distance result files, RSS and page-fault statistics.

Results default to a timestamped `scripts/lvc_unified_cache_results/bigann_1B_*`
directory. Every completed or failed case updates `summary.csv` and `status.json`;
each attempt has a separate directory, search log, command and round records.
Successful cases are saved in `done.json`. To resume the exact same experiment:

```bash
OUTPUT_DIR=/path/to/existing/run RESUME=1 \
  bash scripts/test_lvc_disk_unified_cache_bigann_1B.sh
```

Resume checks the binary, dataset file sizes/mtimes and parameters, and reruns
failed cases in a new attempt directory. An output-directory lock prevents two
runners writing the same run. Cache-node hashes must agree for the same count;
differential forest hashes must also agree across codecs. Any failure is recorded
and the final exit status is nonzero.

`summary.csv` accounts bytes for global PQ codes, allocated PQ tables/rotation,
entry-point vectors and IDs, fixed-slot adjacency, both lookup bucket arrays,
vector payload, forest/record metadata and decoding state cache. Their sum is
`accounted_index_bytes`; state-cache map overhead is a conservative estimate.
Allocator headers, query scratch, runtime libraries and other small index objects
are excluded, so this is an accounted index size, not total process RSS. Current
and peak RSS are recorded separately; parallel loading uses 32 I/O contexts.
RSS peak includes loading and temporary buffers. No estimate is substituted for
missing components. Compression ratio is F32 vector bytes divided by payload.

Overrides: `BUILD_DIR`, `DATA_DIR`, `INDEX_PREFIX`, `QUERY_FILE`, `GT_FILE`,
`OUTPUT_DIR`, `CACHE_NODES`, `CODECS`, `L_VALUES`, `LOAD_THREADS`, `NUM_ROUNDS`,
`BEAMWIDTH`, `EXPECTED_POINTS`, `BUILD_THREADS`, `SKIP_BUILD`, `VERIFY_CACHE`,
`DRY_RUN`. For example `DRY_RUN=1` prints all commands without loading data.
The default needs BIGANN-1B with 32-byte PQ codes and F32 128-d queries.

Run a correctness smoke test with an existing BIGANN-1M index/ground truth first:

```bash
bash scripts/test_lvc_disk_unified_cache_smoke_bigann_1M.sh
```

It uses 1024 cached nodes, W=1, two rounds and verifies differential payloads
bitwise against disk, ALP encoding against its serial codec and tolerance, and
DeXOR/Gorilla/Elf result IDs and distance bits against Raw. ALP uses its existing
1e-4 scaling tolerance; it is not assumed bitwise lossless for arbitrary F32.
The smoke test's verification overhead makes its timing unsuitable for benchmark
claims. Set `INDEX_PREFIX`, `QUERY_FILE`, or `GT_FILE` if filenames differ.
