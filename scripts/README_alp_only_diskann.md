# R3.Q1: original DiskANN batch search + ALP cached vectors

Based on `test_bigann`. Default `--cache_codec raw` keeps the original raw vector
buffer, batch candidate processing, PQ navigation, libaio, and 10% BFS cache cap.
`--cache_codec alp` changes only the RAM cache payload representation and uses a
50% BFS cap. SSD vectors/graph, PQ codes, neighbor layout and search scheduling
are unchanged. There is no predecessor forest, decoder state cache, alpha
candidate filter, or asynchronous pipeline/io_uring integration.

`include/alp_cache_codec.h` copies the ALP encoder/decoder verbatim from the
`apply_differential` commit identified in that header, including its decimal
exponent selection, 1e-4 tolerance, and byte layout. Do not infer universal
bitwise losslessness from its name. BIGANN integer-valued F32 vectors are checked
by the correctness script. `--verify_cache` bitwise-checks EVERY cached vector
at loading time and aborts at the first mismatch, reporting node ID, coordinate,
expected value and actual decoded value. No patch, raw escape, or changed codec
is silently added. ALP mode requires float/L2/full-precision SSD vectors.

ALP loads in blocks of eight, immediately encodes each vector and discards the
block's raw coordinates. It stores a variable-length byte stream and node-ID to
byte-offset map. Cached hits decode into the existing per-query scratch, then
call the original full-precision distance function. Raw cache hits still use
their original vector pointer. No query-time decode timer is added.

## BIGANN1M correctness

Run from the DiskANN repository, on test_bigann with these changes:

```bash
bash scripts/test_alp_only_disk_cache_bigann_1M.sh
```

Defaults: existing `build/data/bigann/disk_index_bigann_1M_R32_L50_A1.2` index,
`bigann_query.fbin`, `bigann_1M_gt.bin`, 10,000 cache nodes for BOTH codecs,
K=1, L={1,2,3,4,5,6,8,10}, threads=1, beamwidth=1, rounds=1.
The script builds search_disk_index (configures CMake if needed). It requires the
original libaio/MKL/Boost/tcmalloc dependencies, not liburing.

It checks actual cached count and cache-list hash, every cached ALP vector's
bitwise recovery, all query result IDs and L2 bits, and mean I/O for every L.
The bin comparator reports the first mismatching query/rank and actual values.
A PASS establishes correctness on this input, not arbitrary nonintegral F32.
Recall, QPS, latency, RSS, load time and payload ratios are preserved in CSVs.
For additional configurations and the 10%/50% cap check:

```bash
THREAD_COUNTS="1 4" BEAM_WIDTHS="1 2" K_VALUES="1 10" CHECK_CAPS=1 \
  bash scripts/test_alp_only_disk_cache_bigann_1M.sh
```

Cap checks additionally request 60% of N (separately for both codecs) and verify
actual cache counts are 10% for Raw and 50% for ALP. The correctness pair itself
must request <=10%, so the two modes cache identical nodes. Optional cap runs
are not compared for identical I/O because their cache sets differ.

## BIGANN1B ALP-only

After BIGANN1M passes:

```bash
nohup bash scripts/search_alp_only_disk_index_bigann_1B.sh \
  > search_alp_only_bigann_1B_output.log 2>&1 &
```

Defaults mirror `search_disk_index_bigann_1B.sh`: single thread, 100 rounds,
beamwidth=2, K=1 with L={1,2,3,4,5,6,8,10}, K=10 with
L={10,11,12,13,14,16,18,20}. ALP default CACHE_NODES=400000000, permitted by the
50% cap (actual cached count is always logged). It runs ALP only, without memory
budget selection. To use the original Raw script's cache count, set
`CACHE_NODES=100000000`. To verify every cached vector also on 1B, set
`VERIFY_CACHE=1`; this changes load cost, not the codec/search path.

## Paths and output

Both scripts accept BUILD_DIR, DATA_DIR, INDEX_PREFIX, QUERY_FILE, GT_FILE,
OUTPUT_DIR, CACHE_NODES, BUILD_THREADS, SKIP_BUILD=1 overrides. The 1M script
accepts ROUNDS (default 1), THREAD_COUNTS, BEAM_WIDTHS, K_VALUES, CHECK_CAPS.
The 1B script accepts ROUNDS (default 100), THREADS, BEAM_WIDTH, VERIFY_CACHE.
All shell files use LF newlines.

Outputs are under `scripts/alp_only_results/bigann_1M` and `bigann_1B`.
Each run has its own log, result bins, and `RESULT_PREFIX_search_summary.csv`.
The correctness run also produces `summary.csv` with all configurations.
The legacy Raw summary path next to the index is additionally retained for
existing scripts; ALP never overwrites that Raw summary.

New summary fields: CacheCodec, ActualCachedNodes, CacheListHash, RawPayloadBytes,
CachePayloadBytes, OffsetValueBytes, PayloadRatio, PayloadAndOffsetsRatio,
CacheLoadSeconds. Payload ratios use original F32 data bytes. OffsetValueBytes
counts just the uint64/size_t offset values, NOT map keys/bucket allocations or
adjacency/PQ storage. These are payload-level statistics, not a total memory
budget or complete index compression ratio. RSS remains a separate metric.
CacheLoadSeconds includes BFS selection, reads, encoding and optional checking.
ActualCachedNodes reflects loaded nodes; Num_Nodes_To_Cache is the requested
count, which may be clipped by the mode-specific cap.

CSV/Markdown can be versioned; generated result bins and logs are ignored.
