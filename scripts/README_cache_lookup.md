# Lookup-only capacity curves

On `apply_differential`, run from the DiskANN repository root:

```bash
bash scripts/test_cache_lookup_bigann_1M.sh
nohup bash scripts/test_cache_lookup_bigann_1B.sh > test_cache_lookup_bigann_1B.log 2>&1 &
```

The 1M test selects up to 500K nodes, stops after the lookup exceeds **32 MiB**,
and independently replays fresh maps to check recorded interval endpoints and
ID/value retrieval. It also loads production Raw caches at a small count and
on both sides of the stopping boundary, checking actual CACHE_BUDGET lookup
bytes/buckets against the curve. Only this small correctness check loads raw
cached vectors. It uses a generated dummy query and exits before search.
The 1B test selects up to 500M nodes and stops at the first
insertion whose combined actual bucket bytes exceed **32 GiB**. If the limit is
not reached by a custom MAX_NODES, the wrapper fails explicitly: increase MAX_NODES
(up to the production 50% cache cap), which creates a different ID file/curve.
The lookup profiling stage never loads cached full-precision vectors, encodes
payloads, or builds a forest. The initial production index load still loads PQ; BFS reads SSD graph
sectors and has temporary memory costs. The index/PQ/BFS objects are destroyed
before map profiling starts. Selecting the large BFS ID list can itself take
substantial time; it is saved once and reused on later runs. 500M ensures the
default 0.5 maximum load cannot hold the whole list under 32 GiB of lookup.
For a quicker pilot, set MAX_NODES=128000000; that pilot may end before the limit.
This does not guarantee the allocator immediately
returns every page to the OS.

## Exact scope: a fixed saved ID prefix

The tool calls production `cache_bfs_levels(..., false)`, sorts the resulting
IDs as production search does, then saves them as DiskANN uint32 bin (N,1).
Both maps use shared production aliases, original default hashes/load factors,
no reserve, and coord-first insertion. Neighbor pointers are null placeholders;
values do not affect key hashing. Bucket sizeof uses the same layout accounting
as CACHE_BUDGET. No adjacency or payload allocations are included.

**The curve is exact only for prefixes of that saved sorted ID file under the
same hash/container/build configuration.** It does not predict independently
rerunning BFS(N), even on the same index: truncated BFS sets followed by sorting
need not be prefixes of BFS(MAX_NODES). Do not use it for another index or ID
file. Existing ID files are deliberately reused; remove/change IDS_FILE when
you want a new graph's selection. Successful formal cache reads are also
required; read failures change the inserted ID sequence.

Use the matching prefix in formal search:

```bash
build/apps/search_disk_index ... --num_nodes_to_cache 32000000 \
  --cache_node_ids_file scripts/cache_lookup_results/bigann_1B/cache_ids_500000000.bin
```

This optional flag bypasses BFS, validates ascending unique IDs/range and length,
and uses exactly the requested prefix. Without it, existing search is unchanged.
The current sampled-memory-budget driver is not automatically switched to this
list or curve; integration is a subsequent step.

## Outputs and prediction

Results are in `scripts/cache_lookup_results/bigann_1M` / `bigann_1B`:

- `cache_ids_N.bin`: reproducible node list; retain it alongside the curve.
- `lookup_curve.csv`: inclusive first_nodes/last_nodes intervals, two bucket
  counts, bucket sizes, combined bytes, ID-prefix hash at last_nodes, and time.
- `probe.log`: every growth event plus progress each million inserts.
- `summary.json`: last measured feasible count and first exceeding count.

```bash
python3 scripts/lookup_capacity_curve.py \
  --curve scripts/cache_lookup_results/bigann_1B/lookup_curve.csv \
  --nodes 1000000 32000000 64000000
```

Counts outside the measured range cause an error, not extrapolation. Use
`lookup_at(load_curve(path), N)` from Python in a later budget estimator.

The byte limit is a **stopping criterion**, not a peak-RSS cap: the insertion
that crosses it is measured and then the tool stops. Rehashing temporarily holds
old and new buckets, so provide additional RAM. This is a lookup-only limit,
not the overall payload+graph+metadata+state budget.

Environment overrides: BUILD_DIR, INDEX_PREFIX, OUTPUT_DIR, MAX_NODES,
LIMIT_BYTES, IDS_FILE, SKIP_BUILD=1, BUILD_THREADS. No queries or GT are required.
