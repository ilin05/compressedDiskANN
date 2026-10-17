# Sampled BIGANN cache budget (apply_differential)

## Server runs

First validate on the existing BIGANN1M disk index:

```bash
bash scripts/test_lvc_disk_sampled_budget_bigann_1M.sh
```

Defaults: 1,000 BFS cache sample nodes; 31 MiB planning target; 32 MiB hard cap
outside PQ; Raw, ALP, DeXOR/Gorilla/Elf at chain=2 and unbounded; 1% root state
cache; K=1, L=1/2/3/4/5/6/8/10, T=1, beamwidth=1, 1 round (set ROUNDS=10 for repeated measurements). Every differential
cached vector is verified bitwise during full loading. All eight final cache
allocations must fit the hard cap; query IDs and distance bits are compared with
Raw at every L (including ALP on the integer BIGANN data). Any mismatch raises an
error with the first query/rank and bit patterns. Sampling independently checks
up to approximately 4,096 regularly spaced differential vectors per codec/chain
(exhaustive for the 1k sample), using the root state cache. The ALP implementation
retains its existing tolerance; do not assume bitwise losslessness on arbitrary
nonintegral F32 inputs. Result comparisons provide the additional BIGANN check.

After 1M passes, run:

```bash
nohup bash scripts/test_lvc_disk_memory_budget_bigann_1B.sh \
  > test_lvc_disk_sampled_budget_bigann_1B_output.log 2>&1 &
```

BIGANN1B defaults: one 1,000,000-node sample from the SAME 1B disk index; 31 GiB
planning target; 32 GiB hard cap; the same eight configurations; K=1,
L=1/2/3/4/5/6/8/10, T=1, beamwidth=2, 10 rounds. No extra alpha candidate
filtering is added/enabled. DiskANN's native PQ navigation is retained, and all
codecs use the same existing pipeline/io_uring search implementation. Do not
mix these results with original test_bigann batch/libaio results as a pure
compression comparison. Existing SSD payloads and neighbor lists are unchanged.

Stop the previous long-running experiment before measuring the new one to avoid
resource contention; these scripts never kill existing processes automatically.
The new output paths end in `bigann_1M_sampled` / `bigann_1B_sampled` and do not
reuse the old full-probe `.ok` files or overwrite old result directories.

Both wrappers accept BUILD_DIR, DATA_DIR, INDEX_PREFIX, QUERY_FILE, GT_FILE,
OUTPUT_DIR, BUILD_THREADS, SKIP_BUILD=1, SAMPLE_NODES, BUDGET_MIB,
PLANNING_BUDGET_MIB, THREADS, ROUNDS, STATE_CACHE_RATIO, RESUME=0/1 (default 1).
The 1B wrapper additionally accepts PERF_W (default 2) and VERIFY_CACHE=1 for
exhaustive full-cache verification. Large/deep verification has a measurable
construction cost. Root ratio remains the existing forest implementation's 1%.
Raw/ALP have no forest or chain depth; their sample rows use chain_max=0.

## Algorithm / memory accounting

A single search_disk_index invocation loads the full PQ once, selects 1M/1k BFS
nodes, reads their raw vectors and stored neighbors once, and filters edges to
the induced cache subgraph. It builds each forest ONCE, then passes that same
forest to the three differential encoders. Samples export a hash of the parent
array, root count, mean/max depth, mean nonroot reference L2, byte counts and
phase times. They do not substitute the separately built BIGANN1M graph for the
1B graph. One shared raw sample remains live while codecs are profiled in turn.

Predicted bytes at N final nodes:

- payload: ceil(N * sampled_payload / sampled_nodes);
- adjacency: N * (_max_degree + 1) * sizeof(uint32_t);
- two node lookups: predicted target bucket counts * ACTUAL bucket-entry sizeof;
- record metadata: ceil(N * sampled_metadata / sampled_nodes);
- decoding state cache: ceil(N * sampled_resident_state / sampled_nodes).

Record metadata includes capacities of parent/depth and offset/length arrays.
State accounting includes vector capacities and a conservative estimate for
unordered-map nodes/buckets. Fixed object sizes, malloc headers, allocator
retention, queries, search scratch and transient construction buffers are not
part of this cap; report RSS separately. Thus the enforced cap is an accounted
resident cache budget, not a hard process RSS or cgroup limit. PQ is common and
reported separately. State cache is NOW included (previous scripts excluded it).

Lookup sizeof uses the vendored tsl bucket layout: typically 32 B per nhood
bucket and 24 B per coordinate bucket on Linux x86_64. The old 16-byte-rounded
conservative accounting is replaced by these actual layout sizes. Automatic
StoreHash can only use existing padding with these default map types.

Important: this robin_map also grows early on long probe sequences (minimum
load threshold 0.15), so max_load_factor=0.5 alone does NOT determine final
bucket count. The sample inserts its actual global IDs in production order into
both exact lookup types. The estimator uses min(max_load_factor,
sample_nodes/sample_bucket_count) as effective occupancy and rounds target
bucket counts to powers of two. This is an estimate, not a guarantee about a
larger BFS prefix. Final actual bucket counts are therefore always checked.
No lookup map for tens of millions of nodes is allocated just to estimate size.

Arithmetic bisection selects the largest N predicted <=31 GiB/MiB, capped at the
common 50% node limit. It does not invoke search_disk_index at each midpoint.
Each final configuration loads/compresses its cache ONCE, prints actual bytes,
and rejects oversize caches BEFORE queries. At most one corrective retry is
allowed: recompute byte rates and lookup occupancy from the oversize actual
allocation and choose a smaller N. It never repeats full probing to fill spare
budget. Low utilization is accepted and explicitly reported. 31 vs 32 gives
3.125% slack but is not a proof of fitting, especially near hash-table jumps.

Final cache counts can differ by codec/chain. Consequently final forests have
different node sets: only the SAMPLE forest is literally shared across codecs;
final runs share the forest construction policy, root fraction and chain bound.
Unbounded can change depth distributions as the cache grows; do not label the
sample depth as the measured full-cache depth. Full forest structure is logged
in CACHE_FOREST. Graph-guided record patches preserve F32 recovery.

## Implementation / compatibility

New search options:

- --cache_budget_sample_csv PATH: profile all eight sample settings and exit;
- --lvc_chain_max 2|-1: actual full-cache forest bound;
- --cache_budget_include_state: include state allocations in --cache_budget_bytes.

The deep decoder wrapper now reports root-state hits and bytes/records over the
whole path, rather than assuming at most a root and child. No search scheduling
change accompanies this. Old non-sampled 1M script and lvc_disk_memory_budget.py
remain available; their default cap still excludes state for compatibility,
but lookup byte accounting now uses the corrected actual bucket sizes.

The existing search_disk_index target compiles the changed sources/headers;
no additional CMake target or external dependency is needed. A local arithmetic
check requiring only Python stdlib is available:

```bash
python3 scripts/test_lvc_disk_sampled_budget.py
```

## Outputs, progress and resume

`sample.csv`: all eight byte-rate/forest/time profiles. `sample.ok.json` binds it
to the binary timestamp, index/PQ/pivot files, input query/GT file identities,
sample count and state ratio. `node_plans.csv` stores INITIAL predicted node
counts; `summary.csv` stores final ACTUAL counts/bytes/utilization, payload
ratios, recall/QPS/latency/I/O and shared-sample depth columns.

`sample.log` and `CODEC_ch*_attempt*.log` receive subprocess stdout directly,
so `tail -f` shows active phases instead of waiting for subprocess exit.
Matching `.timing.json` files record commands, return codes and wall time,
including failed oversize attempts. Report sample/estimation and corrective
retry costs separately from the successful final cache-load time.

Successful case `.ok.json` files checkpoint actual allocations and search
parameters. Resume skips only completed matching sample/cases (and requires
result bins when --check-results is enabled). Aggregated CSV is updated after
each completed case. CSV existence alone is not proof that the entire run or
cross-codec correctness comparison succeeded; check run_status.json status=complete
and the final PASS messages.
