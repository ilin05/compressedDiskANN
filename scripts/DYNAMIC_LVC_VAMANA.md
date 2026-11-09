# Vamana Raw / LVC-ALP dynamic updates

## 实现范围

本轮实现内存 Vamana 的单线程 insertion / logical deletion，复用原生 `insert_point`、robust pruning、反向边更新和 `lazy_delete`。

- 默认共用已有 **Raw 图文件**。Raw 加载 Raw `.data`，ALP 加载已有 compressed `.data`，不使用 compressed 图文件，避免初始图差异。
- 静态索引加载后增加一个隐藏 frozen entry clone，指向原入口；不改写原图的边。该适配同时用于 Raw 与 ALP。
- 外部 tag = 原数据行 ID + 1。删除移除活跃标签，但保留物理记录供图导航；重插入使用同一 tag 和新物理位置。活跃向量集合始终相同，原 BIGANN GT 无需重算。返回结果转换回原数据行 ID 后计算 recall@1。
- ALP 沿用已有 `1e-4` scaling tolerance；BIGANN 整数转 F32 的验收要求误差为 0。ALP base payload 保持只读，新记录追加到 4 MiB 分块 arena；offset 区按总插入量预分配，已有 payload 不搬迁、不重编码。兼容当前旧式 ALP `.data` 文件，拒绝差分 codec 文件。
- 插入期间的 decoded-vector cache 覆盖候选搜索、pruning 与反向边更新；每次插入结束清空。首版不提供全局 hub cache。
- 独立 PQ sidecar 使用已有 8-bit codebook / codes，并为新物理位置编码；不重新训练 PQ。插入筛选和查询筛选分别控制。每个扩展池取 `ceil(alpha * pool_size)`（非空池至少一个）并计算全精度距离，只有这些节点进入候选队列。关闭筛选 / alpha=1 完全按全精度距离运行。该动态实验不复用旧静态路径中混合 PQ/精确分数的行为。
- 首版不做 consolidation、物理回收、slot reuse、并发更新或更新后索引序列化。CSV 的 COMPLETE 只支持整组配置复用，不支持从中间 batch 恢复。
- 100M、20 轮、每轮 1% 时预分配 120M 普通位置及 1 个 frozen 位置。Raw 的向量数组也按此容量分配，不能将原静态文件大小当作动态进程的 RSS。

## 服务器执行

在 `apply_differential` 上更新代码后，从仓库根目录执行。默认读取 `build/data/bigann/`；查询使用 `bigann_query.fbin`，GT 使用对应 `bigann_1M_gt.bin` / `bigann_100M_gt.bin`。

```bash
# 1. BIGANN1M 正确性：缺少 1M 内存索引时自动准备 shared Raw graph 和 ALP payload。
# 使用现有 1M GT；若缺少 base fbin，则从 bigann_100M.fbin 截取前 1M。
bash scripts/test_dynamic_lvc_vamana_bigann_1M.sh

# 2. BIGANN100M 预实验：复用现有 100M 索引，2 batch，每轮 0.1%，L=10。
bash scripts/test_dynamic_lvc_vamana_bigann_100M_pilot.sh

# 3. 正式实验：20 batch，每轮 1%，L=5,10,20，R=3，检查点 0,2,...,20。
bash scripts/test_dynamic_lvc_vamana_bigann_100M.sh
```

三个入口均单线程更新、单线程检索。正式和预实验依次执行 Raw、ALP（默认 insertion alpha=0.1），ALP 查询分别 TLS off / on(alpha=0.2)。构建 1M fixture 可以用 `PREPARE_THREADS=32`，与更新/查询线程数独立。相同 seed 下 Raw / ALP 的每个 batch 选择相同节点。

1M 正确性矩阵为 Raw、ALP exact/cache on、ALP alpha=1+PQ、ALP exact/cache off、ALP insertion alpha=0.1。前三种 ALP 对照必须与 Raw 的逐 batch 图 hash、结果 ID、F32 距离 bits 相同；所有模式检查恢复向量、新 PQ 码、活跃/已删除节点计数以及删除后查询不返回被删标签。BIGANN 整数转 F32 的重构误差必须为 0。全部通过生成 `CORRECTNESS_PASS`。

常用覆盖参数：

```bash
# 显式复用服务器文件；若 bigann_100M.fbin 缺失，DATA 默认回退到 Raw .data。
DATA_DIR="$PWD/build/data/bigann" \
RAW_PREFIX="$PWD/build/data/bigann/memory_index_bigann_100M_R32_L50_A1.2" \
ALP_PREFIX="$PWD/build/data/bigann/compressed_memory_index_bigann_100M_R32_L50_A1.2" \
PQ_PREFIX="$PWD/build/data/bigann/bigann_100M.fbin" \
bash scripts/test_dynamic_lvc_vamana_bigann_100M_pilot.sh

# 复用已有完整且配置/输入一致的结果；不覆盖不完整的目录。
RESUME=1 SKIP_BUILD=1 bash scripts/test_dynamic_lvc_vamana_bigann_100M.sh

# 改实验参数时使用新输出目录。
INSERT_ALPHA=0.2 OUTPUT="$PWD/scripts/dynamic_lvc_vamana_results/bigann_100M_alpha02" \
bash scripts/test_dynamic_lvc_vamana_bigann_100M.sh
```

可调环境变量还包括 `BUILD_DIR`, `BUILD_JOBS`, `DATA`, `QUERY`, `GT`, `PQ_BYTES`(32), `BATCHES`, `BATCH_RATIO`, `SEARCH_L`(逗号分隔), `ROUNDS`, `SEARCH_EVERY`, `QUERY_COUNT`(0=完整查询), `SEED`, `MODE=all|raw|alp`, `PREPARE`, `RAW_BINARY`, `LVC_BINARY`。已有 1M 索引/PQ 时可以 `PREPARE=0` 跳过准备工具。

注意：`generate_pq` 输出 `_pq_pivots.bin` / `_pq_compressed.bin`，sidecar 使用 `PREFIXpq32_*.bin`。1M 准备流程显式建立命名链接；100M 默认复用原内存检索脚本的 `bigann_100M.fbinpq32_*.bin`。

## 使用 test_bigann 原始 Raw 库

默认 Raw datastore 是从 `test_bigann` 原始 F32 实现复制并改名的类；当前分支同一驱动可以直接完成所有对照。如果要求 Raw 使用原始分支的编译库，另准备一个 `test_bigann` checkout：

```bash
RAW_REPO=/path/to/test_bigann_checkout \
bash scripts/build_dynamic_vamana_raw_baseline.sh

RAW_BINARY=/path/to/test_bigann_checkout/build/dynamic_vamana_driver/test_dynamic_lvc_vamana_raw \
bash scripts/test_dynamic_lvc_vamana_bigann_1M.sh
```

helper 使用 Unix Makefiles，并复用原生 build_memory_index 的编译和完整链接配置，兼容静态 MKL/libdiskann 构建；已有 build 使用其他 generator 时另设一个新的 RAW_BUILD_DIR。可先用 `git worktree add ../compressedDiskANN_test_bigann test_bigann` 准备源码 checkout，索引仍读取现有数据目录，不需要复制。helper 不切换分支；仅在 Raw checkout 的 `include/index.h` 添加一个 friend 声明（会显示为一行 git diff），让相同实验驱动能够导入静态图。随后 Raw 库与驱动使用同一份头文件重编译，图更新、距离计算和数据布局均不变。可用 `MODE=raw`、`MODE=alp` 分别启动并写入同一结果根目录；勿同时运行基准测试，以免争用资源影响 latency。

## 结果与口径

结果保存在 `scripts/dynamic_lvc_vamana_results/`，不写入被忽略的 build 目录。各配置各自保存：

- `config.json`：参数及输入/二进制路径、长度、mtime，用于防止误复用旧结果。
- `updates.csv`：初始与每一 batch 的 insertion/deletion 均值、总耗时、load 时间、编码/PQ 编码耗时、解压次数/插入缓存命中、向量恢复验证、内存组成。
- `search.csv`：各检查点、query mode、L、每一轮完整检索的 recall@1（0–1，严格匹配 GT 首列 ID）/ QPS / 均值 latency / 空结果数。
- `timing.csv`：load、准备、delete、insert、验证、大小统计、预热、每轮查询和总 wall time；
- `run.log`、`COMPLETE`；正确性实验额外保存逐查询 `results.csv`。

每完成一组即刷新根目录 `all_updates.csv`、`all_search.csv`、`summary.csv`。程序每一 batch/搜索轮也会 flush 本地 CSV；失败组的已写入数据可以保留检查。summary 的 QPS/latency 是逐轮算术平均，完整每轮数据均保留。

内存字段区分：`payload_bytes` 是实际压缩记录字节（含保留的删除记录），`payload_capacity_bytes` 包含 arena 的已分配块；`offset_capacity_bytes` 按物理预留容量计；`graph_bytes` 按当前保存的边计，`graph_capacity_bytes` 含邻接 vector 对象及边数组容量；`pq_capacity_bytes` 包含所有预留 codes 和码本。`tag_serialized_bytes` 是 tags+delete list 的序列化字节估计。`index_accounted_bytes` 汇总 payload、offset 容量、逻辑 graph、PQ 容量、序列化 tag 估计，**不是总驻留内存或已保存的索引文件大小**，不含锁和哈希表桶等开销；总进程占用另看 `current_rss_kib` / `peak_rss_kib`（包括输入 mmap、scratch、分配器等）。不要混用这两个口径。

采样、输入复制、删除查询验证、结果检查、大小统计不计入 insertion/deletion 时间。`insert_exact_distances` / `insert_pq_distances` 记录 ALP 插入的精确距离调用 / PQ 候选打分次数（Raw 原始分支未插桩，字段 0 表示未采集）。`graph_and_metadata_seconds` 为 insertion 总时间减去 payload/PQ 编码时间，仍包含解码、搜索、pruning、锁和标签维护。load 包含数据、图、tag/frozen 适配和 PQ 加载，不包含离线 fixture 构建。

L 很小时，队列可能只剩已删除的导航节点，原生 `search_with_tags` 会返回空结果；记录为 recall miss，不增大 L，也不读取未初始化结果。更新后 recall 是实测值，不要求与更新前完全相等。

## 本地小样本验收

```bash
cmake --build build --target test_dynamic_lvc_vamana prepare_dynamic_lvc_vamana -j8
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/test_dynamic_lvc_vamana_smoke.py \
  --binary build/apps/test_dynamic_lvc_vamana \
  --prepare-binary build/apps/prepare_dynamic_lvc_vamana
```

小样本使用 1024 × 128 整数 F32、真实原生构图、两轮 5% delete/reinsert，以及合法的 native PQ 码本/数据格式；不是 1M/100M 服务器实验的替代结果。
