# DEEP cross-graph 补齐

入口：`scripts/test_lvc_cross_graph_deep_resume.sh`。
只补 DEEP，不运行 BIGANN Raw/ALP load 计时，不重新构图或训练 PQ。

## 运行

在服务器的 DiskANN 仓库根目录执行：

```bash
# 可选：无需真实数据集的 AddressSanitizer 边界回归测试。
bash scripts/test_lvc_deep_codec_bounds.sh

# 只检查已有文件、输入语义和可复用结果，不编译、不运行实验。
bash scripts/test_lvc_cross_graph_deep_resume.sh --plan

# 先在真实 DEEP 的 1024 个前缀向量及诱导子图上检查四种链长。
bash scripts/test_lvc_cross_graph_deep_resume.sh --smoke-only

# 小测试通过后补全量实验。主流程也会检查/复用已通过的小测试。
nohup bash scripts/test_lvc_cross_graph_deep_resume.sh \
  > run_lvc_cross_graph_deep_resume.log 2>&1 &
```

默认输出目录为 `build/data/cross_graph_transfer/deep-image-96-angular`。
可用 `OUTPUT_DIR` 或 `--output` 指定已有 DEEP 结果所在目录；该目录的上一级应为
cross-graph 结果根目录。HNSW 路径自动检测同级 `hnswlib_cpp_py`，否则使用
`~/HNSWCW/compressedhnsw`，也可设置 `HNSW_DIR`。
构建目录分别可用 `DISKANN_BUILD_DIR`、`HNSW_BUILD_DIR` 指定。

默认 `BUILD_THREADS=32`、单线程检索、`ROUNDS=3`。例如：

```bash
BUILD_THREADS=16 HNSW_DIR="$HOME/HNSWCW/compressedhnsw" \
  bash scripts/test_lvc_cross_graph_deep_resume.sh
```

默认编译转换、验证、Vamana 检索工具；需要补 HNSW 检索时才编译其检索工具。
复用服务器原有、已经配置好的 CMake build 目录。
`--skip-build` / `SKIP_BUILD=1` 仅适用于已用本次源码重新编译的二进制；旧 converter
没有 `--threads` 参数，也没有这次的 DeXOR 越界修复。

## 复用和补齐规则

- 输入必须是已有的完整 **9,990,000 × 96** DEEP 数据，`config.json` 声明
  `unit_normalized_f32`，HNSW/Vamana 共用相同 F32 语义。不会重新归一化或覆盖输入。
- 复用 HNSW 的 12 个 codec/chain 构建结果，以及原 Vamana graph 和 12B PQ 码。
- HNSW 检索 CSV 只有在 codec、degree policy、chain=2、K=1、1% cache、TLS、
  efSearch 扫描和轮数均匹配时才复用，否则只重新检索现有 chain=2 索引。
- Vamana 完整的 chain=2 CSV/records 可复用；其 raw/三种 codec 的全部向量均重新
  做一次 F32 bitwise/L2 验证。验证通过记录可在后续续跑中复用。
- chain=4/8/unbounded 只补压缩统计。converter 会对**全部节点**解码并逐 bit 验证，
  通过后才写该 codec 的完整统计行。不会保存深链检索索引。
- 空 CSV、缺少 codec 行、错误节点数/字节数/深度、缺失 records 均视为不完整。
  新压缩产物先写 `deep_resume/pending_ch*`，三个 codec 都完成后才发布到原文件名。
  原脚本已经完整生成的深链 CSV 可复用；forest CSV 本身不构成通过证据。
- 某条深链压缩失败会保留日志并继续其他链长和 chain=2 检索；整体返回非零。
  修复后再次执行同一命令，复用带输入/输出签名的已完成阶段。
- 不同时运行旧 cross-graph runner；新入口用文件锁防止自身重复启动。

## 保持的实验配置

- 每张图各自按 degree 选择最高 1% 初始 roots，chain=2/4/8/-1。
- **保留原跨图 forest 策略**，包括全局 fallback；不启用 BIGANN 的 nearest-root-local
  fallback 优化，也不把小测试诱导子图的压缩率作为全量实验结果。
- 仅 chain=2 检索：K=1，TLS off/on，alpha=0.2，state cache=1%。
- HNSW efSearch：10,20,30,40,60,80,100,150,200。
- Vamana L：1,2,3,4,5,6,8,10。每组 3 轮，单线程。
- 每种 TLS 配置分别比较 Vamana raw 与三种 codec 的 ID/L2 输出；TLS-on 对照
  raw TLS-on，不能要求近似筛选结果与 TLS-off 相同。
- DEEP 压缩率小于 1 时照常保留；是否纳入汇总只由完整性和正确性决定。

## 输出和验收

原 DEEP 目录新增/补齐 `vamana_lvc_ch{4,8,-1}_compression.csv`、
`vamana_*_k1_tls*.csv` 及结果 bin、`retrieval_ch2.csv`。
过程信息集中于 `deep_resume/`：

- `status.csv`：每次实际执行的阶段、退出码、耗时和日志路径。
- `compression_ch*.log`：forest、各 codec 编码及全量解码验证的阶段信息。
- `verify_*.log`：chain=2 全量 F32/L2 正确性；`search_*` 保存检索输出。
- `correctness.csv`：96 项比较（3 codecs × 2 TLS × 8 L × ID/距离），要求全部相同。
- `decision.json`：只有全部完成且正确性通过才写 `status=complete`、`include_deep=true`。

完整检索预期 **118 行**：HNSW 54 行 + Vamana 64 行。
成功后调用现有汇总脚本，在上一级刷新 compression/retrieval/recall-aligned CSV，
并将相同的四份汇总 CSV 同步到 `scripts/lvc_cross_graph_transfer_results/`；
五个数据集均完整时 compression summary 为 **120 行**，retrieval summary 为 **590 行**。
根目录已有的其他四个数据集文件不会重跑或覆盖。

如果实验已经通过，只需要更新汇总目录，执行以下命令即可，无需重新运行补齐脚本、
编译或执行检索：

```bash
python3 scripts/summarize_lvc_cross_graph_transfer.py \
  --root build/data/cross_graph_transfer \
  --output-dir scripts/lvc_cross_graph_transfer_results
```

`--root` 是原始实验结果的输入根目录；`--output-dir` 是汇总 CSV 的发布目录，
默认值为上述 `scripts/lvc_cross_graph_transfer_results/`。为兼容旧流程，
四份汇总 CSV 也会保留在 `--root` 中。

## 本次定位与修复的边界

本地 AddressSanitizer 在归一化 F32 小样本、DeXOR chain=4 编码阶段复现了
`DeXORTools::getP10()` 越界：很小的非零值在粗精度上舍入为零，被当作整数，
导致精度搜索不断增大指数并越过幂表。修复排除这种零舍入，检查五位指数范围，
不支持的指数使用 DeXOR 已有 raw exception 路径；F32 patch 机制保持不变。
改动仅在 DiskANN 自带的 codec 副本中，不依赖 HNSW 仓库头文件。

这说明存在可复现的相关缺陷，**不能单凭本地复现断言服务器旧 SIGSEGV 的唯一原因**。
新日志可进一步确定全量失败位置；已有 HNSW 数值结果沿用其原构建产物。
本地小样本与流程测试通过也不等于服务器完整 DEEP 实验已通过。

本次本地验证：512 个归一化 F32 向量（含零、极小正负坐标），三种 codec、
四种链长、1/4 线程均通过 AddressSanitizer 和恢复检查；线程数改变未改变 forest
或压缩字节统计。另用模拟工具验证了中断后续跑、半成品拒绝、输入变更失效、
118 行检索汇总及 96 项结果比较。服务器真实 9.99M 数据仍需按上述命令验收。
