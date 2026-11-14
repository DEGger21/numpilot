# 无缺口迁移归档（2026-11-12）

> **历史归档。** 本目录冻结的是 2026-11-12 的批次证据，保留原始 AsNumpy/CANN 运行时、哈希和计时口径，不随当前组件基线滚动更新。当前可复现 demo 请从仓库根目录的 `examples/demo/` 和 `components/asnumpy/` 开始；本页的旧版本号只用于复核该批次。

本目录保存 71 个“仅使用 AsNumpy 已覆盖 API”的 NumPy 程序，以及 NumPilot
通过真实配置模型（deepseek-flash）端到端迁移得到的 AsNumpy 程序。迁移批次
使用 `migrate --mode contract` 并开启 strict input types，71 例全部通过框架的
表面门禁与数值门禁（shim 后端）；70 例零修复，H012_logarithmic_grid 触发
1 次真实修复。

## 1. 来源与可复现性

- 批次运行根目录：`runs/20261112-translation-mainline/no-gap-batch-strict`（服务器工作目录，未随仓库发布）
- `batch-summary.json` SHA-256：`3e244359d34d556c55029cfcca8faa68d9f808c062333f2554ce01ab5b8e0a5b`
- `batch.log` SHA-256：`65842d24f2bc92ecd23482e0329cdaa5da4acffd715d3ede871dd4f8b23ac8a6`
- 模型：`deepseek-flash`（openai-compatible，`response_model_identity=exact`）
- 迁移总耗时：363.22 s（71 例，含模型调用、执行与数值验证）
- 框架测试：`2644 passed in 98.20s`
- 每个 case 的源程序/迁移程序 SHA-256、路由、rewrite 数、`n_elements`、
  模型身份见 `MANIFEST.json`。

> `group` 字段沿用 benchmark 原始分类；H/I 前缀表示该用例在设计时用于
> 覆盖官方映射/模型补全场景，不代表本次迁移仍存在缺口。

> 批次验收是在 shim 后端完成的。真实 NPU 上的运行结果见 `timings/` 与
> 第 4 节；两者不要混用。

## 2. 目录结构

```
MANIFEST.json                 71 例的元数据与哈希
cases/<case>/program.py       原始 NumPy 程序（数值 oracle）
cases/<case>/migrated.py      框架生成的 AsNumpy 程序
cases/<case>/case.json        单例迁移记录（不含模型私有字段）
timings/timings.json|csv      真实 NPU/CPU 端到端计时
timings/logs/                 计时 worker 日志（已扫描，无私有绝对路径）
timings/SHA256SUMS            计时证据哈希
MIGRATION_TIME.md             迁移耗时 vs 人工估算
PROMO_SUMMARY.md              可直接引用的口径与边界
```

## 3. 计时口径

`tools/bench_no_gap_archive.py` 为每个 case 启动两个独立 worker：

| 口径 | 定义 |
| --- | --- |
| `cpu_end_to_end_ms` | `build_inputs()` + `forward()`，绑定单核，BLAS/OMP=1 |
| `npu_end_to_end_ms` | `build_inputs()` + H2D + `forward()` + D2H 同步 |
| `cpu_compute_only_ms` | 预构造输入，仅 `forward()`，单核 |
| `npu_device_resident_ms` | 输入常驻设备，仅 `forward()` + D2H 同步 |

复现命令：

```bash
python tools/bench_no_gap_archive.py \
  --archive examples/no_gap_archive_20261112 \
  --out runs/20261112-no-gap-timing/full \
  --warmup 5 --iters 20
```

运行环境：Ascend NPU + AsNumpy `0.4.1.dev12+gc9115e8ce.d20261031`。

## 4. 真实 NPU 正确性结果（原始规模，71 例）

- 71/71 执行成功；NPU 日志中可见真实 `aclnn*` 算子（如 `aclnnAdd`、
  `aclnnMul`、`aclnnSqrt`、`aclnnMatmul`、`aclnnReduceSum` 等）。
- 数值：69/71 在 `rtol=1e-4, atol=1e-5` 内一致。两例差异均为已登记的
  运行时契约差异，不是迁移错误：
  - `B011_nan_inf_extrema`：`fmax` 在 NPU 上走 `aclnnMaximum`，NaN 传播；
    NumPy `fmax` 忽略 NaN。分析器已将其标记为
    `semantic_divergence / review_required`。
  - `B018_signed_zero`：`copysign` 丢失 `-0.0` 的符号；NumPy 保留。
    同样已在 `gap_filler` 的 `_CONTRACT_NOTES` 登记。

**关于计时**：这 71 例是正确性用例，`n_elements` 仅 1–528（中位数 36），
计时由 NPU 启动与 H2D/D2H 开销主导，不代表产品在大规模计算上的性能，
因此不作为加速比口径。原始计时数据仍保留在 `timings/` 供审计。71 个样例
均已生成大规模性能变体，逐行加速比、数值边界与复现方式见
[`examples/scaled_benchmark_20261112/BENCHMARK_TABLE.md`](../scaled_benchmark_20261112/BENCHMARK_TABLE.md)。

## 2026-11-14 复核

本批没有新增无缺口样例，71 例数据保持不变。仓库记录入口见
[`BENCHMARK_TABLE.md`](BENCHMARK_TABLE.md)；当前规模性能表见
[`examples/scaled_benchmark_20261112/BENCHMARK_TABLE.md`](../scaled_benchmark_20261112/BENCHMARK_TABLE.md)。
