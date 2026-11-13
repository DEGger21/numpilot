# 规模化性能基准（2026-11-12）

> **历史归档。** 本目录对应 2026-11-12 的规模化计时批次，结果与运行环境按本页记录冻结，不能直接作为当前静态组件基线。当前版本的可复现路径是 `examples/demo/`、`examples/scientific/` 与 `components/asnumpy/`；本页保留旧批次数据用于审计和对比。

本目录覆盖 `examples/no_gap_archive_20261112` 中全部 **71** 个无缺口迁移样例，用于衡量 NumPilot 迁移代码在真实昇腾 NPU 上的稳态计算性能。每个用例包含：

- `program.py`：大规模数据版 NumPy 源程序（数值 oracle）
- `migrated.py`：NumPilot 生成的 AsNumpy 程序（算子图与归档用例一致）
- `case.json`：来源用例、规模说明与可扩展性标记

对外宣传清单位于 [BENCHMARK_TABLE.md](BENCHMARK_TABLE.md)。本页保留完整口径、覆盖范围、数值边界与复现方式。

## 1. 生成方法

`tools/scale_no_gap_archive.py` 从归档用例复制模块，只替换 `build_inputs()`；`forward()` 逐字保留。H012/H013 的网格点数写在 `forward()` 中，因此同步把 `17`/`21` 改为 `4_000_000`。除此之外没有改动任何算子或常量，也没有重新调用模型。

- 71 个归档 base case 全部覆盖。
- 72 个计时用例：C011 保留 512 宽（S3）与 1024 宽（S4）两个规模点，表中取 S4。
- 生成清单见 `scale_generation.json`。

## 2. 覆盖范围

| 组 | 用例数 | 说明 |
| --- | ---: | --- |
| A | 15 | 基础逐元素 / 归约 / 矩阵运算 |
| B | 16 | dtype、内存布局、空轴、NaN 等语义契约 |
| C | 15 | 真实结构型程序（MLP、层归一化、训练步等） |
| D | 6 | 别名与组合算子 |
| F | 1 | 排序 kind 契约 |
| G | 1 | 多通道 cbrt |
| H | 15 | 别名、组合与统计 |
| I | 2 | 缺口算子接入 |
| **合计** | **71** | 71 个无缺口样例 |

## 3. 计时口径

**加速比 = CPU 单核 `forward()` 中位耗时 ÷ NPU 设备常驻 `forward()` 中位耗时。**

CPU 侧输入已构造在内存中；NPU 侧输入已预置在设备上，计时含 `forward()` 与结果回传，不含 H2D 与输入构造。该口径衡量稳态计算吞吐。含输入构造与 H2D 的端到端原始计时保存在 `timings/timings.csv`，可审计。环境：CANN 9.2.0-beta.2；Ascend 910B3 64 GB HBM；Kunpeng-920 192 核（单核基线）；1.5 TiB 内存；AsNumpy 0.4.1.dev12+gc9115e8ce.d20261031；warmup 5 次、iters 20 次取中位数。

## 4. 结果摘要

| 指标 | 数值 |
| --- | --- |
| 覆盖无缺口样例 | 71 |
| 规模化计时用例 | 72 |
| 给出加速比的计算样例 | 66 |
| 加速比范围 | 1.34x–612.41x |
| 加速比中位数 | 11.06x |
| 数值对拍通过 | 60 / 71 |
| 契约/语义样例（不计加速比） | 5 |

### 加速比前 15 名

| # | 用例 | CPU / NPU (ms) | 加速比 |
| ---: | --- | ---: | ---: |
| 1 | `A009_matrix_chain_2048` | 1225.11 / 2.00 | **612.41x** |
| 2 | `A007_matrix_product_2048` | 608.88 / 1.68 | **361.58x** |
| 3 | `B015_fortran_matmul_2048` | 634.30 / 1.79 | **355.01x** |
| 4 | `H015_real_vecdot_16384x256` | 91.98 / 0.56 | **164.79x** |
| 5 | `C013_markov_relaxation_4096x1024` | 2846.40 / 22.55 | **126.22x** |
| 6 | `S7_row_max_4096` | 16.92 / 0.19 | **88.97x** |
| 7 | `A008_matrix_activation_2048` | 647.52 / 7.70 | **84.06x** |
| 8 | `S6_row_sum_4096` | 11.91 / 0.20 | **58.58x** |
| 9 | `I001_cube_root_4m` | 113.37 / 2.02 | **56.10x** |
| 10 | `S4_residual_gemm_1024` | 363.09 / 6.98 | **52.03x** |
| 11 | `C002_mlp_inference_65536x128` | 565.85 / 13.26 | **42.67x** |
| 12 | `H009_variance_64x256x256` | 37.39 / 1.03 | **36.19x** |
| 13 | `B019_boolean_keepdims_512x256x64` | 12.88 / 0.36 | **35.72x** |
| 14 | `D1001_signed_compression_16384x256` | 436.57 / 12.38 | **35.27x** |
| 15 | `F007_sort_kind_2048x2048` | 270.06 / 8.19 | **32.98x** |

## 5. 数值边界

60 / 71 例在 `rtol=1e-4, atol=1e-5` 内一致。其余样例的差异均为 float32 归约/累加顺序（含其在 round 边界上的放大）或 IEEE 边界语义差异，不是迁移错误：

- `A009_matrix_chain_2048`：max abs diff 1.611e-02。
- `A007_matrix_product_2048`：max abs diff 3.967e-04。
- `B015_fortran_matmul_2048`：max abs diff 1.000e+00。
- `H015_real_vecdot_16384x256`：max abs diff 1.709e-03。
- `A008_matrix_activation_2048`：max abs diff 1.683e-04。
- `S4_residual_gemm_1024`：max abs diff 9.918e-05。
- `S2_heat_diffusion_1024`：max abs diff 4.578e-05。
- `A014_prefix_sum_4096x1024`：max abs diff 1.373e-04。
- `D2002_round_alias_16384x256`：max abs diff 1.000e-02。
- `B011_nan_inf_extrema_4m`：max abs diff 0.000e+00。（NaN 传播差异，逐元素 NaN 掩码不一致）
- `B018_signed_zero_4m`：max abs diff 1.000e+00。（signed zero 符号位差异）

绝对差随输出量级变化：例如 B015 的 1.0 绝对差对应约 4e6 输出量级的 2.4e-7 相对差；D2002 的 0.01 是 `round(decimals=2)` 在 .005 边界上放大上游 float32 归约差异。逐例 `max_abs_diff` / `max_rel_diff` 见 `timings/`。

B011（fmax 对 NaN 的传播）与 B018（copysign 的 -0.0）在归档阶段已登记为 CANN 契约差异，不属于迁移缺陷。

## 6. 复现

```bash
python tools/bench_no_gap_archive.py \
  --archive examples/scaled_benchmark_20261112 \
  --out runs/20261112-scaled-timing \
  --warmup 5 --iters 20
```

逐例 CPU/NPU JSON、worker 日志与 SHA-256 见 `timings/`；用例文件哈希见 `MANIFEST.json`；机器可读汇总见 `results_summary.json`。
