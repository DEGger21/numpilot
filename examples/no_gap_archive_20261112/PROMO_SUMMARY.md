# NumPilot 无缺口迁移：可引用口径（2026-11-12）

> 本页只列已验证数字。引用时必须带口径与边界，不得省略。

## 1. 自动化迁移

- 71/71 个无缺口 NumPy 程序由 NumPilot 自动迁移为 AsNumpy 程序；全部通过框架表面与数值门禁（shim 后端）；70/71 零人工修复。
- 端到端迁移总耗时 6.05 分钟（71 例，真实 deepseek-flash 模型，含执行与数值验证）。
- 参数化人工估算：29.8–155.6 小时（3.7–19.4 人日），即框架快约 295x–1542x。该区间是估算，不是实测。

## 2. 真实 NPU 性能（71 例规模化覆盖）

基线：单核 NumPy（BLAS/OMP=1）；设备：Ascend 910B3 + CANN 9.2.0-beta.2 + AsNumpy `0.4.1.dev12+gc9115e8ce.d20261031`；warmup 5，iters 20，取中位数。

稳态加速比 = 单核 CPU `forward()` 中位耗时 ÷ NPU 设备常驻 `forward()` 中位耗时（输入预置，NPU 含结果回传，不含 H2D 与输入构造）。

- 71 个无缺口样例全部规模化，共 72 个计时用例；66 个计算样例给出加速比，范围 1.34x–612.41x，中位 11.06x。
- 5 个契约/语义样例（0 维、空轴、dtype 提升、float64、非连续布局等）以 “—” 标注，衡量正确性而非加速比。
- 严格数值阈值 `rtol=1e-4, atol=1e-5` 下 60/71 一致；其余为 float32 大规模归约/累加顺序（含 round 边界放大）或已登记的 IEEE 边界语义差异。

### 加速比前 10 名

| 用例 | 规模 | 稳态加速 | CPU / NPU (ms) |
| --- | --- | ---: | ---: |
| A009_matrix_chain | two 2048 x 2048 float32 matmuls; 34.4 GFLOP | **612.41x** | 1225.11 / 2.00 |
| A007_matrix_product | 2048 x 2048 float32 matmul; 17.2 GFLOP | **361.58x** | 608.88 / 1.68 |
| B015_fortran_matmul | 2048 x 2048 Fortran-ordered float32 matmul | **355.01x** | 634.30 / 1.79 |
| H015_real_vecdot | 16,384 x 256 float32; vecdot over both axes | **164.79x** | 91.98 / 0.56 |
| C013_markov_relaxation | 4,096 states x 1,024 classes; eight relaxation steps | **126.22x** | 2846.40 / 22.55 |
| A006_row_max | A006 row-max graph on a 4096x4096 float32 matrix | **88.97x** | 16.92 / 0.19 |
| A008_matrix_activation | 2048 x 2048 float32 matmul plus tanh | **84.06x** | 647.52 / 7.70 |
| A004_row_sum | A004 row-sum graph on a 4096x4096 float32 matrix | **58.58x** | 11.91 / 0.20 |
| I001_cube_root | 4,000,000 float32 elements; signed zero retained | **56.10x** | 113.37 / 2.02 |
| C011_residual_gemm | C011 residual-GEMM graph with 1024x1024 projections | **52.03x** | 363.09 / 6.98 |

- 完整 71 行清单（源程序、迁移程序、替换 API、迁移耗时、加速比）见
  `examples/scaled_benchmark_20261112/BENCHMARK_TABLE.md`。
- 含输入构造与 H2D 的端到端原始计时见
  `examples/scaled_benchmark_20261112/timings/timings.csv`。

## 3. 不能说的话

- 不能说“71 个正确性用例平均加速 X 倍”：它们只有 1–528 个元素，计时由 NPU 启动与 H2D/D2H 开销主导，不是产品性能；对外统一引用规模化清单。
- 不能说规模化 71 例“全部逐位一致”或“全部通过严格 allclose”：严格阈值下 60/71 一致，其余是 float32 归约顺序差异，均已逐例记录。
- 不能把 shim 后端的 71/71 说成“真实 NPU 71/71 数值一致”；真实 NPU 上 B011（fmax NaN）、B018（copysign -0.0）是已登记的契约差异。
- 不能把标注为 “—” 的契约/语义样例计入平均加速比。

## 4. 建议一句话

“NumPilot 用真实大模型在 6.05 分钟内自动迁移 71 个 NumPy 程序；在昇腾 910B3 上对全部 71 个样例完成大规模改写，66 个计算样例相对单核 NumPy 取得 1.34x–612.41x 的稳态计算加速（中位 11.06x），单例迁移耗时 1.52–23.55 秒。”
