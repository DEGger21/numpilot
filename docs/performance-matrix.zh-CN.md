# 性能矩阵（Phase 9）

本页记录 Phase 9 的性能验收口径、实测数据、数值 gate 和证据边界。性能数据由
`tools/performance_matrix.py` 在真实 Ascend 设备上生成，机器可读原件保存在仓库外的
`runs/phase9-performance-20261105/matrix-02/perf-matrix.json`。

## 结论

- 测量日期：2026-11-05；矩阵 `matrix-02`；覆盖 `matmul` 和 `sin_matmul` 两个 profile，
  规模 64/128/256/512/1024，共 10 行、170 个数值检查，全部通过。
- `correctness_ok=true`，`status=profile_required`，`performance_valid=false`。
  `sin` 和 `matmul` 的运行时 provenance 仍为 `UNKNOWN`（未审计的 AsNumpy 版本），
  按 [策略](#发布边界) 所有加速比只作诊断，不是可对外宣称的 strict-NPU 加速比。
- 单逻辑 CPU 核对单卡 NPU 的实测趋势（中位数）：
  - `matmul` resident 从 128 起超过 CPU（1.60x），256/512/1024 为
    10.73x/78.23x/483.90x。
  - `sin_matmul` resident 从 256 起超过 CPU（4.25x），512/1024 为 18.32x/46.75x。
  - warm end-to-end（每次含 H2D、前向和 D2H）`matmul` 从 256 起超过 CPU
    （2.39x），512 为 8.12x，1024 为 27.79x；`sin_matmul` 在 256/512/1024 为
    1.95x/7.12x/23.35x。
  - cold end-to-end 全部为 0.03x–0.04x：NPU 冷进程启动约 9.54–9.56 s，主要花在
    CANN 初始化；CPU 冷进程约 0.29–0.42 s。冷启动场景 CPU 明显占优。
- 小规模 64 的 NPU 固定开销约 0.12 ms（`matmul`）和 0.31 ms（`sin_matmul`），
  因此 resident/warm 都低于单核 CPU；64 不是 NPU 优势区间。
- 所有结论都是在“单逻辑 CPU 核 + BLAS 单线程”基准下得到的。多核 CPU、不同 BLAS
  或不同 Ascend 型号会得到不同交点，不能直接外推。

## 测量口径

### 环境

| 项目 | 值 |
| --- | --- |
| 设备 | Ascend910B3（device 0） |
| CPU | aarch64，192 个逻辑 CPU |
| 内核 | Linux 5.10.0-182.0.0.95.r3450_281.hce2.aarch64 |
| CANN | 9.2.0-beta.2 |
| Python | 3.12.14 |
| AsNumpy | 0.4.1.post0+numpilot.bee27ba2 |
| 运行时审计 | `unaudited-asnumpy-version`，`status=audit_required` |
| 计时器 | `time.perf_counter` |

### 隔离与线程

- 每个样本在独立 worker 子进程中运行；CPU worker 和 NPU worker 都绑定到同一个
  逻辑核（本次为 core 0），receipt 中 `affinity=[0]`。
- worker 强制 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、
  `MKL_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`、`PYTHONHASHSEED=0`；
  worker 环境会剔除含 `TOKEN/SECRET/PASSWORD/API_KEY/ACCESS_KEY/PRIVATE` 的变量。
- NPU worker 在导入 NumPy/AsNumpy 之前完成绑核，保证 host 侧比较基准一致。

### 三种 scope

| scope | 预算 | 定义 |
| --- | --- | --- |
| `cpu_warm` | 5 预热 + 20 重复 | 进程内 NumPy 前向，单核、BLAS 单线程 |
| `cpu_cold` | 7 个新进程 | 每个样本一个全新进程，父进程记录进程墙钟 |
| `npu_resident` | 3 预热 + 15 重复 | 输入只在 setup 上传一次；计时段只重复前向，无 H2D/D2H；计时后做一次 D2H 供校验 |
| `npu_warm_e2e` | 3 预热 + 15 重复 | 每个样本执行 H2D → 前向 → D2H |
| `npu_cold_e2e` | 7 个新进程 | 每个样本一个全新进程，父进程记录进程墙钟（含 CANN 初始化） |

- resident 的 `measured_transfers` 为 1 次 D2H（4 MiB，1024 规模），计时段
  H2D/D2H 均为 0；setup 为 2 次 H2D（8 MiB）。这证明 resident 数值没有把拷贝时间
  计入。
- 每个 NPU 样本在计时前后调用 `aclrtSynchronizeDevice`；同步库和 API 名写入
  receipt。
- 比率 = CPU 中位数 / NPU 中位数；同时报告 p10/p90、标准差、最小/最大值和
  2000 次 bootstrap 的 95% 置信区间。
- 只有 bootstrap 区间不跨 1.0、且保守估计 `cpu_p10 / npu_p90 >= 1.1`、且满足对应
  阈值（resident 1.5x、warm 1.2x）时，比率才会标记为 publishable；本矩阵没有
  publishable 比率（provenance 不满足）。

### 数值 gate

GEMM 的输出不能沿用逐元素 `atol + rtol*|reference|` 规则：1024 维归约的绝对值误差
会随规模增长，而逐元素相对误差在结果接近 0 时没有意义。Phase 9 使用 float64 oracle：

1. 在 float64 中计算参考结果；
2. 相对 Frobenius 误差 `||actual-reference||_F / ||reference||_F <= 1e-5`；
3. 最大绝对误差 `<= max(1e-5, 2*eps32*size*max|A|*max|B|)`；
4. NPU 相对误差 `<= max(2*cpu_relative_frobenius, 1e-6)`。

本次 170 个检查全部通过。各行最大误差：

| profile | size | max abs diff | max relative Frobenius | max-abs 上限 |
| --- | ---: | ---: | ---: | ---: |
| matmul | 64 | 5.28e-06 | 1.43e-07 | 2.25e-04 |
| matmul | 128 | 1.72e-05 | 2.04e-07 | 5.68e-04 |
| matmul | 256 | 4.18e-05 | 2.88e-07 | 1.16e-03 |
| matmul | 512 | 1.28e-04 | 4.06e-07 | 2.97e-03 |
| matmul | 1024 | 2.09e-04 | 5.73e-07 | 5.58e-03 |
| sin_matmul | 64 | 4.57e-06 | 1.42e-07 | 5.70e-05 |
| sin_matmul | 128 | 1.54e-05 | 2.03e-07 | 1.20e-04 |
| sin_matmul | 256 | 3.78e-05 | 2.89e-07 | 2.54e-04 |
| sin_matmul | 512 | 5.62e-05 | 4.06e-07 | 5.58e-04 |
| sin_matmul | 1024 | 1.77e-04 | 5.74e-07 | 1.22e-03 |

`matrix-01` 使用旧的逐元素 float32 gate，在 512/1024 把 NPU 输出判为不一致。
用 float64 oracle 复核后，NPU 的最大绝对误差（512 为 4.36e-05、1024 为 1.11e-04）
低于同规模 NumPy float32 相对 float64 的误差（1.28e-04、2.09e-04），说明失败来自
gate 不适合大归约，而不是候选结果错误。`matrix-01` 作为被否决的测量保留，最终
候选为 `matrix-02`。

## 结果

### 汇总（中位数，单位 ms）

#### matmul

| size | CPU warm | NPU resident | NPU warm E2E | CPU cold | NPU cold E2E | resident | warm | cold | label |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 0.0406 | 0.1198 | 0.3012 | 295.03 | 9553.48 | 0.34x | 0.13x | 0.03x | CPU_PREFERRED |
| 128 | 0.1900 | 0.1186 | 0.3221 | 285.75 | 9546.54 | 1.60x | 0.59x | 0.03x | RESIDENT_GAIN |
| 256 | 1.30 | 0.1215 | 0.5448 | 304.09 | 9564.27 | 10.73x | 2.39x | 0.03x | DIAGNOSTIC |
| 512 | 9.70 | 0.1240 | 1.19 | 321.62 | 9558.29 | 78.23x | 8.12x | 0.03x | DIAGNOSTIC |
| 1024 | 76.93 | 0.1590 | 2.77 | 412.85 | 9552.21 | 483.90x | 27.79x | 0.04x | DIAGNOSTIC |

#### sin_matmul

| size | CPU warm | NPU resident | NPU warm E2E | CPU cold | NPU cold E2E | resident | warm | cold | label |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 0.0641 | 0.3060 | 0.4904 | 290.54 | 9540.85 | 0.21x | 0.13x | 0.03x | CPU_PREFERRED |
| 128 | 0.2846 | 0.3144 | 0.5484 | 290.74 | 9550.20 | 0.90x | 0.52x | 0.03x | CPU_PREFERRED |
| 256 | 1.71 | 0.4026 | 0.8799 | 296.06 | 9537.78 | 4.25x | 1.95x | 0.03x | DIAGNOSTIC |
| 512 | 11.47 | 0.6263 | 1.61 | 314.26 | 9536.89 | 18.32x | 7.12x | 0.03x | DIAGNOSTIC |
| 1024 | 85.69 | 1.83 | 3.67 | 418.15 | 9551.83 | 46.75x | 23.35x | 0.04x | DIAGNOSTIC |

标签含义：

- `CPU_PREFERRED`：NPU warm E2E 低于 CPU；
- `RESIDENT_GAIN`：resident 达到 1.5x，但 warm E2E 未达到 1.2x；
- `DIAGNOSTIC`：数值和测量有效，但 strict-NPU provenance 不完整，不能作为公开结论；
- `FAILED`：数值 gate 失败，不输出任何 speedup。

### 门槛与交点

| profile | resident >=1.0 起点 | resident >=1.5 起点 | warm >=1.0 起点 | warm >=1.2 起点 | cold >=1.0 起点 |
| --- | ---: | ---: | ---: | ---: | ---: |
| matmul | 128 | 128 | 256 | 256 | 无 |
| sin_matmul | 256 | 256 | 256 | 256 | 无 |

### 详细统计

每组都列出预热次数、重复次数、中位数、p10、p90、标准差、最小值和最大值，单位 ms。
cold 行为 7 个全新进程的墙钟，`internal_forward_samples_ms` 另行保存在矩阵 JSON 中。

#### matmul 详细统计

| size | mode | warmup | repeats | median | p10 | p90 | stdev | min | max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | cpu_warm | 5 | 20 | 0.0406 | 0.0404 | 0.0413 | 0.0023 | 0.0404 | 0.0506 |
| 64 | npu_resident | 3 | 15 | 0.1198 | 0.1161 | 0.1298 | 0.0064 | 0.1133 | 0.1374 |
| 64 | npu_warm_e2e | 3 | 15 | 0.3012 | 0.2974 | 0.3189 | 0.0093 | 0.2972 | 0.3262 |
| 64 | cpu_cold | 0 | 7 | 295.03 | 293.92 | 297.01 | 1.42 | 293.88 | 297.24 |
| 64 | npu_cold_e2e | 0 | 7 | 9553.48 | 9533.89 | 9560.47 | 12.53 | 9530.63 | 9561.00 |
| 128 | cpu_warm | 5 | 20 | 0.1900 | 0.1893 | 0.2157 | 0.0185 | 0.1889 | 0.2639 |
| 128 | npu_resident | 3 | 15 | 0.1186 | 0.1119 | 0.1279 | 0.0120 | 0.1108 | 0.1598 |
| 128 | npu_warm_e2e | 3 | 15 | 0.3221 | 0.3172 | 0.3384 | 0.0127 | 0.3160 | 0.3649 |
| 128 | cpu_cold | 0 | 7 | 285.75 | 284.87 | 292.06 | 3.29 | 284.43 | 292.06 |
| 128 | npu_cold_e2e | 0 | 7 | 9546.54 | 8545.63 | 9553.02 | 489.89 | 8533.77 | 9556.06 |
| 256 | cpu_warm | 5 | 20 | 1.30 | 1.29 | 1.35 | 0.0363 | 1.29 | 1.45 |
| 256 | npu_resident | 3 | 15 | 0.1215 | 0.1160 | 0.1308 | 0.0065 | 0.1140 | 0.1378 |
| 256 | npu_warm_e2e | 3 | 15 | 0.5448 | 0.5385 | 0.5719 | 0.0300 | 0.5343 | 0.6557 |
| 256 | cpu_cold | 0 | 7 | 304.09 | 301.15 | 304.99 | 1.94 | 299.83 | 305.58 |
| 256 | npu_cold_e2e | 0 | 7 | 9564.27 | 9140.65 | 9567.92 | 387.44 | 8535.06 | 9569.89 |
| 512 | cpu_warm | 5 | 20 | 9.70 | 9.68 | 9.75 | 0.0915 | 9.66 | 10.10 |
| 512 | npu_resident | 3 | 15 | 0.1240 | 0.1210 | 0.1382 | 0.0190 | 0.1205 | 0.1964 |
| 512 | npu_warm_e2e | 3 | 15 | 1.19 | 1.18 | 1.22 | 0.0793 | 1.17 | 1.50 |
| 512 | cpu_cold | 0 | 7 | 321.62 | 320.12 | 328.04 | 6.06 | 318.71 | 336.89 |
| 512 | npu_cold_e2e | 0 | 7 | 9558.29 | 9535.83 | 9965.33 | 380.20 | 9527.24 | 10556.68 |
| 1024 | cpu_warm | 5 | 20 | 76.93 | 76.59 | 77.88 | 0.6387 | 76.51 | 79.09 |
| 1024 | npu_resident | 3 | 15 | 0.1590 | 0.1538 | 0.1762 | 0.0193 | 0.1524 | 0.2300 |
| 1024 | npu_warm_e2e | 3 | 15 | 2.77 | 2.73 | 2.81 | 0.2570 | 2.72 | 3.76 |
| 1024 | cpu_cold | 0 | 7 | 412.85 | 412.12 | 421.76 | 5.34 | 411.91 | 426.59 |
| 1024 | npu_cold_e2e | 0 | 7 | 9552.21 | 9144.52 | 9554.79 | 381.38 | 8542.71 | 9555.72 |

#### sin_matmul 详细统计

| size | mode | warmup | repeats | median | p10 | p90 | stdev | min | max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | cpu_warm | 5 | 20 | 0.0641 | 0.0637 | 0.0672 | 0.0112 | 0.0635 | 0.1143 |
| 64 | npu_resident | 3 | 15 | 0.3060 | 0.2994 | 0.3354 | 0.0149 | 0.2964 | 0.3392 |
| 64 | npu_warm_e2e | 3 | 15 | 0.4904 | 0.4818 | 0.5097 | 0.0130 | 0.4782 | 0.5278 |
| 64 | cpu_cold | 0 | 7 | 290.54 | 289.65 | 293.35 | 2.55 | 288.79 | 296.76 |
| 64 | npu_cold_e2e | 0 | 7 | 9540.85 | 9164.82 | 9555.28 | 350.53 | 8617.44 | 9560.58 |
| 128 | cpu_warm | 5 | 20 | 0.2846 | 0.2837 | 0.3145 | 0.0122 | 0.2835 | 0.3213 |
| 128 | npu_resident | 3 | 15 | 0.3144 | 0.3089 | 0.3371 | 0.0118 | 0.3078 | 0.3462 |
| 128 | npu_warm_e2e | 3 | 15 | 0.5484 | 0.5358 | 0.5659 | 0.0134 | 0.5349 | 0.5817 |
| 128 | cpu_cold | 0 | 7 | 290.74 | 286.56 | 292.19 | 2.50 | 286.17 | 292.63 |
| 128 | npu_cold_e2e | 0 | 7 | 9550.20 | 9139.79 | 9564.05 | 383.86 | 8537.48 | 9566.50 |
| 256 | cpu_warm | 5 | 20 | 1.71 | 1.71 | 1.74 | 0.0312 | 1.71 | 1.85 |
| 256 | npu_resident | 3 | 15 | 0.4026 | 0.3946 | 0.4234 | 0.0145 | 0.3937 | 0.4454 |
| 256 | npu_warm_e2e | 3 | 15 | 0.8799 | 0.8648 | 0.8991 | 0.0476 | 0.8558 | 1.06 |
| 256 | cpu_cold | 0 | 7 | 296.06 | 294.81 | 298.06 | 1.96 | 293.64 | 300.19 |
| 256 | npu_cold_e2e | 0 | 7 | 9537.78 | 8511.68 | 9543.04 | 546.69 | 8467.73 | 9543.77 |
| 512 | cpu_warm | 5 | 20 | 11.47 | 11.46 | 11.50 | 0.0977 | 11.44 | 11.90 |
| 512 | npu_resident | 3 | 15 | 0.6263 | 0.6063 | 0.6636 | 0.0340 | 0.5962 | 0.7342 |
| 512 | npu_warm_e2e | 3 | 15 | 1.61 | 1.56 | 1.75 | 0.0941 | 1.56 | 1.90 |
| 512 | cpu_cold | 0 | 7 | 314.26 | 313.62 | 316.43 | 1.37 | 313.51 | 317.39 |
| 512 | npu_cold_e2e | 0 | 7 | 9536.89 | 8563.75 | 9553.60 | 478.98 | 8563.18 | 9564.68 |
| 1024 | cpu_warm | 5 | 20 | 85.69 | 85.19 | 87.05 | 0.8598 | 84.78 | 88.07 |
| 1024 | npu_resident | 3 | 15 | 1.83 | 1.70 | 2.17 | 0.1990 | 1.68 | 2.33 |
| 1024 | npu_warm_e2e | 3 | 15 | 3.67 | 3.63 | 3.74 | 0.2871 | 3.63 | 4.78 |
| 1024 | cpu_cold | 0 | 7 | 418.15 | 417.30 | 419.87 | 1.32 | 416.38 | 420.60 |
| 1024 | npu_cold_e2e | 0 | 7 | 9551.83 | 9153.60 | 9558.37 | 377.02 | 8557.41 | 9558.53 |

## resident 计时可信度核查

resident 的 NPU 时间在 64–512 几乎不变（约 0.119–0.124 ms），1024 才到 0.159 ms。
这既可能是启动开销占主导，也可能是同步没有生效。为排除“看起来很快”的测量错误，
做了两组独立核查：

1. **相同输入 vs 不同输入**：在 1024 规模，重复调用同一对输入 20 次耗时 2.708 ms
   （0.135 ms/次）；轮换 5 对不同的输入 20 次耗时 2.713 ms（0.136 ms/次）。
   两条路径没有差异，说明不存在结果缓存。
2. **规模扩展**：1024/2048/4096 的单次调用为 0.202/0.408/2.674 ms。4096 的
   68.7 GFLOP 在 2.674 ms 内完成，约 25.7 TFLOPS，说明计时包含真实设备计算；
   1024 以下被约 0.12 ms 的固定启动/同步开销掩盖。

`aclrtSynchronizeDevice` 在计时后约 0.006 ms 返回，因为当前 AsNumpy 的
`aclnnMatmul` 调用在 host 侧同步完成（调用返回时 log 已打印 `aclnnMatmul completed`，
且随后 D2H 立即得到正确结果）。因此 resident 数据有效，但这条结论绑定当前
`0.4.1.post0+numpilot.bee27ba2` 运行时；接入审计后的 AsNumpy 或异步执行路径时，
必须重新验证同步语义后再使用这些数字。

## 发布边界

- 不发布加速比：`sin` 和 `matmul` 的 provenance 是 `UNKNOWN`，
  `audit_id=unaudited-asnumpy-version`，矩阵 `performance_valid=false`。矩阵中的
  比率只用于展示趋势和定位交点。
- 官方 ACLNN overlay 只覆盖 `concatenate`、`ones`、`zeros`，不覆盖 `sin` 或
  `matmul`；当前没有已审计 wheel。
- 获得 publishable 比率的条件：安装审计后的 AsNumpy wheel、两个算子都返回
  `native_npu` provenance、重跑本矩阵，并满足 bootstrap 区间和保守 p10/p90 gate。
- `sin_matmul` 的 resident 同时包含 `sin` 和 `matmul` 两个算子，不能拆成单个
  `sin` 的加速比。
- cold 数据的 9.5 s 来自 CANN 初始化，不代表稳态推理成本；对外演示应使用 warm
  或 resident 口径，并明确标注。

## 复现

```bash
cd <repo>
python tools/performance_matrix.py \
  --out runs/phase9-performance-20261105/matrix-02 \
  --profiles matmul,sin_matmul \
  --sizes 64,128,256,512,1024 \
  --core 0 \
  --cpu-warmup 5 --cpu-repeats 20 \
  --npu-warmup 3 --npu-repeats 15 \
  --cold-samples 7
```

产物：

- `perf-matrix.json`：完整测量、provenance、环境、工具身份和 170 个检查；
- `PERF-MATRIX.md`：矩阵摘要；
- `workers/<profile>/<size>/<mode>/receipt.json` 和 `output.npy`：逐样本回执与输出；
- `SHA256SUMS`：全部产物的 SHA-256，`sha256sum -c SHA256SUMS` 通过。

冻结身份：

| 项目 | SHA-256 |
| --- | --- |
| `tools/performance_matrix.py` | `ba29fcd9ce70dff337a9b3db3db1436036d4a802885cc415fa738905a3530182` |
| `numpilot/perf_matrix.py` | `c1a2e4bda982cae0978246ba611743a32d90e976cb368b834ff07e4ed07622e3` |
| repo HEAD | `74801d145d9f22152d9850994d07225381bfeb27` |

## 当前限制

- 只有 10 行、两个 profile；尚未覆盖更多 dtype、非方阵、批量 matmul。
- strict-NPU provenance 未满足，因此所有结果仍是 `profile_required`。
- 单核 CPU 基准会放大 NPU 的相对优势；与多核 CPU 比较时需要另建口径。
- cold 测量包含 Python 和 CANN 初始化，适合说明冷启动成本，不适合比较算子吞吐。
