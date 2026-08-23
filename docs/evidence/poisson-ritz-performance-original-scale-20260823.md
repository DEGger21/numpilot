# Poisson Ritz 原始规模重复性能证据

日期：2026-08-23 UTC

## 结论

在本机 Ascend 910B4、CANN 9.0.0 和哈希绑定的 AsNumpy move wheel 上，Poisson Ritz
案例已完成原始 `48×48` 内部配点、四边各 48 点、`7×7` RBF、3000 轮训练的 strict-NPU
数值门，以及独立进程中的 2 次预热和 7 次重复样本。

本次合格结论不是 NPU 加速，而是：在“输入已驻留、每个样本执行 10 个完整训练迭代、
末尾同步、排除状态重置和 H2D/D2H”的口径下，单线程 CPU 约快 `2.670779×`。

| 项目 | CPU | NPU |
|---|---:|---:|
| 中位每轮时延 | `17.752825701609254 ms` | `47.41388170514256 ms` |
| 3000 轮 loss | `8.553647994995117 → 1.5687222480773926` | `8.553648948669434 → 1.5687224864959717` |
| 重复样本 | 7 | 7 |
| worker PID | 32164 | 32709 |

CPU/NPU 最终五个返回值共 197 个元素通过 float32 `atol + rtol` 门；最大绝对差为
`2.0325183868408203e-05`，最大相对差为 `0.0002941576240118593`。这里的训练门只声明
“最终 loss 有限且低于初始值”，不声称 3000 轮内 loss 单调下降。

## 计时与 CPU 线程口径

CPU 和 NPU 由同一个受信 runner 顺序启动为两个独立子进程。每个计时样本都从相同的
确定性初始参数开始；状态列表重置发生在计时开始前。

- CPU：同步 NumPy 完成作为计时终点；
- NPU：`aclrtSynchronizeDevice` 成功返回作为计时终点；
- 每个样本包含 10 个完整训练迭代；
- 2 个预热样本不进入中位数，随后保留 7 个有序原始样本；
- 30 个输入在 NPU 计时前上传；1 个初始 loss 和 5 个最终结果在全部计时与 3000 轮训练后下载；
- CPU 子进程启动前固定 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、
  `MKL_NUM_THREADS=1` 和 `NUMEXPR_NUM_THREADS=1`；
- `threadpoolctl` 在隔离 venv 中不可用，因此 runner 通过已加载 OpenBLAS 的只读运行时 API
  实测 `num_threads=1`，并记录精确共享库路径和 OpenBLAS 配置。

CPU 7 个样本（每个样本 10 轮，毫秒）：

```text
177.87755001336336
176.6154000069946
180.09314499795437
177.52825701609254
188.1626108661294
171.18827300146222
171.38164513744414
```

NPU 7 个样本（每个样本 10 轮，毫秒）：

```text
474.1388170514256
476.6133138909936
480.88073381222785
478.42158610001206
463.51567283272743
468.09972589835525
449.4409861508757
```

第一个 NPU 预热样本为 `1333.2930949982256 ms`，明显包含首次运行开销；它按预先声明的
规则排除，第二个预热样本为 `490.01046596094966 ms`。没有从 7 个测量样本中选择性删值。

## 同规模 profiler 门

性能计时不在 profiler 下执行。另一次与最终 runner SHA-256、源码、迁移产物、overlay、
capability audit 和规模完全一致的短运行由 CANN `msprof` 采集。该运行包含 5 个完整迭代，
因此预期 `117×5=585` 个公开 API 调用。

机器审计结果：

- 585 个 runtime 调用与 585 条 kernel task 逐 API 完全一致；
- `ApplyGradientDescent` 20 条，`NumPilotSmoothMeanLoss` 10 条；
- 无未知 kernel 前缀；
- 36 次 `aclrtMemcpy`、共 `1,916,036 bytes`，恰等于 30 H2D + 1 初始 loss D2H + 5 最终 D2H；
- strict-NPU、运行时 profile 与目标函数身份门全部通过。

profiler 的 compact `memcpy_direction` 仍全部为 `0`；方向结论继续依赖已完成的最小方向
探针和精确调用/字节账目，不把该字段强行解释成 ACL 公共枚举。

## 固定身份

```text
runner SHA-256:    e6409acbe6492e7706b5c77ffa9b1a202db86e70fcddec3655d4dbc140889f88
source SHA-256:    1deeeec0e5c32c96c4a33467687ce5181a6c8d2c018c6b81b3dc066e3e8beb1a
migrated SHA-256:  49779934abf65eececc5dac074d6ab858ef63ad23999052e531f25069248f195
overlay SHA-256:   1c5aa2ae5c4ebf2db8d542f24defd9c7ca42ccc4d87b6e10fafc60c4eb708789
final evidence:    fa69fd9f947474e317018b13882c6852524395947d7e36d6d45f42cfcbe4532e
profiler audit:    deb772db967a767ffcf6aa655612558d1083241366b68411b7d82bc72fac5132
runtime.db:        100484cd9929602ee83e001be7e8d0a407ff0dad35c47077d5f7ca0e8c99ae51
rts_track.db:      a58f69e867967f3730009a5afe8df380cb09ae0b42a6b11567a33e58795b3d22
```

精选证据保存在
[`evidence/poisson-ritz-p0-20260821/performance-original-scale/`](../../evidence/poisson-ritz-p0-20260821/performance-original-scale/)。
完整原始运行树保存在：

```text
/home/ma-user/work/numpilot-gap-runs/poisson-ritz-p0-20260821/direct-local-v5/
  poisson-performance-original-scale-v4-3000/
  poisson-performance-original-scale-profiler-v3/
```

## 适用边界

该结论只适用于本次硬件、软件身份、float32 profile、固定输入和上述 device-resident 计时
范围。它不是端到端时延，不包含首次 H2D、最终 D2H、模型生成、构建或安装，也不能外推到
其他规模、dtype 或科学计算网络。

当前一轮训练会发射 117 个较细粒度 kernel；在本规模上，调度与逐算子开销没有被计算量
覆盖，因此 CPU 明显更快。后续优化方向应是融合/图化和减少 launch 数，而不是把这份负收益
包装成 NPU 加速。

该方向的第一轮优化已经完成：广播/CSE 变体把调用数降到 101，NPU 中位时延下降
`19.71%`，且 3000 轮最终输出与本基线字节级相同。详见
[`poisson-ritz-broadcast-optimization-20260823.md`](poisson-ritz-broadcast-optimization-20260823.md)。
