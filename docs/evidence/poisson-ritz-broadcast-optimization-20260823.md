# Poisson Ritz 广播与公共子表达式优化证据

日期：2026-08-23 UTC

## 结论

在与原始规模基线相同的 Ascend 910B4、CANN 9.0.0、AsNumpy move wheel、qualification
overlay、`48×48` 内点、四边各 48 点、`7×7` RBF 和 3000 轮配置下，广播/CSE 优化把
NPU device-resident 中位每轮时延从 `47.41388170514256 ms` 降至
`38.06900561321527 ms`：优化版为基线的 `1.245472×`，时延下降 `19.7092%`。

这仍不是 NPU 加速 CPU。优化后的单线程 CPU 中位每轮为 `13.2244145963341 ms`，CPU
约快 `2.878691×`。本报告只声明同一 Poisson 实现的前后优化收益与当前 CPU/NPU 关系。

| 项目 | 基线 | 广播/CSE 优化 | 变化 |
|---|---:|---:|---:|
| 每轮公开 API / kernel | 117 | 101 | `-13.6752%` |
| `outer` | 14 | 2 | `-85.7143%` |
| `multiply` | 43 | 40 | `-6.9767%` |
| `sqrt` | 2 | 1 | `-50%` |
| NPU 中位每轮 | `47.41388170514256 ms` | `38.06900561321527 ms` | `-19.7092%` |
| CPU 中位每轮 | `17.752825701609254 ms` | `13.2244145963341 ms` | `-25.5081%` |
| NPU 相对 CPU | CPU 快 `2.670779×` | CPU 快 `2.878691×` | NPU 未反超 |

## 优化内容与语义

新增的
[`examples/poisson_ritz_iteration_broadcast.py`](../../examples/poisson_ritz_iteration_broadcast.py)
保留原实现的 RBF、Laplacian、Smooth L1 损失、解析梯度和四组参数更新，只改变中间张量的
形成方式：

- 内点和边界坐标在 `build_inputs` 中构造成 `(points, 1)` 连续列向量，与 RBF 参数向量
  直接广播，省去坐标、中心、宽度和权重对应的 12 个 `outer`；
- 每轮共享一组 width 的平方、平方根、三次方、四次方和五次方，删除重复 `sqrt/multiply`；
- 复用 `phi_in × laplace_factor_in` 和 `phi_bd × boundary residual gradient`，删除两个
  重复乘法；
- 两个真正需要把一维残差梯度扩成矩阵的 `outer` 保留，不引入未审计的
  reshape/transpose 设备算子；
- 输入数从 30 降为 28，参数仍位于 `inputs[6:10]`，轮间继续只接力设备参数。

CPU 上默认规模连续 250 轮及原始规模抽查均与基线逐输出位级一致。最终证据中，优化前后
CPU 输出 NPZ 的 SHA-256 同为
`53b376f2cde6383f34cebf9993d952b174d0cf4df780143a145897bafe817815`，NPU 输出 NPZ 的
SHA-256 同为 `ffb92d06d0f251b43d487b979bbb4e7f6ab6c197319af4f10f9d63365cadc615`。

## 最终数值与计时

优化版 3000 轮结果：

| 项目 | CPU | NPU |
|---|---:|---:|
| loss | `8.553647994995117 → 1.5687222480773926` | `8.553648948669434 → 1.5687224864959717` |
| 中位每轮 | `13.2244145963341 ms` | `38.06900561321527 ms` |

CPU/NPU 最终五个返回值共 197 个元素通过 float32 `atol=1e-5, rtol=1e-4` 门；最大绝对差
为 `2.0325183868408203e-05`，最大相对差为 `0.0002941576240118593`。这些数值与基线证据
完全相同。

CPU 7 个测量样本（每个样本 10 轮，毫秒）：

```text
131.17122906260192
130.4700430482626
133.7010560091585
133.18462180905044
132.244145963341
134.18418890796602
131.44635991193354
```

NPU 7 个测量样本（每个样本 10 轮，毫秒）：

```text
395.2850771602243
393.0136919952929
394.34411004185677
380.6900561321527
378.806872991845
369.3436770699918
363.49578690715134
```

2 个 NPU 预热样本为 `1130.1859351806343 ms` 和 `424.7140809893608 ms`，均按预先声明
规则排除；7 个测量样本没有选择性删值。CPU 与 NPU 仍由独立子进程运行，CPU BLAS 运行时
实测为 1 线程，H2D 在计时前、D2H 在计时与收敛后，NPU 计时终点为
`aclrtSynchronizeDevice`。

## 同规模 profiler 门

与最终源码、迁移产物、runner、overlay、capability audit 和规模完全一致的 5 轮短运行
由 CANN `msprof` 采集：

- 505 个 runtime 调用与 505 条 kernel task 逐 API 完全一致；
- `outer=10`、`multiply=200`、`sqrt=5`，与 `2/40/1 × 5` 一致；
- `ApplyGradientDescent=20`、`NumPilotSmoothMeanLoss=10`；
- `ai_core_op_summary.db` 中 505 个 kernel 的 `duration_time` 合计 `2.880715 ms`，平均
  `5.704 µs/kernel`；
- 无未知 kernel 前缀；
- 34 次 `aclrtMemcpy`、共 `1,906,052 bytes`，恰等于 28 H2D + 1 个初始 loss D2H +
  5 个最终返回值 D2H；
- strict-NPU、运行时 profile 和目标函数身份门全部通过。

profiler 运行只用于 kernel/传输门，不发布 profiler 下的性能数字。最终时延来自独立的非
profiler 重复运行。

## 固定身份

```text
runner SHA-256:          df3072f7edb0e43cb72a8064dc23a97f6d38aeb37fdc9703c1f7f2c93bf0298a
source SHA-256:          0b4719834998678cccd761597970c14169b97fc2d7a32a908978bb6ab5da0abe
migrated SHA-256:        5758e44a9751b0c5c25f82a83617998587473f2d961d8329618ff8446f45e6e6
overlay SHA-256:         1c5aa2ae5c4ebf2db8d542f24defd9c7ca42ccc4d87b6e10fafc60c4eb708789
final evidence:          adb026e219df1ad92aba52de5751daa45e2ee5531c39a6756efdba4d030e9104
profiler audit:          0a09d4eaa462daea57370156b805369ec78921dcb9695563bd5246e391609b94
runtime.db:              66aa3835c8daeb80acb861b116530a2922f209f388b172153338301e93c9d358
rts_track.db:            d5ed8aed6125212e2ab77fbf6df8870a3f21bf8dde33f60471709c3f48d5611f
ai_core_op_summary.db:   e334f31ecc27f8fedd2809a6775bfaadf5d3a4eade27a303e5967066bb2d4e8c
```

精选证据保存在
[`evidence/poisson-ritz-p0-20260821/performance-broadcast-optimized/`](../../evidence/poisson-ritz-p0-20260821/performance-broadcast-optimized/)。
完整原始运行树保存在：

```text
/home/ma-user/work/numpilot-gap-runs/poisson-ritz-p0-20260821/direct-local-v5/
  poisson-broadcast-iteration-strict-npu-v3/
  poisson-broadcast-original-scale-profiler-v2/
  poisson-broadcast-original-scale-v1-3000/
```

## 适用边界与下一步

收益只适用于本次硬件、软件身份、float32 profile、固定规模和 device-resident 计时范围。
它不包含首次 H2D、最终 D2H、迁移、构建或安装，也不能外推到其他网络。

剩余 101 个细粒度调用中，5 轮 profiler 的 AI Core kernel 本体合计仅 `2.880715 ms`，
远小于端到端调度时间；继续显著缩小 CPU 差距需要图化/融合或批量提交，而不是继续堆叠
等量逐元素替换。
