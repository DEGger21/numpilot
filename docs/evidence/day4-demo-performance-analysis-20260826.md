# Demo 性能对比与问题分析（2026-08-26）

## 结论

“AsNumpy 版本理论上一定比 NumPy 快”不成立。当前机器上的证据呈现清晰的规模分界：

- 对 Case03/06/10 的真实 GLM 迁移脚本体，NPU 稳态分别比单核 NumPy 慢 `2.68x / 3.76x / 5.28x`。
- 对同三份真实脚本的 fresh-process 冷启动，NPU 分别慢 `19.61x / 21.85x / 19.58x`；这里主要测到 Python、import 和 CANN 初始化等固定成本。
- 对冻结的 Case06 百万级常驻设备四算子链，NPU 比单核 NumPy 快 `9.47x–11.44x`。这表明该冻结四算子链在这些规模上的设备驻留吞吐正常，但该定向 adapter 不是迁移脚本端到端结果。
- Case02 有真实语义兼容问题：`max_val` 从 NumPy 的 `float32` 漂移为 AsNumpy 的 `float64`，因此数值门禁失败，性能字段保持 `null`。

机器可读汇总见 `day4-demo-performance-results-20260826.json`。原始证据分别位于：

- `/home/ma-user/work/.numpilot-state/runs/day4-demo-steady-v2-20260826/evidence.json`
- `/home/ma-user/work/.numpilot-state/runs/day4-case06-scale-20260826/evidence.json`
- `/home/ma-user/work/.numpilot-state/runs/day4-demo-cold-v4-20260826/evidence.json`

三份权威证据都使用同一 host 逻辑核 `0`。steady/scale worker 内 OpenBLAS 实测为 1 线程；cold 子进程设置相同线程环境并绑定已通过的 steady receipt，但没有在每个短命子进程内重新查询 BLAS。NPU scale worker 本身有 9 个 OS 线程，但都受同一单核 affinity 约束，所以准确表述是“相同单 host-core affinity”，不是“整个 NPU 进程只有一个线程”。

## 真实迁移脚本稳态

计时边界是缓存 import 后的完整脚本体；每侧预热 2 次、测量 7 次，表中为 `p10 / median / p90`，单位 ms。`CPU/NPU` 大于 1 才表示 NPU 更快。

| Case | NumPy | AsNumpy | CPU/NPU | 结论 |
|---|---:|---:|---:|---|
| 03 | `0.694 / 0.699 / 0.711` | `1.855 / 1.875 / 1.890` | `0.373` | NPU 慢 2.68x |
| 06 | `0.493 / 0.497 / 0.528` | `1.844 / 1.869 / 1.941` | `0.266` | NPU 慢 3.76x |
| 10 | `0.368 / 0.371 / 0.380` | `1.939 / 1.961 / 1.999` | `0.189` | NPU 慢 5.28x |

Case04 数值一致，但属于 scoped migration，只保留诊断数据；Case01/08 含 fallback，不执行 NPU 性能计时。Case02 即使浮点差值仍在容差内，也因 dtype 不同严格失败，不能用“值差不多”绕过合同。

## Case06 固定规模

受信 adapter 冻结为 `power -> log -> exp -> equal`、`float32`、1M/4M/9M 元素。下表只表示常驻设备公共计算链，包含算子派发、kernel 和算子输出分配，排除输入分配、H2D、D2H 与输出释放。

| 元素数 | NumPy p10/median/p90 | AsNumpy p10/median/p90 | CPU/NPU median | 保守 CPU p10/NPU p90 |
|---:|---:|---:|---:|---:|
| 1M | `48.879 / 50.424 / 51.042` | `4.978 / 5.101 / 8.215` | `9.89x` | `5.95x` |
| 4M | `199.888 / 200.040 / 201.148` | `20.913 / 21.130 / 21.830` | `9.47x` | `9.16x` |
| 9M | `454.324 / 456.016 / 457.596` | `39.764 / 39.848 / 60.129` | `11.44x` | `7.56x` |

这个结果说明该冻结四算子链的大规模设备驻留路线没有失效。性能问题集中在固定成本、公共 wrapper、分配、同步和数据边界。

## 冷启动

每个样本是一个全新 Python 进程，从创建前计时到 exact script 退出；每侧 7 个独立进程，无预热。设备、驱动和文件系统缓存没有重置。

| Case | NumPy p10/median/p90 | AsNumpy p10/median/p90 | NPU 慢 |
|---|---:|---:|---:|
| 03 | `181.835 / 182.404 / 185.311` | `3561.714 / 3576.365 / 4347.016` | `19.61x` |
| 06 | `179.572 / 180.832 / 275.998` | `3569.542 / 3951.128 / 4945.515` | `21.85x` |
| 10 | `180.735 / 182.225 / 243.589` | `3553.374 / 3567.636 / 3851.331` | `19.58x` |

初版 cold harness 使用 `Popen.wait(timeout=...)`，POSIX 下的轮询会给退出观测引入最高约 50ms 量化误差。最终版改为测量路径上的阻塞 `wait()`，另用 watchdog 执行超时杀进程组，并以新的 harness/plan hash 全量重跑。错误解释器导致 NPU 子进程失败的一轮也被 fail-closed 拒绝，没有进入结论。

## 已确认的性能问题

1. 公共 wrapper 发生冗余深拷贝和同步。`ap.power/log/exp/equal` 都把 `_core` 返回值再包装为 `ndarray(...)`；该 Python 子类构造调用绑定的 `NPUArray` copy constructor，而 copy constructor 会再次 `aclrtMalloc`、整张 D2D copy，并 `aclrtSynchronizeDevice`。同时 native executor 本身已经在每个算子后同步。Case06 公共四算子链相对私有 core 诊断链慢 `21.51x / 59.15x / 57.88x`，与这条静态路径一致。

2. 细粒度同步阻止异步流水。通用 unary/binary executor 每个算子都执行 `aclrtSynchronizeDevice`，公共包装又同步一次，算子链不能把派发和 kernel 重叠起来。

3. 分配成本偏高。冻结规模下，两次 NPU `full` 分配比 NumPy 参考慢约 `1.86x–2.21x`；当前路径也没有可见的输出/workspace allocator cache。

4. `clip` 标量路径泄漏 ACL scalar handle。源码创建 `amin_scalar` 和 `amax_scalar` 后未调用销毁接口。它主要影响长时间重复调用与稳定性，因此 Case03 没有追加高轮次压力测试。

5. Case02 dtype 漂移。这个问题不是纯性能问题，而是必须先修的正确性问题；NumPilot 已按 fail-closed 规则阻止其冷启动计时和 speedup 发布。

## 边界

- scale adapter 不执行 GLM 生成的 `migrated.py`，所以不能据此声称“大规模迁移脚本端到端快 10 倍”。
- allocation、H2D、compute、D2H 是不同样本分布，不能把各自中位数相加后冒充 E2E。
- 私有 `_core` 只用于定位 wrapper 税，不是支持用户采用的优化 API。
- 每组只有 7 个样本、固定 CPU 后 NPU 顺序、无随机交错和置信区间；1M/9M 的 NPU 高尾尤其明显。
- scale 使用固定有限常量输入，没有覆盖随机、NaN/Inf 和边界值分布。

因此，下一步优化优先级应是：消除公共结果的深拷贝、减少每算子同步、引入分配/workspace 复用，然后再用同一冻结计划重测。迁移框架本身应继续保持当前门禁：小任务不承诺加速、fallback/scoped/数值失败不发布 speedup、大规模结论必须明确是否为 exact artifact E2E。
