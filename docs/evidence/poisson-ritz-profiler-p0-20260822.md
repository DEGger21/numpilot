# Poisson Ritz P0：首轮 CANN profiler 记录

日期：2026-08-22 UTC

## 结论快照

首轮 `msprof` 采集和文本导出已成功完成。采集对象不是性能发布配置，而是 10 轮
确定性缩小 Poisson/RBF strict-NPU 正确性运行；对应 evidence 状态为 `pass`，10 轮共
执行 1,170 次公开 NPU API 调用。导出的 1,170 条算子记录与 NumPilot 预期的 API 种类和
次数逐项一致，两个定向补充算子均出现了真实的自定义 kernel：

- `ApplyGradientDescent`：40 次；
- `NumPilotSmoothMeanLoss`：20 次。

本轮还发现 1,235 次 `aclrtMemcpy`。NumPilot runner 记录的 30 次循环前显式 H2D 和 5 次
循环后显式 D2H，描述的只是 runner 自己发起的边界传输，不能再扩大表述成运行期总共只有
35 次复制。CANN 原始 `MemcpyInfo.memcpy_direction` 在本次采集中全部为数值 `0`；在没有
把这个采集字段的编码与具体 H2D/D2H/D2D 语义可靠闭环前，本记录不猜测其方向。

因此当前可以证明“117 个调用点每轮均形成设备 kernel，两个补充算子真实执行，10 轮数值
对拍通过”；仍不能据此发布端到端性能、零隐式复制或设备驻留零复制结论。

## 证据位置与采集身份

运行根：

```text
/home/ma-user/work/numpilot-gap-runs/poisson-ritz-p0-20260821/direct-local-v5/poisson-profiler-strict-npu-v1
```

Profiler 工程：

```text
PROF_000001_20260822000031293_00323450OOPHMLKP
```

采集配置由工程内 `device_0/sample.json` 固化，关键项为：

```text
runtimeApi: on
taskTime: l1
ai_core_profiling: on
ai_core_profiling_mode: task-based
aiv_metrics: PipeUtilization
runtimeTrace: on
ts_memcpy: on
device: 0
```

`msprof` 采集进程退出码为 `0`，采集结束并完成 text export。第一次尝试因输出目录不存在
而以 `255` 退出，目标程序没有执行；保留的上述工程来自创建明确输出目录后的成功重跑。

## 运行正确性

Profiler 包裹的目标是
[`tools/poisson_iteration_evidence.py`](../../tools/poisson_iteration_evidence.py) 的 10 轮运行。
其独立证据位于 `evidence-run/evidence.json`：

| 项目 | 结果 |
|---|---:|
| 状态 | `pass` |
| 迭代 | 10 |
| 每轮公开 API 调用 | 117 |
| 累计公开 API 调用 | 1,170 |
| CPU loss | `9.948076248168945 → 9.38313102722168` |
| NPU 最终 loss | `9.38313102722168` |
| 最大绝对误差 | `2.2351741790771484e-08` |
| 最大相对误差 | `1.844382381932519e-06` |
| strict-NPU | `true` |
| runtime profile | `true` |
| runtime target identity | `true` |

固定身份：

```text
runner SHA-256:   0a60b2202d012aa60d7700cc7c356f2d07f2e14a7b917d74d33fed8ec9627e6d
source SHA-256:   1deeeec0e5c32c96c4a33467687ce5181a6c8d2c018c6b81b3dc066e3e8beb1a
migrated SHA-256: 49779934abf65eececc5dac074d6ab858ef63ad23999052e531f25069248f195
overlay SHA-256:  08625ef3704db848281cab041e1a7da7d69777b1d425a329c6c43cea45ff1b6e
evidence SHA-256: f8b286799feebf1c4540871064574a93fe28b6cd1b518b620e1440b6c0601256
```

## 导出算子核对

`op_summary_20260822000043.csv` 恰有 1,170 条数据行；`op_statistic` 汇总如下。次数与
NumPilot 迁移报告的 10 轮累计调用完全一致：

| 导出 OP Type | NumPilot API | Core | 次数 | 总 task time（μs） |
|---|---|---|---:|---:|
| `Mul` | `multiply` | AI_VECTOR_CORE | 430 | 737.858 |
| `Ger` | `outer` | AI_VECTOR_CORE | 140 | 453.788 |
| `MatMulV2` | `matmul` | AI_CORE | 100 | 311.806 |
| `RealDiv` | `divide` | AI_VECTOR_CORE | 140 | 245.586 |
| `Add` | `add` | AI_VECTOR_CORE | 110 | 203.085 |
| `Sub` | `subtract` | AI_VECTOR_CORE | 70 | 138.343 |
| `ApplyGradientDescent` | `apply_gradient_descent` | AI_VECTOR_CORE | 40 | 106.822 |
| `Neg` | `negative` | AI_VECTOR_CORE | 50 | 73.422 |
| `ClipByValueV2` | `clip` | AI_VECTOR_CORE | 30 | 57.801 |
| `NumPilotSmoothMeanLoss` | `smooth_l1_loss` | AI_VECTOR_CORE | 20 | 56.861 |
| `Exp` | `exp` | AI_VECTOR_CORE | 20 | 32.941 |
| `Sqrt` | `sqrt` | AI_VECTOR_CORE | 20 | 29.921 |
| **合计** | | | **1,170** | **2,448.234** |

`task_time` 另行确认其中 100 条是 AI_CORE、1,070 条是 AI_VECTOR_CORE；文件还含
`PROFILING_ENABLE`、`PROFILING_DISABLE` 和 `PLACE_HOLDER_SQE` 各一条，它们不计入上述
1,170 条算子任务。

两个补充算子的导出身份为：

```text
op_summary:
  aclnnApplyGradientDescent_ApplyGradientDescentAiCore_ApplyGradientDescent   × 40
  aclnnSmoothL1Loss_SmoothL1LossAiCore_NumPilotSmoothMeanLoss                × 20

task_time:
  ApplyGradientDescent_bb1e227ed5914ba606d8645f844eaa3c_0                    × 40
  NumPilotSmoothMeanLoss_0b8616694c9ed41d02d1e849facf93f1_0                 × 20
```

ACL API 汇总还记录：

| API | 次数 | API 总时间（μs） |
|---|---:|---:|
| `aclnnApplyGradientDescent` | 40 | 1,456.50 |
| `aclnnApplyGradientDescentGetWorkspaceSize` | 40 | 1,207.28 |
| `aclnnSmoothL1Loss` | 20 | 919.20 |
| `aclnnSmoothL1LossGetWorkspaceSize` | 20 | 1,402.29 |
| `aclrtMemcpy` | 1,235 | 40,469.21 |

这里的 ACL API 时间与 kernel task time 属于不同统计层级，不能相加当作端到端时间。

## Memcpy 事实与未决边界

原始 `host/sqlite/runtime.db` 的 `MemcpyInfo` 表记录：

```text
rows: 1,235
sum(data_size): 1,594,076 bytes
min(data_size): 4 bytes
max(data_size): 2,304 bytes
distinct memcpy_direction: {0}
```

按大小分组：

| bytes/copy | 次数 |
|---:|---:|
| 4 | 49 |
| 36 | 278 |
| 128 | 44 |
| 256 | 54 |
| 1,152 | 262 |
| 2,304 | 548 |

ACL 头文件中的 `aclrtMemcpyKind` 确实按 HostToHost、HostToDevice、DeviceToHost、
DeviceToDevice 顺序定义；但 profiler 的 compact `MemcpyInfo` 解析器只是原样保存采集字段，
没有在本次导出中给出方向文字。不能仅凭两个不同接口都出现数值 `0` 就认定语义相同。
下一步需要用一个已知单次 H2D/D2H/D2D 的最小探针，或找到对应采集协议的权威枚举映射，
再对 1,235 条复制分类。

无论最终方向如何，1,235 已显著大于 runner 的 35 次显式边界传输，说明包装层/ACLNN
调用链中的复制必须被继续审计。它是当前性能发布前的高优先级问题。

## 原始文件哈希

```text
api_statistic CSV: e4e729a29224f72a1399939b335859ed7153297e49e4247544ddd7bad6da1d3b
op_statistic CSV:  9afdaff8baf564b03fa4cd080f4be8b149867f9cf4082a7597fe8c82439d5991
op_summary CSV:    9368b3956e3038b6bf213e3c96539e05552917e477b6dc5047fedd1424a54877
task_time CSV:     b41751212aa6a5d1346aaaa484645d6f88a82ba57bfee75fb043437bfcb4e768
runtime.db:        fe2e13e11a6a206f8c8b3d61d554fe5bb8ed544655f8ed7b0f690f78369dd006
```

## 工具安全告警

`msprof` export 虽然完成，但控制台报告系统安装文件
`.../analysis/msconfig/msprof_slice.json` 可被其他用户写入。只读核验显示该文件和父目录当前
均为 `root:root`、mode `777`。本轮没有修改系统 CANN 权限或文件。这个环境问题必须记录，
后续应由环境管理员按 CANN 安装要求修复，不能把告警静默忽略。

## 后续门槛

1. 完成已知方向的 memcpy 最小探针，解释 1,235 条复制的方向和来源；
2. 审计 AsNumpy `ndarray` 包装/构造是否在每个返回值上产生 D2D 深拷贝；
3. 若确认，消除不必要复制后重新采集同一 10 轮基线；
4. 再扩到合理规模、预热和多次重复样本；
5. 数值、fallback、传输和统计门都通过后，才形成性能结论。
