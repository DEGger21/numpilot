# Poisson Ritz P0：CANN profiler 与包装层复制闭环

日期：2026-08-22 UTC

> 公开精选 profiler 证据：
> [`evidence/poisson-ritz-p0-20260821/profiler/`](../../evidence/poisson-ritz-p0-20260821/profiler/)；
> 文中的绝对路径是原始裸金属运行身份。

> 修复后公开证据：
> [`evidence/poisson-ritz-p0-20260821/profiler-copy-elision/`](../../evidence/poisson-ritz-p0-20260821/profiler-copy-elision/)；
> `profiler/` 保留修复前基线，不覆盖历史证据。

## 更新结论：复制问题已闭环

已知方向探针、AsNumpy 源码定位、one-shot move 修复、重新资格化和同一 10 轮 profiler
重采均已完成。修复前后使用相同迁移源码、117 次/轮公开 API 调用和数值门：

| 项目 | 修复前 | 修复后 |
|---|---:|---:|
| 迭代 | 10 | 10 |
| 公开 API / kernel tasks | 1,170 / 1,170 | 1,170 / 1,170 |
| `aclrtMemcpy` | 1,235 | 35 |
| 复制字节 | 1,594,076 | 13,472 |
| CPU/NPU 最大绝对误差 | `2.2351741790771484e-08` | `2.2351741790771484e-08` |
| strict-NPU / profile / target identity | 全部通过 | 全部通过 |

修复后的 35 次复制恰等于 runner 循环前 30 次 H2D 与循环后 5 次 D2H。被消除的 1,200 次
由每次 1,170 个 native 算子结果包装时的深拷贝，以及 30 个上传结果由 private native
对象包装为 public `ndarray` 时的深拷贝组成；总计减少 1,580,604 bytes。

最小探针分别执行了大小唯一的 4,093-byte H2D、2,053-byte D2D 和 1,021-byte D2H，回传
字节逐项一致，但 compact `MemcpyInfo.memcpy_direction` 三条记录仍全部为 `0`。因此该字段
在当前 CANN 9.0 采集格式中不具方向区分力；本次闭环依赖精确大小、调用账目和修复前后
差分，不把 profiler 的 `0` 强行映射成 ACL 公共枚举。

修复保持 public `ndarray(existing_public_array)` 的深拷贝语义，只允许 Python 包装层以
必需且 keyword-only 的 `_move=True` 消费 private `_core.ndarray` 临时结果。新 wheel 又通过
19 个选定真机测试、v6 六门资格化、10 轮重采和 250 轮 29,250 次调用回归。

关键身份：

```text
AsNumpy wheel SHA-256: afe845c45d17e7f5cd619831f555fba3c2344296ffe02f1d251017b04cf0282d
capability overlay SHA-256: 1c5aa2ae5c4ebf2db8d542f24defd9c7ca42ccc4d87b6e10fafc60c4eb708789
10-round evidence SHA-256: 77165cd7535f2a7cdfae39eea88d1c0989483d6b42ca3ccf96117275516d4a8c
after runtime.db SHA-256: 6283c653c63d4ca5726515330a736a0e9cf8dfe895f710f4c3ca81b8a594ed65
after rts_track.db SHA-256: 8319903565f7ebe8470b5cbfeada73f5c5a5f1c0af82f85bd790b2c47de0d077
```

机器审计的 10 项账目、kernel、数值和身份检查全部为 `true`。这是一份复制消除正确性
证据，不包含预热或重复耗时样本，`performance.published` 明确为 `false`。

## 首轮基线结论快照（修复前）

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

因此这份修复前基线可以证明“117 个调用点每轮均形成设备 kernel，两个补充算子真实执行，
10 轮数值对拍通过”；它本身不能证明端到端性能、零隐式复制或设备驻留零复制。本文开头
的修复后证据完成了复制闭环，但仍不发布性能结论。

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

## 修复前 Memcpy 事实与当时的未决边界

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
后续已用已知单次 H2D/D2D/D2H 探针确认三者仍全部记录为 `0`，因此该字段不能承担方向
分类；详见本文开头的更新结论。

无论最终方向如何，1,235 已显著大于 runner 的 35 次显式边界传输，说明包装层/ACLNN
调用链中存在额外复制。后续精确定位与重采已完成，这一高优先级问题现已闭环。

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

1. ~~完成已知方向的 memcpy 最小探针，解释采集字段边界~~（已完成）；
2. ~~审计 AsNumpy `ndarray` 包装/构造的逐返回值深拷贝~~（已完成）；
3. ~~消除不必要复制后重新采集同一 10 轮基线~~（已完成，1,235 → 35）；
4. 再扩到合理规模、预热和多次重复样本；
5. 数值、fallback、传输和统计门都通过后，才形成性能结论。
