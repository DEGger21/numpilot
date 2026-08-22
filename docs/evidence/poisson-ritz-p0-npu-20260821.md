# Poisson Ritz P0：定向补算子、资格化与 strict-NPU 证据

日期：2026-08-21 UTC

> 公开精选证据：[`evidence/poisson-ritz-p0-20260821/`](../../evidence/poisson-ritz-p0-20260821/)；
> 文中的 `/home/ma-user/work/...` 是原始裸金属运行身份，保留用于审计。

## 当前结论

NumPilot 已按比赛框架完成一条不依赖 `ops-nn` 启动的 P0 纵向链路：

```text
扫描完整动态 forward
→ 复用已有且经固定版本审计的 AsNumpy API
→ 只把两个领域接口识别为缺口
→ 用户 LLM + 固定 CANNBot Skill 定向生成本地 Ascend C 工程
→ 模型驱动 repair、不可变封存、六门资格化
→ capability overlay 显式激活
→ 一轮完整动态训练 strict-NPU
→ 250 轮参数驻留设备的确定性缩小案例
→ profiler 定位并消除逐返回值包装深拷贝
→ 重新资格化、同负载 profiler 与 250 轮回归
```

两个缺口是：

- `operator.apply_gradient_descent` → `asnumpy.apply_gradient_descent`；
- `operator.smooth_l1_loss` → `asnumpy.smooth_l1_loss`。

真实生成与 repair 使用 `/home/ma-user/work/.numpilot-config/llm.json` 中的用户模型，并把
固定版本的华为 CANNBot Skill 实际嵌入请求。首轮没有拉取、构建或依赖 `ops-nn`；生成物
保存在本地隔离目录，只有独立资格化通过后才进入显式 overlay。

当前可以证明：两个定向生成算子在限定 profile 内合格；修复后的 `outer`/`matmul` 公开
路径在本例限定的 float32、非空、rank 1/2 范围内调用 ACLNN；一个包含 RBF、Laplacian、
损失、解析梯度和参数更新的完整动态训练轮已通过 strict-NPU；该迁移产物又完成了 250 轮
参数驻留设备的 CPU/NPU 收敛对拍。

首轮 10 轮 CANN profiler 逐项确认 1,170 条设备算子任务，同时暴露了 1,235 次
`aclrtMemcpy`。已知方向探针证明当前 compact profiler 方向字段不具区分力；源码与精确
账目进一步定位到 public wrapper 对每个 private native 结果的深拷贝。显式 one-shot move
修复后，同负载仍有 1,170 条 kernel，复制降为恰好 30 H2D + 5 D2H 的 35 次，并再次通过
六门资格化和 250 轮回归。当前仍不能证明比赛级性能收益，也不能把 8×8 内部点、9 个 RBF
的确定性缩小案例扩大表述为原始 48×48、49 个 RBF、3000 轮配置已经完成性能验收。

## 生成来源与封存

```text
CANNBot commit: abcb35fc0ff77f4bc5eda74de6bbcf28cbc1572b
CANNBot bundle SHA-256: 879f183ad762286251027734ca173398eb01e368499627c9274aaaf75e7e4e4d
SoC: Ascend910B / observed 910B4
CANN: 9.0.0
run root: /home/ma-user/work/numpilot-gap-runs/poisson-ritz-p0-20260821/direct-local-v5
```

运行目录保留 73 份 HTTP 成功且内容哈希绑定的生成/repair 回执；最终两条谱系绑定其中
57 份。权威封存 Artifact 为：

| 算子 | Artifact ID | 最终谱系回执 | 目录 |
|---|---|---:|---|
| `apply_gradient_descent` | `draft-d124f69b25efd0696161` | 20 | `sealed-final/apply_gradient_descent` |
| `smooth_l1_loss` | `draft-d3a03b173efd34240c66` | 37 | `sealed-runtime-final-v2/smooth_l1_loss` |

Artifact 状态仍为 `generated_unverified`；是否可执行由外部 qualification evidence 与
overlay 决定，不能靠回写生成物状态绕过门禁。

## AsNumpy v9 基线与基础能力审计

权威 wheel：

```text
asnumpy-p0-wheel-v9/asnumpy-0.1.dev106+g21368abf9.d20260821-cp310-cp310-linux_aarch64.whl
SHA-256: 260062a29fb76bb738a96eea47ae0ace2dd10c2113becb06f4c81e40e1cf8b84
```

本轮修复了外部 AsNumpy 工作树中 `outer` 和 `matmul` 的 Host fallback/错误 rank 路径：

- `outer(float32, 1-D × 1-D)` 使用 `aclnnGer`；
- `matmul(float32, 2-D × 1-D)` 使用 `aclnnMv`；
- `matmul(float32, 1-D × 2-D)` 使用 `aclnnMatmul`；
- 1-D × 1-D、2-D × 2-D 仍走 ACLNN，并使用 KEEP_DTYPE。

Python wrapper、`linalg/direct.py` 与 `_core` 二进制均由精确 SHA-256 绑定；NumPilot 只在
`dtype=float32, rank>=1, rank<=2, contiguous=true, numel>=1` 的已测范围内放行
`outer`/`matmul`。选定 linalg 测试结果为 `14 passed, 26 deselected`。这不是对 AsNumpy
全部 dtype、rank 或 linalg API 的泛化背书。

基于 profiler 定位的包装层修复继续保留上述限定能力，并新增 private native 临时返回值到
public `ndarray` 的显式 one-shot move；公开数组构造仍保持深拷贝语义。当前权威 wheel 为：

```text
asnumpy-p0-wheel-v10-move-oplogin/
  asnumpy-0.1.dev106+g21368abf9.d20260822-cp310-cp310-linux_aarch64.whl
SHA-256: afe845c45d17e7f5cd619831f555fba3c2344296ffe02f1d251017b04cf0282d
```

其 `__init__.py`、`math.py`、`linalg/direct.py`、`utils.py` 和 `_core` 二进制均由 capability
manifest 精确绑定，选定真机测试为 `19 passed`。可应用的最小生产补丁保存在
[`integrations/asnumpy/`](../../integrations/asnumpy/)，完整外部工作树仍单独保留。

## v5 历史六门资格化

权威证据根：

```text
qualification-evidence-v5
```

可信 harness 从两份封存 Artifact 重新构建，只安装到证据树内的私有 OPP，执行 artifact
测试、public AsNumpy API、真实 NPU 数值 worker，再验证 schema v2 报告并原子发布 overlay。

| 算子 | Qualification ID | 测试 | 数值 case | kernel launch | 最大绝对误差 |
|---|---|---:|---:|---:|---:|
| `apply_gradient_descent` | `qual-1479c0f3e8ad5fadb6f1` | 13 passed | 3 | 3 | `0.0` |
| `smooth_l1_loss` | `qual-b0233d6345dab583d8a0` | 13 passed | 3 | 3 | `5.960464477539063e-08` |

关键哈希：

```text
qualification-summary.json: c152025c7d2d7551479d99e485bd12419a6a076c55812b68597ca15041a17e35
runtime/opp-manifest.json:    1b0f72ce91fc473e6f596dafb4d0722674f07b971922a07872c9c93e3a553788
apply qualification:          18c1ddc1674f734ab7efb4cf86e056d3904375750b50bf20c676851e03043723
smooth qualification:         fdaf5604fb24c033b919148ff294f212e8187f0d2ae2198c791d370c85dc0b08
capability overlay:            08625ef3704db848281cab041e1a7da7d69777b1d425a329c6c43cea45ff1b6e
```

旧 v4 overlay 会因 v9 基础 manifest 变化被 `base manifest drift` 正确拒绝；v9 wheel
只能与 v5 overlay/evidence 成对使用。更早失败或被替代的目录均保留用于审计。

## move wheel 对应的 v6 正式资格化

包装层修复改变了基础运行时二进制身份，因此 NumPilot 没有复用 v5 overlay，而是从封存
Artifact 重新执行全部六门资格化并发布 v6 overlay：

| 算子 | Qualification ID | 测试 | 数值 case | kernel launch | 最大绝对误差 |
|---|---|---:|---:|---:|---:|
| `apply_gradient_descent` | `qual-d5c4c3bd44db3cc4301f` | 13 passed | 3 | 3 | `0.0` |
| `smooth_l1_loss` | `qual-82672ac622245bfa6fe3` | 13 passed | 3 | 3 | `5.960464477539063e-08` |

```text
qualification-summary.json: 3935bf9b2d8d3712eb48aef71eda91ad79fa71f2591827da4ea06e2f7139d68a
runtime/base manifest:        a334e5186b40a4598d5246b3e319fda7811837f32042bfb84ea4018c9a61207c
runtime/opp manifest:         2fec30e42fd0ad5baa35db2d4dab2613dbf39f97926796be551f6a9bfe42fd24
capability overlay:           1c5aa2ae5c4ebf2db8d542f24defd9c7ca42ccc4d87b6e10fafc60c4eb708789
```

## 完整动态一轮 strict-NPU

仓库新增 [`examples/poisson_ritz_iteration.py`](../../examples/poisson_ritz_iteration.py)。与旧
`poisson_ritz_step.py` 先在 CPU 预计算梯度不同，新样例把以下计算全部放入唯一 `forward`：

- 内部点和边界点的 RBF 矩阵；
- 内部点 Laplacian 与边界预测；
- 两次 SmoothL1 损失；
- PDE/边界解析梯度；
- 四组参数更新与宽度裁剪。

分析得到 111 个 NumPy 调用点和 6 个 OperatorSpec helper 调用点。已有接口均复用，缺口
仍只有上述两个领域算子；它们已在本地 overlay 中资格化，因此这次扩展动态样例没有再次
调用 LLM，也没有拉取 `ops-nn`，而是复用了本地封存能力。权威结果：

```text
directory: poisson-full-iteration-strict-npu-v2
status: pass
strategy: whole_forward_npu
total rewrites: 117
strict_npu: true
runtime_profile_ok: true
runtime_target_integrity_ok: true
runtime_trace_ok: true
input/kernel/artifact comparison: all true
max_abs_diff: 9.5367431640625e-07
max_rel_diff: 1.4264183789691742e-07
performance: not measured/published
migrated.py SHA-256: 49779934abf65eececc5dac074d6ab858ef63ad23999052e531f25069248f195
report.json SHA-256: 1d8634f652cfe4815cb10cb15ce1b5abb26343ac3e71cd114a6334c3ceaf4591
```

每轮调用点：

| API | 次数 | API | 次数 |
|---|---:|---|---:|
| `multiply` | 43 | `divide` | 14 |
| `outer` | 14 | `add` | 11 |
| `matmul` | 10 | `subtract` | 7 |
| `negative` | 5 | `apply_gradient_descent` | 4 |
| `clip` | 3 | `exp` | 2 |
| `smooth_l1_loss` | 2 | `sqrt` | 2 |

`poisson-full-iteration-strict-npu-v1` 因学习率张量错误为长度 9，被已资格化 API 以
`alpha must contain one value` 失败关闭；修正为单元素张量后才产生 v2 通过结果。v1 未被
覆盖或删除。

## 250 轮驻留设备证据

[`tools/poisson_iteration_evidence.py`](../../tools/poisson_iteration_evidence.py) 会重新校验
迁移产物等于当前源码的确定性 AST 输出、报告调用点一致、overlay 身份有效，然后：

1. 循环前上传 30 个输入张量；
2. 250 轮内只把设备端返回的 `weight/cx/cy/width` 传给下一轮；
3. 每次调用前执行 profile 与目标身份守卫；
4. 循环后才下载最终 loss 和四组参数；
5. 与相同初值的 CPU 250 轮结果比较。

权威结果：

```text
directory: poisson-multiround-strict-npu-v4-move
status: pass
iterations: 250
actual public API calls: 29,250 (= 117 × 250)
profile rejects: 0
H2D: 30, before loop
D2H: 5, after loop
CPU loss: 9.948076248168945 → 6.602624893188477
NPU final loss: 6.602625370025635
final CPU/NPU max_abs_diff: 4.76837158203125e-07
final CPU/NPU max_rel_diff: 8.621329783409237e-08
strict_npu/runtime profile/target identity: all true
performance: not measured/published
runner SHA-256: 0a60b2202d012aa60d7700cc7c356f2d07f2e14a7b917d74d33fed8ec9627e6d
evidence.json SHA-256: cc4e7de121b9ffb94e7cb2ae052d75f64f5d3fa42ae6926f80e74d699a881139
```

这里的 30/5 是证据 runner 自己执行的显式边界传输次数。修复后的 10 轮 profiler 已独立
确认运行期复制恰好等于这 35 次边界传输，但该结论只覆盖当前 workload 和哈希绑定的 move
wheel，不能泛化成 AsNumpy 所有程序“循环内零复制”。10 轮先导证据保留在
`poisson-multiround-strict-npu-v1`，未绑定 runner 哈希的 250 轮 v2 与 v9 wheel 的 v3
也保留作前序记录，v4-move 才是当前权威版本。

## CANN profiler 与复制闭环

独立记录见 [`poisson-ritz-profiler-p0-20260822.md`](poisson-ritz-profiler-p0-20260822.md)。
10 轮 strict-NPU 采集成功完成：`op_summary` 的 1,170 条记录与 117 × 10 个迁移调用点逐项
一致，其中 `ApplyGradientDescent` 40 次、`NumPilotSmoothMeanLoss` 20 次，证明两个定向
补充算子形成了真实 kernel 任务。

首轮 runtime 数据库记录 1,235 次 `aclrtMemcpy`、总计 1,594,076 bytes。随后用唯一大小的
H2D/D2D/D2H 最小探针确认：三种已知方向的 compact `memcpy_direction` 均为 `0`，不能按
ACL 公共枚举解释。源码和精确调用账目定位到 AsNumpy public wrapper 对 private native
结果的深拷贝；显式 move 修复并重新资格化后，同一 workload 的 kernel 仍为 1,170 条，
复制降为 35 次、13,472 bytes，恰等于 runner 的 30 H2D + 5 D2H。机器审计的 10 项检查
全部通过。两轮采集都只用于正确性、kernel 身份、调用与复制核验；没有发布性能结论。

## 当前边界与下一步

当前可准确表述为：“NumPilot 先扫描接口，只定向补齐两个缺失领域算子；两算子经真实 LLM、
CANNBot Skill、封存和六门资格化后显式激活；确定性缩小 Poisson/RBF 案例已完成完整动态
单轮和 250 轮参数驻留设备的 strict-NPU 数值对拍；同一 10 轮 profiler 已定位并消除
1,200 次包装层深拷贝，修复后 1,170 条 kernel 不变，只剩 35 次边界复制。”

仍待完成：

1. 为更大配置保留 kernel timeline、预热与重复原始样本；
2. 只有数值、fallback、传输与统计门全部通过后才发布性能；
3. 将 Poisson 专用 qualification/evidence harness 泛化为产品级 runner；
4. 发布包、许可证/SBOM、演示视频和普通用户可复现环境；
5. CodeArts 继续按用户决定后置，本阶段不把它当作 LLM 链路前置条件。
