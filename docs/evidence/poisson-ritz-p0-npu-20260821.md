# Poisson Ritz P0：定向补算子、资格化与 strict-NPU 证据

日期：2026-08-21 UTC

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

首轮 10 轮 CANN profiler 已完成，并逐项确认 1,170 条设备算子任务；它同时暴露了
1,235 次尚待完成方向归类和来源审计的 `aclrtMemcpy`。当前仍不能证明比赛级性能收益，
也不能把 8×8 内部点、9 个 RBF 的确定性缩小案例扩大表述为原始 48×48、49 个 RBF、
3000 轮配置已经完成性能验收。

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

## AsNumpy v9 与基础能力审计

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

## v5 正式六门资格化

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

旧 v4 overlay 会因 v9 基础 manifest 变化被 `base manifest drift` 正确拒绝；当前只能将
v9 wheel 与 v5 overlay/evidence 成对使用。更早失败或被替代的目录均保留用于审计。

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
directory: poisson-multiround-strict-npu-v3
status: pass
iterations: 250
actual public API calls: 29,250 (= 117 × 250)
profile rejects: 0
H2D: 30, before loop
D2H: 5, after loop
CPU loss: 9.948076248168945 → 6.602624893188477
NPU final loss: 6.602625370025635
final CPU/NPU max_abs_diff: 4.76837158203125e-07
final CPU/NPU max_rel_diff: 1.9883130164544127e-07
strict_npu/runtime profile/target identity: all true
performance: not measured/published
runner SHA-256: 0a60b2202d012aa60d7700cc7c356f2d07f2e14a7b917d74d33fed8ec9627e6d
evidence.json SHA-256: 9f726a7a293f2822b68f029ac6d5e39f58eff7ff7b543cad8f9d391a51a399fd
```

这里的 30/5 是证据 runner 自己执行的显式边界传输次数，不代表 ACL/包装层内的全部
复制。首轮 profiler 已记录 1,235 次 `aclrtMemcpy`，方向编码和来源仍待专项审计；因此
不能把这一节表述成“循环内零复制”。其间所用公开 API 还受 v9 的 source/binary 哈希审计
约束。10 轮先导证据保留在 `poisson-multiround-strict-npu-v1`，未绑定 runner 哈希的
250 轮 v2 也保留作前序记录，v3 才是当前权威版本。

## 首轮 CANN profiler

独立记录见 [`poisson-ritz-profiler-p0-20260822.md`](poisson-ritz-profiler-p0-20260822.md)。
10 轮 strict-NPU 采集成功完成：`op_summary` 的 1,170 条记录与 117 × 10 个迁移调用点逐项
一致，其中 `ApplyGradientDescent` 40 次、`NumPilotSmoothMeanLoss` 20 次，证明两个定向
补充算子形成了真实 kernel 任务。

同时，runtime 数据库记录 1,235 次 `aclrtMemcpy`、总计 1,594,076 bytes。原始
`memcpy_direction` 字段本次全部为 `0`，但当前没有足够证据把 profiler 采集编码直接解释为
ACL 公共枚举方向，因此尚不发布 H2D/D2H/D2D 分类。此次采集只用于正确性、kernel 身份和
调用次数核验；没有发布性能结论。

## 当前边界与下一步

当前可准确表述为：“NumPilot 先扫描接口，只定向补齐两个缺失领域算子；两算子经真实 LLM、
CANNBot Skill、封存和六门资格化后显式激活；确定性缩小 Poisson/RBF 案例已完成完整动态
单轮和 250 轮参数驻留设备的 strict-NPU 数值对拍。”

仍待完成：

1. 用已知方向最小探针解释首轮 profiler 的 1,235 次 `aclrtMemcpy`，并定位包装层复制；
2. 消除不必要复制后重采 10 轮基线，再为更大配置保留 kernel timeline 与原始样本；
3. 只有数值、fallback、传输与重复样本门全部通过后才发布性能；
4. 将 Poisson 专用 qualification/evidence harness 泛化为产品级 runner；
5. 发布包、许可证/SBOM、演示视频和普通用户可复现环境；
6. CodeArts 继续按用户决定后置，本阶段不把它当作 LLM 链路前置条件。
