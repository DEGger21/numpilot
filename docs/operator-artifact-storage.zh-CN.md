# Benchmark 算子产物存储契约

NumPilot 的 150 个 benchmark 样例共享一个 **benchmark run 根目录**。中间生成的算子不能散落在每个样例目录，也不能写入安装好的 AsNumpy；它们统一写入该 run 的 `operator-artifacts/`，由 `OperatorArtifactStore` 管理。样例只保存引用，因此同一个语义算子可以被多个样例复用。现有 GapCompletion 的 `draft-*`、official candidate 和 qualification ID 保持不变；本层只建立索引和跨样例引用，不替换上游 ID。

## 目录

```text
<run-root>/operator-artifacts/
├── store.json
├── .store.lock
├── operators/<operator_id>/
│   ├── operator.json
│   ├── candidates/<candidate_id>/
│   │   ├── candidate.cpp
│   │   └── candidate.json
│   ├── qualifications/<qualification_id>/qualification.json
│   └── activations/<activation_id>/activation.json
└── samples/<sample_id>-<operator_id>.json
```

`operator_id` 由算子名、语义签名、Gap 类型、目标 SoC、CANN 版本和 runtime identity 哈希共同计算。它不包含 benchmark 样例名，所以 150 个样例遇到同一个语义缺口时可以复用候选和资格化结果。`candidate_id`、`qualification_id`、`activation_id` 都是内容寻址哈希目录，文件使用原子写入，已存在的不同内容不能覆盖旧产物。

## 生命周期

1. `candidate`：CANNBot/模型产生的候选。状态固定为 `generated_unverified`，必须记录源代码哈希，且 `compiled=false`、`activated=false`。
2. `qualification`：绑定 candidate 后记录构建、公开 API、真实 NPU 执行和 NumPy 数值验证回执。只有 `status=qualified` 且 `interface_available=true` 才能进入下一步。
3. `activation`：绑定 qualification 的 run-scoped 激活回执。`activated=true` 时必须同时证明 `upstream_modified=false`；它只影响当前运行上下文，不改写上游 AsNumpy。
4. `sample binding`：每个 benchmark 样例写一条轻量引用，绑定 operator、candidate、qualification 和 activation。最终 150 例 summary 通过这些引用回放完整证据链。

存储层拒绝 token、endpoint、raw model response、prompt 等敏感字段。模型原始响应不进入 benchmark 产物；报告只保存脱敏身份、哈希和 artifact locator。复制整个 run 根目录即可离线重放，不需要再次调用模型。

实现：`numpilot/operator_artifacts.py`，测试：`tests/test_operator_artifacts.py`。

## 150 例复用与清理

`operator_id` 同时绑定 AsNumpy baseline、API catalog、CANNBot bundle 和 runtime identity 哈希；任一环境哈希变化都会产生新的算子键，旧候选不会被误复用。benchmark 每个样例的 `operator-ref.json` 只引用 `operator-artifacts/samples/` 中的绑定；direct、rejected、unsupported 样例写 `not_applicable`/`rejected`/`unsupported`，不伪造候选。

保留策略采用 mark-and-sweep：仍被 150 例 summary、qualification overlay 或用户 pin 引用的对象永不删除；未引用的 failed/blocked candidate 默认 7 天、未引用 qualification/build 默认 30 天。GC 首先只生成 dry-run 清单，二次确认引用关系后才允许删除，并写入独立 GC receipt；benchmark receipt 永不删除。并行写入使用 store lock，run root 和 artifact 子目录要求 0700，拒绝符号链接。


## 官方范围限定的样例资格

benchmark可在global runtime为audit_required时存储单个样例的正式scope资格。
store要求完整source/generated/profile/runtime/overlay/guard/数值证据并独立重验，
不能只传strict=true或scope标签。性能另验，不能随数值或scope资格自动升格。
原始候选/官方qualification包仍独立归档；样例回执只引用经过校验的链。
复现与跨机恢复边界见[范围限定benchmark](scoped-benchmark.zh-CN.md)。
