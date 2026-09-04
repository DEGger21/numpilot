# Gap 恢复状态与证据链

`numpilot.gaps.recovery` schema 2 是 ordinary `GAP_REQUIRED` 后的确定性控制面。一次 run
固定绑定 source、目标 SoC/CANN、runtime identity 和授权 profile；每次状态变化都发布为私有、
write-once 的 JSON hash chain。状态机只验证事实和收据，真正的 resolver、候选物化、模型生成
和后续构建由阶段 producer 执行。

当前产品链已实现：

```text
CREATED → ANALYZED → GAP_PREPARED → RESOLVED
  → CANDIDATE_READY                 # installed official_binary
  → ADAPTER_CANDIDATE               # reviewed numpy.matmul official adapter
```

`INTEGRATED_CANDIDATE → QUALIFIED → ACTIVATED → RESCANNED → FINAL_MIGRATED → VERIFIED`
仍是后续阶段。`generated_unverified` 是 provider 产物状态，不是 Recovery phase；
`ADAPTER_CANDIDATE` 只证明该未验证 overlay 已被 typed receipt 绑定，不代表已经构建、集成、
执行、资格化或激活。

## 固定阶段与收据

```text
CREATED
  → ANALYZED                  analysis_report
  → GAP_PREPARED              gap_spec
  → RESOLVED                  official_resolution_decision
  → CANDIDATE_READY           operator_candidate
  → ADAPTER_CANDIDATE         asnumpy_adapter_candidate
  → INTEGRATED_CANDIDATE      integration_transaction
  → QUALIFIED                 qualification_report
  → ACTIVATED                 activation_receipt
  → RESCANNED                 capability_rescan_report
  → FINAL_MIGRATED            script_migration_report
  → VERIFIED                  script_verification_receipt
```

状态不能跳级。一个 run 包含多个 gap 时，各 gap 可独立推进，全局 phase 始终等于最慢 gap；
`AWAITING_INPUT`、`BLOCKED` 和 `FAILED_ATTEMPT` 只记录原因，不覆盖已有成功状态。

每次 phase `+1` 必须在同一事件中新增 `StageReceiptRef`，严格绑定 phase/kind、当前 recovery
binding SHA-256、紧邻 parent state SHA-256、精确 gap coverage，以及 producer 产物的 locator
和内容 SHA-256。只有新增的 `satisfied` receipt 能推进其覆盖的 gap。

## 冻结授权 profile

CLI 不提供任意布尔 policy 开关，只允许在首次 `ingest-official` 时选择一个命名 profile：

| Profile | 允许的后续副作用 | 典型用途 |
|---|---|---|
| `receipt-only` | 无 | 只导入和审计已有收据；默认值 |
| `candidate-generation` | subprocess、network、已选且获授权的边界模型（configured 或 CodeArts-agent） | official binary candidate；仅 reviewed MatMul 切片可到未验证 `ADAPTER_CANDIDATE`，ops-nn/CANNBot 各停在源码证据/未验证草稿；CodeArts-agent/no-llm 不读取 `llm.json` |
| `full-local-recovery` | 上述全部，再加 system write、AsNumPy import、build、execution、activation | 仅授权未来完整本地恢复链；当前后续 producer 未接线 |

profile 创建后不可升级或改写。授权只表示 producer 可以执行某类动作，不等于动作已经发生；
每个事件仍必须逐项记录真实 side-effect truth。进入 `ACTIVATED` 必须记录本次 activation，进入
`VERIFIED` 必须记录本次 execution、verification 和最终验证收据。
`candidate-generation` 不能在后续事件中执行 system write、AsNumPy import、build、
execution 或 activation，也不能通过续接升级为 `full-local-recovery`。要走完整本地链，
必须从首次 ingest 就创建一个独立、冻结为 `full-local-recovery` 的 run。

## 存储合同

```text
<run-root>/
└── states/
    ├── 000001.json
    ├── 000002.json
    └── ...
```

根目录与 `states/` 必须是当前用户的 `0700` 私有目录，状态文件为 `0600`。加载从 revision 1
开始复验规范 JSON、连续 revision、parent digest、不可变 binding、合法阶段和副作用策略。
未知文件、symlink、宽松权限、缺失 revision、篡改或并发覆盖均失败关闭。

## 1. 导入 official-first 前缀

`recover-gaps ingest-official` 严格消费 schema-2 migration report、其 sibling `gap-request.json`
和 `official-resolution.json`，交叉绑定 source、runtime、SoC/CANN、GapSpec 与每个 decision，
幂等推进到 `RESOLVED`。这是离线 receipt ingest：不重跑 resolver、不联网、不调用模型。

```bash
python -m numpilot recover-gaps ingest-official /absolute/path/to/source.py \
  --migration-report /absolute/private/run/script-migration-report.json \
  --gap-request /absolute/private/run/gap-request.json \
  --official-resolution-report /absolute/private/run/official-resolution.json \
  --state-dir /absolute/private/run/recovery \
  --authorization-profile candidate-generation \
  --json
```

默认 `receipt-only` 不能事后升级；需要继续物化或调用模型时，必须在创建 run 时选择
`candidate-generation` 或 `full-local-recovery`。

## 2. 物化 installed official binary

`materialize-candidates` 当前只接受 `official_binary`。它 fresh 重跑 installed-CANN resolver，
要求 `resolution_id` 精确一致，并重新哈希 selected header/library；随后发布 route-neutral
`OperatorCandidateArtifact`、五类 adapter refs、外部 installed-CANN terms 收据和逐 gap
`operator_candidate` receipt：

```bash
python -m numpilot recover-gaps materialize-candidates \
  --state-dir /absolute/private/run/recovery \
  --out /absolute/private/run/candidates \
  --json
```

产物目录为 `0700`、文件为 `0600`、write-once。该 producer 如实记录
`subprocess_performed=true`，model/network/build/import/execution/activation 均为 false。
installed toolkit 使用外部 CANN terms，收据固定 `accepted=false`；不得借用 ops-nn 的 CANN OSL
冒充用户已经接受 installed binary 条款。`official_source` 和 `remote_llm` materializer 目前失败
关闭，尚不能推进 Recovery 的 `CANDIDATE_READY`。

## 3. 生成并导入 reviewed MatMul adapter

当前只审计 `numpy.matmul` 的 `official_aclnn_existing_module` 路线。模型门初始关闭；只有
candidate、GapSpec、official decision、mapping/header/library/symbol/semantic-review 五类 refs
和目标 AsNumPy context plan 全部复验后才打开。official context plan 使用 schema 2
`prompt_dependencies`，每个目标只允许 `reference`、`signed_official_header` 或
`generated_overlay` 依赖。MatMul 的固定映射是：

```text
product.cpp     ← reference/product.hpp + signed_official_header/aclnn_matmul.h
direct.py       ← reference/bind_linalg.cpp + reference/utils.py + reference/bind_utils.cpp
test_product.py ← generated_overlay/direct.py
```

schema-2 mapping 和 context hashes 共同进入 `plan_sha256`，并作为现有 schema-4 report
内的 `asnumpy_adapter_context_plan` 保存，不增加 report schema 或顶层字段。每个目标请求
只投影自己的 hash-bound 依赖，并嵌入 content-free manifest；manifest 绑定依赖
SHA-256、provenance 和 `projection_sha256`。`generated_overlay` 还必须记录
rebuilt/base/response hashes 及 producer receipt 的 request/response hashes。system + user messages
包络上限为 64 KiB，实际请求体由 remote receipt 的 `request_body_sha256` 绑定。

模型使用用户配置的 endpoint 和 NumPilot 内置 AsNumPy integration Skill，但不返回完整
目标文件；它返回 `NUMPILOT_EXACT_EDIT_V1` OLD→NEW 语义编辑。本地依次校验目标路径、
base SHA-256、1..16 个编辑、每个 OLD 在 base 中恰好出现一次、区间不重叠，再机械
重建精确三路径的 schema-4 `generated_unverified` overlay：

- `product.cpp` 的 target edit window 只包含 Matmul 函数；
- `test_product.py` 的 target edit windows 只包含 imports/helpers 和 matmul 测试段；
- 每个 OLD 必须完整落在单一允许窗口内，不得跨窗口匹配。

GLM exact route 使用严格 SSE 事件聚合，并在请求中关闭 thinking、将 reasoning
设为 minimal。只有 SSE 聚合后 `content` 为空才会从 8192 响应预算升级到
16384；非空响应的协议或编辑违约不会触发该升级。真实端点验收已产生 5 份
成功 remote receipts，对应 Recovery 状态是 `ADAPTER_CANDIDATE`；其 side-effect truth 仍表明
build/execution/activation 均未发生。

```bash
python -m numpilot generate-asnumpy-adapter \
  --candidate /absolute/private/run/candidates/gaps/<gap-id>/operator-candidate.json \
  --recovery-state-dir /absolute/private/run/recovery \
  --asnumpy-reference /absolute/path/to/pinned/asnumpy \
  --out /absolute/private/run/adapter
```

生成成功后 CLI 自动调用 receipt ingest；若发布后因中断或状态竞争未能追加，可安全重放：

```bash
python -m numpilot recover-gaps ingest-adapter \
  --state-dir /absolute/private/run/recovery \
  --candidate /absolute/private/run/candidates/gaps/<gap-id>/operator-candidate.json \
  --adapter-output /absolute/private/run/adapter \
  --json
```

`ingest-adapter` 自身不联网、不调用模型。它复验 provider schema 4、完整 overlay tree、模型回执、
official candidate、本地精确重建后的完整 adapter 文件与 reviewed context plan，然后把
producer 已发生的
`network=true/model=true` 如实带入状态事件；build/integration/import/execution/verification/
system-write/subprocess/activation 仍为 false。

## 路由中立候选与当前边界

`OperatorCandidateArtifact` 定义 `official_binary`、`official_source`、`remote_llm` 三种来源；
来源必须与 official-resolution route 精确对应，所有 candidate 固定
`activation_allowed=false`。当前自动接线只覆盖 installed `official_binary` 与 reviewed MatMul
adapter。仍待实现：

1. `official_source` 和 `remote_llm` 的 Recovery candidate materializer；
2. 其他 API/route 的 reviewed AsNumPy adapter planner 与 receipt adapter；
3. 隔离 integration transaction、构建、六门 qualification、激活与回滚；
4. overlay-aware capability rescan、按已选 lane 的最终 migration 和 final verification（CodeArts-agent/no-llm 不调用 configured model）；
5. ordinary strict-NPU provenance 与独立性能证据。

当前 `candidate-generation` 的自动 producer 只到 `ADAPTER_CANDIDATE`。下列 integration/
qualification/activation/rescan/final migration 任务只能由初始就冻结为
`full-local-recovery` 的 run 授权，但对应 producer 尚未自动接线。

因此，任何 `generated_unverified` 或 `ADAPTER_CANDIDATE` 都不能写成“本地 AsNumPy 已具备该
接口”。只有后续 `INTEGRATED_CANDIDATE`、`QUALIFIED`、`ACTIVATED` 和 `RESCANNED` 均有独立
收据后，能力目录才允许改变。
