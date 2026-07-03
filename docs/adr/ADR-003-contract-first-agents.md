# ADR-003:契约先行的智能体边界

- 状态:已接受(Accepted)
- 日期:2026-07-02
- 相关:[ADR-001](ADR-001-strategy-migration-engine.md)、[ADR-002](ADR-002-ports-adapters-backend.md)、[quality-scenarios.md](../quality-scenarios.md) QA-4

## 背景(Context)

NumPilot 是多智能体系统(分析 / 决策 / 迁移 / 验证 / 编排)。若智能体之间以松散的 dict 或位置参数互传数据,边界会随实现漂移,难以独立测试与替换,组件耦合度上升。课程要求清晰、可讲清的组件边界。

## 决策(Decision)

先定义**契约层** `numpilot/contracts.py`(dataclass),再实现智能体。每个智能体只消费/产出契约类型:

| 契约 | 生产者 → 消费者 |
|------|----------------|
| `AnalysisResult`(含 `op_uses` / `unsupported` / `migratable`) | Analysis → Decision / Orchestrator |
| `OffloadPlan`(`placement` / `threshold_elements`) | Decision → Migration |
| `MigrationResult`(`engine` / `migrated_source`) | Migration → Verification |
| `VerificationReport`(`numerically_consistent` / `max_rel_diff` / `tol` / `speedup` / `status`) | Verification → Orchestrator |

"组件边界 = 契约边界":智能体内部实现可自由替换,只要契约不变。

## 后果(Consequences)

**正向**
- **可测试**:测试构造契约实例即可驱动单个智能体,无需拉起整条流水线或真实 NPU/LLM(见 QA-4,8 个组件测试文件 + e2e)。
- **可替换**:配合 ADR-001/002 的策略与端口,替换实现不破坏边界。
- 数据形状显式、可审计;`VerificationReport` 直接携带判定所需数值字段。

**负向 / 权衡**
- 契约演进需同步所有生产者/消费者;字段增删是跨组件改动。
- 早期设计成本略高于"随手传 dict"。

## 备选方案(Alternatives)

- **裸 dict / 元组传参**:零前期成本,但边界隐式、易漂移、难单测——正是要规避的。
- **完整 IDL / protobuf**:对单进程 Python demo 过重,dataclass 已足够表达契约。
