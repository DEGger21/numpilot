# ADR-002:端口-适配器实现可移植执行后端

- 状态:已接受，2026-07-18 修订
- 日期:2026-07-02
- 相关:[ADR-001](ADR-001-strategy-migration-engine.md)、[quality-scenarios.md](../quality-scenarios.md) QA-2

## 背景(Context)

验证阶段需要**实际执行**迁移后的 `ap.<op>` 代码来做数值对比与测速。但真实执行依赖 `asnumpy` + CANN + 昇腾 NPU 硬件,而课程评审、开发机与 CI 通常无 NPU。若验证智能体直接依赖真实 asnumpy,则离线环境完全无法跑通、无法测试。

## 决策(Decision)

定义执行后端**端口** `ExecutionBackend`(抽象方法 `get_module()` / `to_device()` / `to_host()`),提供两个**适配器**:

- `ShimBackend`(`name="shim"`,**默认**):返回一个 numpy 支撑的假 `asnumpy` 模块(`ap.add` 等转发 numpy,`NPUArray.from_numpy/.to_numpy` 包装)。处处可跑、数值恒等。
- `NpuBackend`(`name="npu"`):导入真实 `asnumpy`,在 NPU 上执行。

验证智能体只依赖端口；后端经 `select_backend(name)` 选择。真实 NPU backend 还必须提供版本化 `CapabilityManifest`，逐 API 区分 strict NPU、host-assisted、CPU fallback 和 unknown。迁移侧通过受全局锁保护、可恢复的临时模块注入满足 `import asnumpy as ap`；正式系统仍需升级为 OS 级隔离子进程。

## 后果(Consequences)

**正向**
- **可移植**:同一验证逻辑在 shim 与 NPU 上零改动运行(见 QA-2)。
- **可测试 / 可复现**:shim 让全套语义与编排测试离线跑通；它只证明迁移语义，不提供 NPU 性能证据。
- 新增后端只需实现端口三方法并在 `select_backend` 注册。

**负向 / 权衡**
- shim 永不发布 speedup；真实 NPU 也只有 capability、三层语义和基线门禁全通过才可发布。
- 假模块需与真实 asnumpy 的接口保持同步,存在漂移风险。
- 进程内 `sys.modules` 注入即使有锁也不是安全沙箱，不能并行执行不可信用户代码。

## 备选方案(Alternatives)

- **直接依赖真实 asnumpy**:离线/CI 完全不可跑,违背零依赖目标。
- **Mock 打桩(测试内替换)**:仅解决单测,无法支撑真实 CLI 端到端运行与后端切换。
