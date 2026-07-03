# ADR-002:端口-适配器实现可移植执行后端

- 状态:已接受(Accepted)
- 日期:2026-07-02
- 相关:[ADR-001](ADR-001-strategy-migration-engine.md)、[quality-scenarios.md](../quality-scenarios.md) QA-2

## 背景(Context)

验证阶段需要**实际执行**迁移后的 `ap.<op>` 代码来做数值对比与测速。但真实执行依赖 `asnumpy` + CANN + 昇腾 NPU 硬件,而课程评审、开发机与 CI 通常无 NPU。若验证智能体直接依赖真实 asnumpy,则离线环境完全无法跑通、无法测试。

## 决策(Decision)

定义执行后端**端口** `ExecutionBackend`(抽象方法 `get_module()` / `to_device()` / `to_host()`),提供两个**适配器**:

- `ShimBackend`(`name="shim"`,**默认**):返回一个 numpy 支撑的假 `asnumpy` 模块(`ap.add` 等转发 numpy,`NPUArray.from_numpy/.to_numpy` 包装)。处处可跑、数值恒等。
- `NpuBackend`(`name="npu"`):导入真实 `asnumpy`,在 NPU 上执行。

验证智能体只依赖端口;后端经 `select_backend(name)`(读 `--backend` 或 `NUMPILOT_BACKEND`)选择。迁移侧通过临时挂载 `sys.modules["asnumpy"] = backend.get_module()` 满足迁移代码的 `import asnumpy as ap`。

## 后果(Consequences)

**正向**
- **可移植**:同一验证逻辑在 shim 与 NPU 上零改动运行(见 QA-2)。
- **可测试 / 可复现**:默认 shim 让全套测试离线跑通,数值恒等(`max_rel_diff=0`),CI 友好。
- 新增后端只需实现端口三方法并在 `select_backend` 注册。

**负向 / 权衡**
- shim 下测得的 `speedup≈1`,不代表真实 NPU 加速——报告需显式标注"当前后端"。
- 假模块需与真实 asnumpy 的接口保持同步,存在漂移风险。

## 备选方案(Alternatives)

- **直接依赖真实 asnumpy**:离线/CI 完全不可跑,违背零依赖目标。
- **Mock 打桩(测试内替换)**:仅解决单测,无法支撑真实 CLI 端到端运行与后端切换。
