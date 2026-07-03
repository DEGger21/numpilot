# NumPilot 质量属性场景

用标准质量属性场景模板(**刺激源 / 刺激 / 制品 / 环境 / 响应 / 响应度量**)描述 NumPilot 的关键架构质量目标。每个场景对应一处具体架构决策与可验证的落地点。

---

## QA-1 可修改性(Modifiability)—— 更换迁移引擎

| 要素 | 内容 |
|------|------|
| 刺激源 | 开发者 |
| 刺激 | 希望把确定性 AST 改写替换为 LLM 驱动改写(或新增一种引擎) |
| 制品 | 迁移引擎(`numpilot/engines/`) |
| 环境 | 设计期 / 开发期 |
| 响应 | 新增 `MigrationEngine` 子类实现 `rewrite(source, plan)`,在 `_make_engine` 注册;编排器、其它智能体、契约均不改动 |
| 响应度量 | 改动局限于 1 个新文件 + `_make_engine` 一处分支;既有 23 项测试零回归;`--engine llm` 未配置时自动回退 AST,流水线仍 `pass` |

**支撑决策**:[ADR-001 策略模式迁移引擎](adr/ADR-001-strategy-migration-engine.md)。

---

## QA-2 可移植性(Portability)—— shim ↔ NPU 后端切换

| 要素 | 内容 |
|------|------|
| 刺激源 | 部署者 / CI |
| 刺激 | 需在无 NPU 的机器上跑通(CI),又能在昇腾 910B 上真实执行 |
| 制品 | 执行后端(`numpilot/backends/`) |
| 环境 | 运行期 |
| 响应 | 通过 `--backend shim\|npu` 或 `NUMPILOT_BACKEND` 环境变量切换;`ShimBackend` 用 numpy 假模块,`NpuBackend` 导入真实 asnumpy;验证智能体只依赖 `ExecutionBackend` 端口 |
| 响应度量 | 切换后端零代码改动;shim 下离线可复现且数值恒等(`max_rel_diff=0`);验证逻辑对两后端完全一致 |

**支撑决策**:[ADR-002 端口-适配器执行后端](adr/ADR-002-ports-adapters-backend.md)。

---

## QA-3 正确性(Correctness)—— 迁移数值一致性校验

| 要素 | 内容 |
|------|------|
| 刺激源 | 迁移流水线 |
| 刺激 | 产出一份迁移后源码,需确认与原程序数值等价 |
| 制品 | 验证智能体(`numpilot/agents/verification.py`) |
| 环境 | 运行期(迁移完成后) |
| 响应 | 双端执行原始(numpy)与迁移(asnumpy)`forward`,按 dtype 容差(fp32=1e-4 / fp16=1e-2)比对 `max_rel_diff`;不一致 → `status="fail"` |
| 响应度量 | `numerically_consistent = (max_rel_diff <= tol)`;正例实测 `max_rel_diff=0.0 < 1e-4`;判定严格来自数值字段,不靠日志推断 |

**支撑决策**:契约中 `VerificationReport` 显式携带 `max_rel_diff`/`tol`/`status`,判定可审计。

---

## QA-4 可测试性(Testability)—— 契约化组件边界

| 要素 | 内容 |
|------|------|
| 刺激源 | 测试工程师 |
| 刺激 | 需对任一智能体单独写单元测试,不牵动其余组件 |
| 制品 | 契约层(`numpilot/contracts.py`)+ 各智能体 |
| 环境 | 开发期 |
| 响应 | 每个智能体只消费/产出 dataclass 契约;测试构造契约实例即可驱动单个组件;端到端另有 `test_e2e.py` |
| 响应度量 | 8 个测试文件按组件切分(coverage/contracts/analysis/decision/migration_ast/migration_agent/backends/verification)+ e2e;全量 23 passed;单组件测试无需真实 NPU/LLM |

**支撑决策**:[ADR-003 契约先行的智能体](adr/ADR-003-contract-first-agents.md)。

---

## QA-5 健壮性(Robustness)—— 缺口算子早停

| 要素 | 内容 |
|------|------|
| 刺激源 | 用户输入 |
| 刺激 | 待迁移程序使用了不在 AsNumpy 覆盖集的算子(如 `np.fft.fft`) |
| 制品 | 分析智能体 + 编排器 |
| 环境 | 运行期 |
| 响应 | 分析比对覆盖集得 `migratable=False`;编排器立即返回 `status="unsupported"` 并在报告列出缺口算子,不进入迁移/验证 |
| 响应度量 | 负例实测 `unsupported=["fft"]`、退出码 1;不产生任何误迁移代码 |

**支撑决策**:[ADR-004 裁掉缺口算子生成](adr/ADR-004-scope-no-gap-op-gen.md)。
