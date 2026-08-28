# NumPilot 文档导航

## 使用与开发

- [README](../README.md)：安装、快速运行、能力边界与 CLI；
- [产物存储策略](artifact-storage.md)：仓库外默认目录、环境变量和备份分层；
- [比赛一页纸](COMPETITION-ONE-PAGER.md)：30 秒价值说明、Poisson 应用形态主结果、合成优势区间、
  评分映射与 3–5 分钟演示；
- [架构](architecture.md)：ordinary 默认入口、五类能力结果、显式 legacy contract、bootstrap 与证据边界；
- [Ordinary script 数值验证](script-verification.md)：`verify-script` 的 report/source/artifact 绑定、双进程 named-output 合同、退出码与可信边界；
- [Ordinary script strict-NPU resident benchmark](script-benchmark.md)：`benchmark-script` 的 marker 分段、单核/同步/runtime 门、原始样本、统计与端到端非声明；
- [Gap 恢复状态与证据链](gap-recovery-state.md)：schema-2 typed stage receipts、write-once hash-chain、命名授权 profile，以及 installed binary 到 Adapter 候选的状态与授权边界；
- [比赛要求验收映射](COMPETITION-REQUIREMENTS-MAP-20260827.md)：将赛题页码、评分点和发布阻断条件转为可执行验收；
- [2026-08-28 进展记录](PROGRESS-20260828.md)：MatMul DSO provenance 审计、catalog-v2/schema-3 修复、新恢复链及上游模型暂停状态；
- [当前整体架构图（draw.io）](numpilot-architecture.drawio)：三页可编辑架构图，依次为 ordinary 用户主流程、gap/资格化恢复流、安装与可信边界；
- [官方算子解析与缺口恢复架构图（draw.io）](numpilot-official-operator-resolution.drawio)：三页可编辑图，展示 installed ACLNN → pinned ops-nn → configured-model 新 kernel 的优先级、门禁与证据边界；
- [五日产品完善计划](FIVE-DAY-PRODUCT-HARDENING-PLAN-20260825.md)：2026-08-26～30 的当前执行主线，覆盖通用模型、持久化、qualified capability、ordinary 验证和 CANNBot 重启 RC；
- [通用模型与回执合同](model-contract.md)：配置 schema 2、精确/显式 alias 身份规则、失败回执与旧 schema 1 只读兼容；
- [初赛 10 天冲刺计划](COMPETITION-10-DAY-PLAN-20260825.md)：范围冻结、每日交付、里程碑与止损线；
- [全新 Notebook 验收路径](FRESH-NOTEBOOK-QUICKSTART.md)：从固定 revision、持久安装与 bootstrap 到 configured-model 迁移和显式授权实跑；
- [全新机器 AI Prompt](FRESH-MACHINE-AI-PROMPT.md)：只克隆 NumPilot、源码安装并在迁移前停点；
- [官方算子复用评估与实现状态](OPS-NN-REUSE-ASSESSMENT-20260825.md)：缺口后的官方候选优先级、只读 resolver/收据门禁，以及仍待完成的语义审计、构建/私有 OPP/资格化范围；
- [质量场景](quality-scenarios.md)：关键质量属性与验收场景；
- [CodeArts 接入](codearts-integration.md)：比赛多智能体配置与演示；
- [贡献指南](../CONTRIBUTING.md)、[安全策略](../SECURITY.md)、
  [第三方说明](../THIRD_PARTY_NOTICES.md)。

## 评测与证据

- [可公开证据包](../evidence/README.md)：随仓库分发的结构化结果与 SHA-256；
- [Poisson-Ritz 源项与鲁棒残差阶段](../evidence/competition-poisson-stage-20260828/README.md)：
  应用形态主结果，resident `2.567×`、保守 `2.471×`；一次性组件和 `0.789×` 且不是直接 E2E，
  不是完整求解器或公开数据负载；
- [逐元素合成优势区间](../evidence/competition-ordinary-20260828/README.md)：resident `8.079×`、
  组件和 `1.220×`（非直接 E2E），只作为大数组设备驻留区间；
- [比赛证据只读审计器](../tools/audit_script_benchmark_receipt.py)：不执行用户代码、不导入
  NumPy/AsNumpy、不联网或访问 NPU，复核两份 evidence 的哈希链、门禁与统计；
- [ops-nn MatMul 缺口迁移案例](../evidence/ops-nn-matmul-20260827/README.md)：历史真机数值与
  host-fallback guard 证据；旧 DSO candidate provenance 已被 2026-08-28 审查否决，当前状态以
  [当日进展记录](PROGRESS-20260828.md)为准；
- [Demo v1 Case 01 ordinary 定向证据](../evidence/demotest-case01-20260825/README.md)：外部受信、哈希绑定 harness 的数值与稳态/冷启动性能边界；
- [当前 7 个 GLM 迁移 demo 的性能对比](evidence/day4-demo-performance-analysis-20260826.md)：严格/诊断/fallback 门禁、微型 exact script、Case06 固定规模与 cold process 分层结论；
- [热扩散 strict-NPU 说明](evidence/heat-diffusion-npu-20260813.md)；
- [Poisson P0 strict-NPU 说明](evidence/poisson-ritz-p0-npu-20260821.md)；
- [Poisson profiler 说明](evidence/poisson-ritz-profiler-p0-20260822.md)；
- [Poisson 原始规模重复性能证据](evidence/poisson-ritz-performance-original-scale-20260823.md)；
- [Poisson 广播/CSE 优化证据](evidence/poisson-ritz-broadcast-optimization-20260823.md)；
- [竞争力与缺口评审](COMPETITION-GAP-REVIEW-20260821.md)。

## 设计记录与开发历史

`adr/` 是仍有效的架构决策记录。`AI-HANDOFF.md`、`NEXT-AI-PROMPT.md`、
`WORK-IN-PROGRESS-*`、`*-WIP-HANDOFF-*` 和 `superpowers/` 是按日期保存的开发历史，便于
追溯决策，但不应代替当前 README、代码或结构化证据。若历史路径或命令与当前 CLI 不同，
以当前 README 和 `python -m numpilot --help` 为准。
