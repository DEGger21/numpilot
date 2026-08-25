# NumPilot 文档导航

## 使用与开发

- [README](../README.md)：安装、快速运行、能力边界与 CLI；
- [产物存储策略](artifact-storage.md)：仓库外默认目录、环境变量和备份分层；
- [架构](architecture.md)：ordinary 默认入口、五类能力结果、显式 legacy contract、bootstrap 与证据边界；
- [当前整体架构图（draw.io）](numpilot-architecture.drawio)：三页可编辑架构图，依次为 ordinary 用户主流程、gap/资格化恢复流、安装与可信边界；
- [初赛 10 天冲刺计划](COMPETITION-10-DAY-PLAN-20260825.md)：范围冻结、每日交付、里程碑与止损线；
- [全新 Notebook 验收路径](FRESH-NOTEBOOK-QUICKSTART.md)：从固定 tag、安装与 bootstrap 到 GLM 迁移和 NPU 实跑；
- [ops-nn 官方算子复用评估](OPS-NN-REUSE-ASSESSMENT-20260825.md)：缺口后的官方候选优先级、门禁与 P1 待定范围；
- [质量场景](quality-scenarios.md)：关键质量属性与验收场景；
- [CodeArts 接入](codearts-integration.md)：比赛多智能体配置与演示；
- [贡献指南](../CONTRIBUTING.md)、[安全策略](../SECURITY.md)、
  [第三方说明](../THIRD_PARTY_NOTICES.md)。

## 评测与证据

- [可公开证据包](../evidence/README.md)：随仓库分发的结构化结果与 SHA-256；
- [Demo v1 Case 01 ordinary 定向证据](../evidence/demotest-case01-20260825/README.md)：外部受信、哈希绑定 harness 的数值与稳态/冷启动性能边界；
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
