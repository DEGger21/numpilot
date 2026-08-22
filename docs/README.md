# NumPilot 文档导航

## 使用与开发

- [README](../README.md)：安装、快速运行、能力边界与 CLI；
- [产物存储策略](artifact-storage.md)：仓库外默认目录、环境变量和备份分层；
- [架构](architecture.md)：智能体、迁移引擎、后端与证据链；
- [质量场景](quality-scenarios.md)：关键质量属性与验收场景；
- [CodeArts 接入](codearts-integration.md)：比赛多智能体配置与演示；
- [贡献指南](../CONTRIBUTING.md)、[安全策略](../SECURITY.md)、
  [第三方说明](../THIRD_PARTY_NOTICES.md)。

## 评测与证据

- [可公开证据包](../evidence/README.md)：随仓库分发的结构化结果与 SHA-256；
- [热扩散 strict-NPU 说明](evidence/heat-diffusion-npu-20260813.md)；
- [Poisson P0 strict-NPU 说明](evidence/poisson-ritz-p0-npu-20260821.md)；
- [Poisson profiler 说明](evidence/poisson-ritz-profiler-p0-20260822.md)；
- [竞争力与缺口评审](COMPETITION-GAP-REVIEW-20260821.md)。

## 设计记录与开发历史

`adr/` 是仍有效的架构决策记录。`AI-HANDOFF.md`、`NEXT-AI-PROMPT.md`、
`WORK-IN-PROGRESS-*`、`*-WIP-HANDOFF-*` 和 `superpowers/` 是按日期保存的开发历史，便于
追溯决策，但不应代替当前 README、代码或结构化证据。若历史路径或命令与当前 CLI 不同，
以当前 README 和 `python -m numpilot --help` 为准。
