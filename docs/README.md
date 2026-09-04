# NumPilot 文档

[English README](../README.md) · [中文 README](../README.zh-CN.md)

NumPilot 将 NumPy 程序的静态分析、迁移候选生成、数值验证和 Ascend NPU 运行证据组织成一套
失败关闭的工作流。本页只索引当前产品文档；运行产生的源码、回执和性能数据默认保存在仓库外。

## 开始使用

- [English installation guide](installation.md) / [中文安装指南](installation.zh-CN.md)：安装、
  AsNumpy 组件激活和分层验收；
- [模型与回执合同](model-contract.md)：`llm.json`、OpenAI-compatible 接口、隐私和来源边界；
- [CodeArts 接入](codearts-integration.md)：终端、侧边栏 Agent 和 hybrid 协作方式；
- [NumPilot Agent Skill](../skill/README.md)：供侧边栏 AI 阅读的迁移工作流入口。

## 工作流与验证

- [系统架构](architecture.md)：ordinary 脚本路径、能力判定和缺口恢复边界；
- [脚本数值验证](script-verification.md)：隔离 worker、输出比较和回执绑定；
- [脚本性能评测](script-benchmark.md)：CPU/NPU 计时口径和 strict-NPU 约束；
- [Gap 恢复状态](gap-recovery-state.md)：授权 profile、状态机和 write-once 收据；
- [产物存储策略](artifact-storage.md)：配置、缓存、运行目录和备份边界；
- [质量场景](quality-scenarios.md)：主要质量属性与失败关闭验收。

## 组件、集成与发布

- [AsNumpy 内部组件锁定](ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md)；
- [AsNumpy 接入说明](../integrations/asnumpy/README.md)；
- [分支治理与发布约定](SUBMISSION.md)；
- [贡献指南](../CONTRIBUTING.md)；
- [安全策略](../SECURITY.md)；
- [第三方软件说明](../THIRD_PARTY_NOTICES.md)；
- [架构决策记录](adr/)。

## 架构图

- [English architecture](numpilot-en-v6.svg) / [中文架构](numpilot-zh-v6.svg)；
- [可编辑 Draw.io 源](numpilot-architecture.drawio)；
- [两种使用方式](assets/architecture/numpilot-two-routes.zh-CN.png)；
- [AsNumpy 自定义算子接入](numpilot-asnumpy-custom-operator-flow.png)；
- [官方算子解析流程](numpilot-official-operator-resolution.drawio)。

若文档示例与当前命令行不一致，以 `python -m numpilot --help` 和机器可读回执为准。
