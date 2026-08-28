# NumPilot 文档中心

这里仅保留面向使用、评审、架构复核和发布验收的当前文档。开发交接稿、阶段计划、失败轮次
和历史性能原始文件不进入比赛发布分支。

## 快速入口

| 目标 | 文档 |
|---|---|
| 了解项目价值与当前成绩 | [比赛一页纸](COMPETITION-ONE-PAGER.md) |
| 从新机器安装并运行 | [全新 Notebook 快速开始](FRESH-NOTEBOOK-QUICKSTART.md) |
| 现场逐项验收 | [可执行 Notebook](NumPilot-Demo-v1-Quickstart.ipynb) |
| 理解 ordinary 与 gap 两条路径 | [系统架构](architecture.md) |
| 查看三页可编辑架构图 | [NumPilot 架构图](numpilot-architecture.drawio) |
| 用于论文或答辩主图 | [中文可编辑架构 PPT](numpilot-architecture-paper-style-zh-readable.pptx) |
| 复核比赛证据 | [证据索引](../evidence/README.md) |
| 在发布分支继续开发 | [比赛提交分支约定](SUBMISSION.md) |
| 准备提交包 | [比赛 RC 发布清单](RELEASE-CHECKLIST-DEMO-V1.md) |

## 核心合同

- [模型与回执合同](model-contract.md)：OpenAI-compatible 配置、模型身份、失败回执和隐私边界；
- [脚本数值验证](script-verification.md)：双进程 named-output 数值门、哈希绑定与退出码；
- [脚本性能评测](script-benchmark.md)：CPU 限核、strict-NPU、同步、样本统计和非 E2E 边界；
- [Gap 恢复状态](gap-recovery-state.md)：typed receipts、write-once 状态链与授权 profile；
- [产物存储策略](artifact-storage.md)：仓库外运行目录、证据筛选和备份分层；
- [质量场景](quality-scenarios.md)：关键质量属性及失败关闭验收；
- [CodeArts 接入](codearts-integration.md)：四角色协作控制面及演示方式。

## 比赛与治理

- [赛题要求映射](COMPETITION-REQUIREMENTS-MAP-20260827.md)：评分点到可执行门禁的映射；
- [贡献指南](../CONTRIBUTING.md)；
- [安全策略](../SECURITY.md)；
- [第三方软件说明](../THIRD_PARTY_NOTICES.md)；
- [架构决策记录](adr/)；
- [官方算子解析图](numpilot-official-operator-resolution.drawio)。

## 当前公开证据

- [Poisson-Ritz 应用阶段](../evidence/competition-poisson-stage-20260828/README.md)：
  resident 中位 `2.567×`、保守 `2.471×`；一次组件和 `0.789×`，不是直接 E2E；
- [逐元素优势区间](../evidence/competition-ordinary-20260828/README.md)：
  resident 中位 `8.079×`、保守 `7.503×`；合成案例，不外推为通用性能；
- [只读 benchmark 审计器](../tools/audit_script_benchmark_receipt.py)：不导入 NumPy/AsNumpy、
  不访问网络或 NPU，独立复核哈希链、门禁和统计。

若文档示例与命令行帮助不一致，以当前发布分支的 `python -m numpilot --help` 和结构化回执为准。
