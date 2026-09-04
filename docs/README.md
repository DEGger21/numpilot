# NumPilot 文档中心

[English README](../README.md) · [中文 README](../README.zh-CN.md)

这里仅保留面向安装、使用、评审、架构复核和发布验收的当前文档。开发交接稿、阶段计划、失败
轮次和未经筛选的原始运行材料不进入始终可发布的 `main`。

答辩视频、逐页讲稿和截图占位 PPT 属于独立提交物，不放在公开源码树；请在比赛提交包中按同一
个冻结 commit 单独绑定。

## 快速入口

| 目标 | 文档 |
|---|---|
| 了解项目价值与当前边界 | [English README](../README.md) · [中文 README](../README.zh-CN.md) |
| 快速安装并分层验收 | [English installation guide](installation.md) · [中文安装指南](installation.zh-CN.md) |
| 在全新 Notebook 完成比赛链验收 | [全新 Notebook 快速开始](FRESH-NOTEBOOK-QUICKSTART.md) |
| 现场逐项验收 | [可执行 Notebook](NumPilot-Demo-v1-Quickstart.ipynb) |
| 比赛提交前就绪度审计 | [Competition readiness（只读）](COMPETITION-READINESS.md) |
| 终端完整迁移录屏 | [命令行迁移录屏手册](TERMINAL-COMPLETE-MIGRATION-DEMO.zh-CN.md) |
| CodeArts/终端/hybrid 三种入口 | [CodeArts 接入与模型来源门禁](codearts-integration.md) · [NumPilot Skill 根目录入口](../skill/SKILL.md) · [CodeArts 镜像](../.codeartsdoer/skills/numpilot-migration-gate/SKILL.md) |
| AsNumpy 缺失 API 自定义算子补全 | [静态 PNG 流程图](numpilot-asnumpy-custom-operator-flow.png) · [固定坐标 draw.io 源](numpilot-asnumpy-custom-operator-flow.drawio) |
| 理解 ordinary 与 gap 两条路径 | [系统架构](architecture.md) |
| 查看三页可编辑架构图 | [NumPilot 架构图](numpilot-architecture.drawio) |
| 查看完整 v6 架构图 | [中文 SVG](numpilot-zh-v6.svg) · [English SVG](numpilot-en-v6.svg) |
| 审计架构状态边界 | [中文边界图](assets/architecture/numpilot-architecture-zh.svg) · [English boundary map](assets/architecture/numpilot-architecture-en.svg) |
| 下载单页架构源文件（非完整答辩 deck） | [中文 PPT](numpilot-zh.pptx) · [English PPT](numpilot-en.pptx) |
| 固化 AsNumpy 内部组件、禁止浮动升级 | [内部组件锁定与发布合同](ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md) |
| 核对 Logo 来源与发布权利 | [品牌资产清单](assets/brand/README.md) |
| 复核比赛证据 | [证据索引](../evidence/README.md) |
| 了解 `main` / `develop` 双分支治理 | [分支治理与比赛提交约定](SUBMISSION.md) |
| 准备提交包 | [比赛 RC 发布清单](RELEASE-CHECKLIST-DEMO-V1.md) |

## 入口与 `llm.json` 约定

终端 configured lane 像传统编程工具一样使用：用户在仓库外填写私有 `llm.json`，启动
`numpilot` 后输入 `check llm` 做真实小请求，再输入 `migrate` 开始交互迁移。无需模型的低层手动
命令仍可单独使用。AI/CodeArts-agent lane 则让交互模型先读 Skill 并自行承担解读、补全、算子/
接入和迁移候选生成，硬性不读取或调用 `llm.json`。代理地址只是现场环境特例，不是仓库或 Skill
配置；框架只检查模型能否返回有效内容。

### 交互式任意源码入口

录制或现场演示时，运行裸命令 `numpilot` 进入带 Logo 的持久 TTY。`migrate` 会手动询问待迁移
代码的绝对路径，先做只读静态分析并显示 API/缺口，再用私有 `llm.json` 请求一次未验证的迁移前
解读；用户看到解读后输入 `y` 才进入缺口路由和 NPU 资格化，并在实际执行原始/规范化源码前于
紧邻执行点另行输入精确的 `execute`。最终只从回执显示迁移后代码位置、所有生成文件、运行状态、
输出规模和有效性能数字；任一授权门拒绝、EOF 或失败都保持相应后续阶段 `not_run`，不伪造
加速比。最小已登记缺口案例为
`examples/add_one_migration_demo.py`；它将未激活的 CANNBot Ascend C 草稿与独立真 NPU 资格化的
run-scoped AsNumpy 组合接口分开报告。

如果用户明确要求“阅读这个 skill、使用 agent 模式、不用 `llm.json` 里的内置模型”，这是
CodeArts-agent/no-llm 入口：在任何配置检查前停止 configured 路径，不读取或探测 `llm.json`，由
侧边栏模型自己完成解读、迁移、缺口补全和算子生成，并把候选标记为 `agent_authored_draft`。

## 核心合同

- [模型与回执合同](model-contract.md)：OpenAI-compatible 配置、模型身份、失败回执和隐私边界；
- [脚本数值验证](script-verification.md)：双进程 named-output 数值门、哈希绑定与退出码；
- [脚本性能评测](script-benchmark.md)：CPU 限核、strict-NPU、同步、样本统计和非 E2E 边界；
- [Gap 恢复状态](gap-recovery-state.md)：typed receipts、write-once 状态链与授权 profile；
- [产物存储策略](artifact-storage.md)：仓库外运行目录、证据筛选和备份分层；
- [质量场景](quality-scenarios.md)：关键质量属性及失败关闭验收；
- [CodeArts 接入](codearts-integration.md)：三种入口、模型来源硬门禁、四角色协作控制面及演示方式。

## 比赛与治理

- [比赛就绪度审计](COMPETITION-READINESS.md)：评分维度、必要附件与证据缺口的只读检查；
- [贡献指南](../CONTRIBUTING.md)；
- [安全策略](../SECURITY.md)；
- [第三方软件说明](../THIRD_PARTY_NOTICES.md)；
- [架构决策记录](adr/)；
- [官方算子解析图](numpilot-official-operator-resolution.drawio)。

## 当前公开证据

以下均为 `git_dirty=true` 的开发态、精确画像回执，不是发布基线；数字不得外推到其他硬件、软件、
shape、dtype 或计时口径。

- [Poisson-Ritz 应用阶段](../evidence/competition-poisson-stage-20260828/README.md)：
  resident 中位 `2.567×`、保守 `2.471×`；一次组件和 `0.789×`，不是直接 E2E；
- [逐元素优势区间](../evidence/competition-ordinary-20260828/README.md)：
  resident 中位 `8.079×`、保守 `7.503×`；合成案例，不外推为通用性能；
- [只读 benchmark 审计器](../tools/audit_script_benchmark_receipt.py)：不导入 NumPy/AsNumpy、
  不访问网络或 NPU，独立复核哈希链、门禁和统计。

若文档示例与命令行帮助不一致，以当前 `main` 的 `python -m numpilot --help` 和结构化回执为准。
