# 运行产物与证据存储

NumPilot 将可版本化源码与运行时状态分开。克隆仓库后，默认命令不会在仓库根生成
`reports/`、`out_demo/`、`out_wizard/` 或 `out_gaps/`。

## 默认目录

```bash
python -m numpilot paths
python -m numpilot paths --json
```

在未覆盖环境变量时，目录遵循 XDG 约定：

| 内容 | 默认路径 |
|---|---|
| 状态根 | `$XDG_STATE_HOME/numpilot`，通常为 `~/.local/state/numpilot` |
| 运行产物 | `<状态根>/runs` |
| 迁移报告 | `<状态根>/reports` |
| LLM 私有配置 | `$XDG_CONFIG_HOME/numpilot/llm.json` |
| CANNBot Skill 缓存 | `$XDG_CACHE_HOME/numpilot/cannbot-skills/<commit>` |

覆盖优先级为命令行 `--out` / `--reports-dir`、NumPilot 专用环境变量、XDG 默认值：

```bash
export NUMPILOT_STATE_HOME=/data/numpilot-state
export NUMPILOT_RUNS_DIR=/data/numpilot-runs
export NUMPILOT_REPORTS_DIR=/data/numpilot-reports
```

`NUMPILOT_CONFIG_HOME`、`NUMPILOT_LLM_CONFIG_FILE` 和 `NUMPILOT_CANNBOT_CACHE` 分别控制
凭据配置与 Skill 缓存。不要把这些目录指回 Git 工作树。

## 三层保留策略

1. **主仓库**：源码、测试、公开文档、精选结构化证据与其 SHA-256；
2. **运行根**：完整 LLM 回执、生成/repair 候选、qualification、benchmark、profiler、
   wheel/OPP 和失败记录；
3. **冷备份/发布附件**：运行根的只读归档，带归档级 SHA-256、条目数和恢复说明。

虚拟环境、上游源码克隆和普通 build 目录通常可重建，但在固定 release manifest、上游 commit、
补丁、构建命令与必要制品之前不要删除。评委需要复核的原始 evidence 应作为 release 附件或
比赛交付包发布，不应把数百 MB 的工作盘直接提交到源码 Git。

## 兼容与迁移

旧版在仓库内生成的 `reports/`、`out_*` 和日志仍可读取；本次改动不会自动删除或覆盖它们。
迁移旧文件时先做完整归档和 SHA-256，再移动到独立运行根。路径变化不影响已显式传入
`--out` 的脚本；建议 CI 一律显式指定临时目录。
