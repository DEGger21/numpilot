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
| 配置根 | `$XDG_CONFIG_HOME/numpilot`，通常为 `~/.config/numpilot` |
| 应用缓存根 | `$XDG_CACHE_HOME/numpilot`，通常为 `~/.cache/numpilot` |
| 运行产物 | `<状态根>/runs` |
| 迁移报告 | `<状态根>/reports` |
| LLM 私有配置 | `<配置根>/llm.json` 或源码 checkout 根的 `llm.json` |
| AsNumpy 源码缓存 | `<应用缓存根>/asnumpy/<commit>` |
| CANNBot Skill 缓存 | `$XDG_CACHE_HOME/numpilot/cannbot-skills/<commit>` |

冻结 release 另有一条**内部 AsNumpy 组件**边界：用户安装使用发行附件或受控内部制品库中的
精确 wheel（推荐 `components/asnumpy/`），而不是依赖上面的可重建源码缓存。组件文件、wheel
metadata/RECORD、安装后 Python/native 模块和 bootstrap receipt 的 SHA-256 必须进入同一份
release manifest；发布模式不得因缓存为空而联网拉取 AsNumpy，也不得后台升级已安装组件。详见
[AsNumpy 内部组件锁定与发布合同](ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md)。源码缓存只供
隔离构建下一版使用，不能作为用户安装时的隐式 fallback。

临时 HOME/Notebook 可在安装时传 `--persistent-root /absolute/root`。统一优先级是：显式细分
变量 > `NUMPILOT_HOME`（或其描述性别名 `NUMPILOT_RUNTIME_ROOT`） > XDG > 标准 HOME。单一根映射为
`<root>/.numpilot-state`、`<root>/.numpilot-config` 和
`<root>/.numpilot-cache/numpilot`。命令行 `--out` / `--reports-dir` 仍优先于这些默认值：

```bash
export NUMPILOT_STATE_HOME=/data/numpilot-state
export NUMPILOT_RUNS_DIR=/data/numpilot-runs
export NUMPILOT_REPORTS_DIR=/data/numpilot-reports
export NUMPILOT_RUNTIME_ROOT=/data/persistent-root
```

`NUMPILOT_CONFIG_HOME`、`NUMPILOT_CACHE_HOME`、`NUMPILOT_LLM_CONFIG_FILE`、
`NUMPILOT_ASNUMPY_CACHE` 和 `NUMPILOT_CANNBOT_CACHE` 分别控制更具体的目录；后两个变量都是
cache base，NumPilot 再追加固定 commit。`NUMPILOT_CANNBOT_SKILLS` 是经过校验的 exact checkout
覆盖，优先于 CANNBot cache base；`NUMPILOT_BOOTSTRAP_RECEIPT` 是精确文件覆盖。不要把会被跟踪的
私有配置或运行状态指回 Git 工作树。仓库根的 `llm.json` 是刻意保留的本地例外：它必须被
`.gitignore` 忽略、权限为 `0600`，且永不提交。

如果模型配置位于单独的持久运行根目录，可只指定运行时根，不必把凭据复制到仓库：

```bash
export NUMPILOT_RUNTIME_ROOT=/absolute/persistent/numpilot-runtime
python -m numpilot configure-llm --check
```

也可以对单次演示显式指定配置文件：

```bash
./run_demo.sh --llm-config /absolute/persistent/numpilot-runtime/.numpilot-config/llm.json --quick
```

`run_demo.sh --llm-config` 只传递私有 JSON 的路径，并覆盖同名继承环境变量；它不会把 Token
放进参数或日志。省略该选项时，脚本继续使用 `NUMPILOT_LLM_CONFIG_FILE`、已 source 的
`NUMPILOT_CONFIG_HOME` 或默认 XDG 路径。NumPilot 不会自动扫描固定机器路径，以免误读其他项目
或用户的凭据。

成功的安装（固定组件或维护者 source-first 模式）会生成 mode `0600` 的 `activate-numpilot.sh`。它只冻结本地路径、venv
和已校验 CANN set_env，不包含模型 Token、endpoint、代理或 pip 索引，也不修改 `.bashrc`。
每个新 shell source 一次即可；重复 source 会去重 PATH/LD_LIBRARY_PATH，并清理不同 CANN
toolkit 根及其 `devlib`，同时保留 driver、NNAL/ATB、无关 `devlib` 和其他自定义库路径。

## 三层保留策略

1. **主仓库**：源码、测试、公开文档、精选结构化证据与其 SHA-256；
2. **运行根**：完整 LLM 回执、生成/repair 候选、qualification、benchmark、profiler、
   wheel/OPP 和失败记录；
3. **冷备份/发布附件**：运行根的只读归档，带归档级 SHA-256、条目数和恢复说明。

虚拟环境、上游源码克隆和普通 build 目录通常可重建，但在固定 release manifest、上游 commit、
补丁、构建命令与必要制品之前不要删除。需要独立复核的原始 evidence 应作为 release 附件发布，
不应把数百 MB 的工作盘直接提交到源码 Git。

## 兼容与迁移

旧版在仓库内生成的 `reports/`、`out_*` 和日志仍可读取；本次改动不会自动删除或覆盖它们。
迁移旧文件时先做完整归档和 SHA-256，再移动到独立运行根。路径变化不影响已显式传入
`--out` 的脚本；建议 CI 一律显式指定临时目录。
