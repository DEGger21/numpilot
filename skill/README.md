# NumPilot Skill

CLI 是 NumPilot 唯一的正式执行契约；本 Skill 是可选的 IDE/CodeArts 薄适配器。它只负责选择入口 lane、收集审批、调用 analyze/migrate/verify/replay 并展示同一套 JSON 回执，不实现第二套迁移或验证逻辑。

这是仓库根目录下的人类可见 Skill 入口。核心说明在 [`SKILL.md`](SKILL.md)，侧边栏界面
元数据在 [`agents/openai.yaml`](agents/openai.yaml)，按需读取的终端与 CodeArts 参考在
[`references/`](references/)。

AsNumpy 缺口补全另有独立 Skill：[`skill/asnumpy-gap-completion/`](asnumpy-gap-completion/)。它只负责
GapSpec 到 `_core` 增量 custom-op overlay 的合同、custom OPP provenance、显式 dlsym 隔离和
数值/性能门禁；CANNBot 仍负责算子内核生成。CodeArts 发现副本位于
`.codeartsdoer/skills/asnumpy-gap-completion/`。

CodeArts 的标准发现路径仍是
`.codeartsdoer/skills/numpilot-migration-gate/`；该目录保留逐字节相同的 Skill 内容，便于
CodeArts 加载；不要只修改其中一份。

快速入口：

```bash
# 终端 configured lane：启动持久会话后检查模型，再开始交互迁移
numpilot
# numpilot › check llm
# numpilot › migrate

# AI/CodeArts lane：把下面一句发给交互模型
# “阅读这个 Skill，使用 agent 模式，不用 llm.json 里的内置模型进行迁移。”

# 自动化适配器仍调用同一 CLI 流程，但必须显式授权
python -m numpilot --llm-config /absolute/private/llm.json \
  interactive-migrate --source /absolute/path/to/source.py --yes --execute-source
```

Skill 是路由和证据合同，不是模型本身。代理地址是现场环境细节，不能写死；`llm.json`、Token
和 endpoint 由终端用户在私有配置中提供。agent/no-llm 模式则完全不读取该配置，由交互模型自行
承担解读、补全、算子/接入和迁移候选生成。

Codex CLI / 外部 AI 也可以直接作为候选生成角色：先执行 complete-gaps --provider prepare 获取 GapSpec，再把 AI 生成的 JSON 或 NUMPILOT_BUNDLE_V1 写入本地文件，通过 --provider injected --response-file 送入同一套合同和 artifact store。该通道不读取 llm.json，结果固定为 generated_unverified，不会自动编译、激活或更新 150 例统计。详细 hand-off 见 asnumpy-gap-completion/references/codex-cli.md。
