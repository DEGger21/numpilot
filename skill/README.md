# NumPilot Skill

这是仓库根目录下的人类可见 Skill 入口。核心说明在 [`SKILL.md`](SKILL.md)，侧边栏界面
元数据在 [`agents/openai.yaml`](agents/openai.yaml)，按需读取的终端与 CodeArts 参考在
[`references/`](references/)。

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

# 自动化入口仍使用同一框架流程，但必须显式授权
python -m numpilot --llm-config /absolute/private/llm.json \
  interactive-migrate --source /absolute/path/to/source.py --yes --execute-source
```

Skill 是路由和证据合同，不是模型本身。代理地址是现场环境细节，不能写死；`llm.json`、Token
和 endpoint 由终端用户在私有配置中提供。agent/no-llm 模式则完全不读取该配置，由交互模型自行
承担解读、补全、算子/接入和迁移候选生成。
