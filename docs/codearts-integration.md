# NumPilot × 华为云码道（CodeArts）集成建议

> 调研日期：2026-07-18。以下能力以华为云官方在线文档为准；产品仍在快速迭代，比赛演示前应再次核对套餐、区域和客户端版本。

## 结论

建议采用“CodeArts 协作控制面 + NumPilot 确定性执行面”的两层架构，而不是把两者混成一套难以验证的 Agent 代码：

```text
CodeArts AgentTeam / 自定义智能体
  ├─ 需求拆解、角色协作、失败诊断、人工确认、Diff 与过程展示
  └─ Skill / 终端 / 后续 MCP 调用
          ↓ 结构化任务与工件
NumPilot CLI + contracts + capability registry
  ├─ ordinary 五态分析 →（terminal/manual、CodeArts-agent 或 hybrid）候选 → 确定性数值验证
  │                         → strict-NPU resident benchmark（仅证据充分时）
  └─ 缺口时 official-first 解析 → 候选/资格化/激活门 → AsNumpy / CANN / Ascend NPU
```

这样比赛中的“多智能体”来自真实 CodeArts AgentTeam 和项目级自定义智能体，NumPilot 则承担必须可重复、可测试、可审计的领域逻辑。

## 模型来源与入口门禁

CodeArts 交互模型可以直接承担迁移、缺口补全和算子/接入候选的生成；NumPilot 负责确定性的
分析、哈希、执行和证据门禁。先在两种模型来源中选择一种；hybrid 只是显式 handoff，不能静默
混用：

- **terminal/configured lane（传统工具用法）**：用户填写私有 `llm.json`，运行 `numpilot` 进入
  带 Logo 的持久终端；`check llm` 做真实小请求，`migrate` 询问绝对路径、显示只读 API/缺口分析
  和 configured 模型的 `interpreted_unverified` 解读。用户输入 `y` 后才进入缺口补全与 NPU
  资格化；在执行已展示且绑定哈希的用户源码前，还必须在紧邻执行点输入精确的 `execute`。无需
  模型的低层手动命令仍可单独使用。
- **CodeArts-agent lane（侧边栏交互）**：先读仓库和 Skill，再由交互模型发散拆解并独立完成
  迁移/补全/算子草稿。用户明确要求 CodeArts 模型或“不用 `llm.json`/内置模型”时，执行硬
  门禁：不得读取 `llm.json`、运行 `configure-llm`、传入 `--provider llm`、探测 endpoint，
  也不得因失败回退。候选放在隔离目录并标记 `agent_authored_draft`；NumPilot 只做可支持的
  确定性 gate，未有适配器绑定完整 agent/provider/model/response/source/artifact 证据前，不得
  发布 configured-model、strict-NPU 或性能结论。仅读取 Skill/生成 handoff 的初始回执必须写
  `candidate_status=not_generated`、`agent_authored_draft=false`；只有交互模型实际写出候选后，
  才另发带来源哈希的 `codearts_agent_receipt` 并使用草稿标签。
- **hybrid handoff**：CodeArts 负责对话、计划和（如请求）生成/审阅草稿，终端负责确定性
  `analyze`、哈希、`verify-script`、`benchmark-script`。保留 CodeArts task/event 与草稿标签；
  若要在 hybrid 中切换 configured model，必须重新声明、授权并新建 run 目录，不能混合回执。

若用户没有说明入口，先询问再接触任何模型配置。代理地址仅是用户可选的运行环境变量（例如
现场设置 `HTTP_PROXY`/`HTTPS_PROXY`），仓库和 Skill 不写死地址、凭据或 endpoint。

可直接把下面的 no-llm 指令粘贴到 CodeArts 侧边栏：

```text
请阅读 $numpilot-migration-gate 和 NumPilot 仓库，使用 agent 模式，明确不用 llm.json 和
NumPilot 内置模型。由你与我交互完成源码解读、缺口补全、算子生成、接入和迁移候选，再执行
可用的确定性验证；如果没有受支持的 migration-report 适配器，就保留
migrate=draft/stopped、verify=not_run、benchmark=not_run，报告回执哈希和不能外推的结论。
```

当前 `verify-script`/`benchmark-script` 仍消费受支持的 hash-bound migration report；若手动或
CodeArts 候选尚无适配器可物化该回执，应停在 `draft` 并展示边界，不得手工伪造 configured-model
报告。

需要注意：官方文档分别说明了 AgentTeam 的动态 Teammate 机制和项目级自定义智能体，但没有明确承诺 AgentTeam 会自动把 `.codeartsdoer/agents/` 中的角色装配成固定队伍。因此比赛账号应提前做一次产品实测。若不能自动装配，就把它们作为两种互补入口展示：AgentTeam 用于动态并行与任务图，自定义 `numpilot-orchestrator` 用 `task` 调用项目子智能体并执行严格领域流程；不要在 PPT 中把尚未验证的自动装配写成现有能力。

## 为什么匹配

CodeArts 官方的 [Agent Team 多任务并行](https://support.huaweicloud.com/usermanual-space/codeartsagent_space_0003.html)采用“Team Leader 编排 + Teammate 自主执行”，支持持续独立上下文、成员双向通信、共享任务池和动态组队。这可以直接承载 NumPilot 的分析、算子工程和证据审计角色。

[自定义智能体](https://support.huaweicloud.com/usermanual-codeartssnap/codeartsagent_ug_0051.html)支持项目级 Markdown 配置，存放在 `.codeartsdoer/agents/`，可配置主/子智能体和 read/edit/bash/task 等工具权限。因此仓库已提供：

- `numpilot-orchestrator`：任务负责人和门禁编排；
- `numpilot-analyzer`：只读能力/数据流分析；
- `numpilot-operator-engineer`：受控的 Ascend C/adapter 实现；
- `numpilot-evidence-auditor`：与开发职责隔离的独立证据审计。

CodeArts [Skill](https://support.huaweicloud.com/usermanual-codeartssnap/codeartsdoer_ug_0024.html)采用
`SKILL.md + scripts/references/assets` 的目录规范，并可随仓库放在 `.codeartsdoer/skills/`。
本仓库的 `numpilot-migration-gate` 将交互分析、doctor、真机、strict provenance 和独立复核固化
为统一流程；仓库中的 analyzer/orchestrator/operator/evidence 角色也必须遵守同一入口与 no-llm
门禁。精确 `numpy.add(array, 1)` 是一个窄的已登记例外：CANNBot Ascend C 产物仍是未激活草稿，
另由框架资格化 run-scoped AsNumpy 组合接口。它不能泛化到其他缺口。当前 CLI 没有通用的
CodeArts/manual 草稿到 ordinary migration-report 物化器，无适配器时只能保留 draft 并把后续
verify/benchmark 标为 `not_run`。

## CodeArts 能力如何使用

| CodeArts 能力 | NumPilot 用法 | 比赛证据 |
|---|---|---|
| AgentTeam | Leader 分解迁移任务，分析/算子/审计并行协作 | Task List、成员事件流、失败后重调度记录 |
| 项目级自定义智能体 | 固化角色、工具权限和门禁 | 仓库中的 `.codeartsdoer/agents/*.md` |
| Skill | 固化三种入口的 NPU 环境检查、迁移、验证和 benchmark SOP | 根目录 [`skill/`](../skill/)（CodeArts 镜像 `.codeartsdoer/skills/numpilot-migration-gate/`）、lane 标记及调用记录 |
| 知识库 | 导入 CANN/Ascend C/AsNumpy 版本化资料与历史缺陷 | 检索命中来源与版本，不把知识库内容直接当 capability 真相 |
| Remote SSH | 让 CodeArts IDE 连接这类有真实 NPU 的开发机 | 同一会话中的源码修改、裸金属测试和设备日志 |
| 沙箱/人工确认 | 普通分析在沙箱；NPU、构建、安装按最小命令授权 | 审批记录和命令白名单 |
| MCP（下一步） | 对外暴露只读 analyze、doctor、capabilities 和受控 migrate/verify | 结构化入参/出参、审计日志、最小权限工具 |
| Hooks（下一步） | 在提交/工具调用前强制检查 report、provenance 和危险安装命令 | 确定性 gate 日志 |

官方还提供 [知识库](https://support.huaweicloud.com/usermanual-codeartssnap/codeartsdoer_ug_0013.html)、[Remote SSH](https://support.huaweicloud.com/usermanual-codeartssnap/codeartsdoer_ug_0030.html)、[MCP](https://support.huaweicloud.com/usermanual-codeartssnap/codeartsdoer_ug_0010.html)、[Hooks](https://support.huaweicloud.com/usermanual-codeartssnap/codeartsdoer_ug_0036.html)和[沙箱](https://support.huaweicloud.com/usermanual-codeartssnap/codeartsdoer_ug_0045.html)。这些能力适合作为工程化亮点，但不应在尚未接通时写成“已实现”。

## 推荐演示脚本

1. 在 CodeArts Agent Space 选择 AgentTeam，输入一个真实科学网络迁移需求。
2. Leader 让 analyzer 生成 callsite、shape/dtype 和 capability 缺口清单。
3. 无缺口时按已选 lane 调用 `numpilot-migration-gate`：terminal/manual lane 可由用户填写
   可选 `llm.json`，或直接提供候选；CodeArts-agent lane 由侧边栏模型先生成隔离的
   `agent_authored_draft`；hybrid lane 将该草稿交给终端。三者最后都由 NumPilot 做确定性
   gate；只有显式授权的 configured sub-run 才执行 `migrate`。有缺口时先生成 GapSpec 并交给
   operator engineer；不得绕过官方优先和模型来源门禁。
4. 在 Remote SSH 的 NPU 机展示迁移 diff、exact AST 门、named-output 数值门和
   runtime/profile/call-count/sync 门。
5. evidence auditor 独立复跑；任一门失败或命中 CPU fallback 时，界面明确显示
   `performance_valid=false`，不得发布 speedup。
6. 最后展示报告中从需求、任务、代码、测试、capability、kernel/profiler 到性能数字的证据链。

## 边界与下一步

- AgentTeam 目前仅在 CodeArts Agent Space 使用；演示环境和账号套餐需提前验证。
- 云端自定义智能体只在基础版/专业版支持，但项目级本地配置可随仓库分发；应在目标账号实测是否全部加载成功。
- 先用项目 Skill + 终端完成可信闭环，再实现一个最小 MCP Server。MCP 第一版只开放白名单命令和结构化 schema，不暴露任意 shell。
- CodeArts 知识库用于检索文档，真正可执行 capability 仍由 NumPilot registry、目标 wheel/CANN/SoC 与测试证据决定。
- 自定义算子安装会影响共享 OPP，必须保留人工确认、隔离目录、artifact hash 和回滚方案。
- `write/edit: false` 不能约束 `bash` 内的重定向或修改命令；evidence auditor 为复跑测试保留了 bash，因此仍须使用 CodeArts 沙箱、Hook/命令白名单和独立证据目录形成硬边界。
