# NumPilot Demo — 执行 Checkpoint

> 落盘时间:2026-07-02 · 分支 `feat/numpilot-demo` · 当前 HEAD `52fdf4b`
> 用途:随时中断/恢复子智能体驱动开发(SDD)的进度快照。

## 一句话状态

11 个任务已完成 **4 个**(Tasks 1-4,均通过任务审查),从 **Task 5** 继续。所有代码已提交进 git,进度台账在 `.superpowers/sdd/progress.md`。

## 已完成任务(git 可追溯)

| Task | 内容 | 提交范围 | 审查 |
|------|------|----------|------|
| 1 | 项目骨架 + AsNumpy 覆盖集快照 | `b7d3ff6..b7eb340` | ✅ clean(1 Minor) |
| 2 | Agent 间 dataclass 契约层 | `b7eb340..e9f32c0` | ✅ clean(1 Minor) |
| 3 | 分析智能体(AST 算子提取 + 覆盖集比对) | `e9f32c0..5fb184e` | ✅ clean(2 Minor) |
| 4 | 决策智能体(规则成本模型 CPU/NPU) | `5fb184e..52fdf4b` | ✅ clean(2 Minor) |

当前测试:`cd /home/lixing/numpilot && python -m pytest -q` 应全绿(12/12)。

## 剩余任务(按计划 `docs/superpowers/plans/2026-07-02-numpilot-demo.md`)

- **Task 5** 迁移引擎(策略接口 + AST 改写 + LLM 可插拔)— BASE 从 `52fdf4b` 起
- **Task 6** 迁移智能体
- **Task 7** 执行后端(shim + npu 适配器)
- **Task 8** 示例程序(MLP 前向 + 缺口负例)
- **Task 9** 验证智能体
- **Task 10** 编排器 + CLI + 端到端
- **Task 11** 课程架构文档(4+1 / C4 / ADR)+ README

## 如何恢复(SDD 循环)

每个任务重复:生成 brief → 派实现子智能体 → 生成 review 包 → 派审查子智能体 → 修复(如有 Critical/Important)→ 记台账。脚本目录:
`/root/.claude/plugins/cache/claude-plugins-official/superpowers/6.1.0/skills/subagent-driven-development/scripts/`

```bash
cd /home/lixing/numpilot
BASE=$(git rev-parse --short HEAD)                       # 派实现前记录
SCR=/root/.claude/plugins/cache/claude-plugins-official/superpowers/6.1.0/skills/subagent-driven-development/scripts
$SCR/task-brief docs/superpowers/plans/2026-07-02-numpilot-demo.md 5   # 生成 Task 5 brief
# → 派 general-purpose 实现子智能体(Task 5/7/9/10 用 sonnet, 其余 haiku)
# 实现提交后:
HEAD=$(git rev-parse --short HEAD)
$SCR/review-package $BASE $HEAD                          # 生成 review 包 → 派 sonnet 审查
```

模型选择:Task 5(引擎)/7(后端)/9(验证)/10(编排)用 **sonnet** 实现;Task 6/8/11 用 **haiku**;审查一律 **sonnet**。

## 待最终审查处理的 Minor 汇总(不阻塞)

- T1:coverage 快照漏 `.io` 4 算子(save/savez/savez_compressed/load),真实 `_LAZY_MAPPING` 里有;可补。
- T2:契约用裸 `list/dict` 注解而非参数化(`list[OpUse]` 等);可提精度。
- T3:`_est_elements` 未处理空 shape `()`(0-d=1 元素)、bool 当 int;out-of-scope。
- T4:决策测试未覆盖同名算子多调用点聚合、阈值边界 `==`;建议补测。

Task 11 或最终 whole-branch review 时统一决定是否修。

## 关键约定(勿忘)

- **LLM 凭据留空**:`engines/llm.py` 从环境变量读 `NUMPILOT_LLM_{BASE_URL,AUTH_TOKEN,MODEL}`,留空即自动回退 AST 引擎。用户自行填,不硬编码。
- 默认路径零外部依赖、零 key:`python -m numpilot migrate examples/mlp_forward.py` 必须能离线跑通。
- 执行后端默认 `shim`(numpy 支撑),真实 NPU 通过 `NUMPILOT_BACKEND=npu`。
- 文档/注释中文,代码/路径英文。
- 完成全部任务后:whole-branch 最终审查 → `superpowers:finishing-a-development-branch` 合并到 master。
