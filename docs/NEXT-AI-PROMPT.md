# NumPilot 下一次 AI 接入 Prompt

> **已过时（2026-08-21）**：不要照搬下方 `/home/lixing` 路径、106 tests 或旧 matmul
> 结论。当前工作树位于 `/home/ma-user/work/numpilot`，全量回归为 `355 passed`；权威续接
> 入口是 `docs/P0-LLM-CANNBOT-RUN-20260821.md`、
> `docs/evidence/poisson-ritz-p0-npu-20260821.md` 和
> `docs/COMPETITION-GAP-REVIEW-20260821.md`。当前下一优先级是 profiler、放大规模和重复
> 样本，CodeArts 按用户决定后置。

> 用法：将本文从“你现在接手 NumPilot 项目”开始完整复制给下一次接入的 AI。

你现在接手 NumPilot 项目，请基于现有进度继续工作，不要从零开始，也不要覆盖当前未提交修改。

## 项目绝对路径

- NumPilot 主仓库（唯一权威工作目录）：`/home/lixing/numpilot`
- AI 续接手册（必须首先完整阅读）：`/home/lixing/numpilot/docs/AI-HANDOFF.md`
- 技术评审与实施复盘：`/home/lixing/numpilot/docs/numpilot-technical-review.md`
- 当前架构：`/home/lixing/numpilot/docs/architecture.md`
- CodeArts 集成方案：`/home/lixing/numpilot/docs/codearts-integration.md`
- 裸金属 NPU 验证记录：`/home/lixing/numpilot/docs/evidence/npu-smoke-20260718.md`
- 项目 README：`/home/lixing/numpilot/README.md`
- AsNumpy 原工程：`/home/lixing/asnumpy`
- AsNumpy 自定义算子接入 demo：`/home/lixing/asnumpyoplogin`
- 比赛题目 PDF：`/home/lixing/挑战杯/XH-202618_基于华为云码道（CodeArts）代码智能体解决复杂软件工程问题.pdf`
- NumPilot 原始设计文档：`/home/lixing/挑战杯/NumPilot20260615-v1.pdf`

注意：`/home/lixing/asnumpyoplogin/.numpilot_stage` 只是上一轮受限环境使用的临时副本，后续一律以 `/home/lixing/numpilot` 为准。

## 接手后的第一步

先完整阅读上述续接手册、技术评审、架构和 CodeArts 文档，然后执行：

```bash
cd /home/lixing/numpilot
git status --short
git diff --check
/home/lixing/np310/bin/python -m pytest -q tests
```

当前预期结果为：`106 passed`。

当前仓库处于未提交状态，这些修改是用户授权并且需要保留的。禁止执行：

```bash
git reset --hard
git checkout --
git clean -fd
```

不要覆盖或撤销不属于你的修改，也不要自行提交 commit，除非用户明确要求。

## 项目目标

NumPilot 的目标是利用 CodeArts 多智能体，把科学计算 NumPy 网络自动迁移到 AsNumpy/Ascend NPU；如果发现能力缺口，则依次尝试：

1. 已有且经过严格审计的 AsNumpy capability；
2. ACLNN；
3. 组合已有 NPU 原语；
4. 对真实热点生成 Ascend C 自定义算子；
5. 无法满足时显式标记 CPU fallback 或 unsupported。

CodeArts AgentTeam、项目级 Agent 和 Skill 负责协作控制面、任务拆解和人工确认；NumPilot CLI、contracts、capability manifest、验证器和 benchmark 负责确定性的执行与证据门禁。

不能把 API 名称存在、Python 函数被调用、编译成功或 shim 测试通过，直接当作真实 NPU kernel 或合格性能证据。

## 当前已经完成

现有实现已经包括：

- 顶层同步 `forward()` 的完整 NPU 岛迁移；
- 严格 AST 分析和迁移边界；
- `expected_rewrites` 与实际 `rewrites` 精确映射校验；
- 三层语义验证；
- runtime `ap.*` callsite 覆盖证据；
- capability/provenance manifest；
- CPU fallback/unknown fail-closed；
- 独立 CPU/NPU benchmark 进程；
- benchmark hash、线程数、计时范围和性能资格门禁；
- CodeArts Agent/Skill 配置；
- Python 3.9 `ast.Match` 兼容；
- shim/NPU 报告真实性修复；
- 相关单元、反例和裸金属 NPU 测试。

现有 NPU 证据：

- strict elementwise 正例：`/tmp/numpilot-final-bench-20260718`
- matmul fallback 拒绝负例：`/tmp/numpilot-final-matmul-rejection-20260718`

`/tmp` 可能被系统清理，不能把它当作永久制品库。

当前 elementwise 小算例中 NPU/CPU 约为 `0.035723×`，即 CPU 约快 `27.99×`。这是证据链验证结果，不得宣传为 NPU 加速。

## 运行环境

本机已有裸金属 Ascend NPU/CANN 环境，不要使用 Docker。

Python：`/home/lixing/np310/bin/python`

NPU 环境初始化：

```bash
source /usr/local/Ascend/cann/bin/setenv.bash
export ASCEND_SOC_VERSION=dav-2201
export LD_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"
```

当前已验证环境包括：

- 8 × Ascend 910；
- CANN 8.5.0；
- dav-2201；
- AsNumpy `0.3.1.dev4+g1929a3ba4`。

## 下一阶段优先任务

优先实现“OS 级隔离执行器”，不要同时扩张所有方向。

目标是把用户源码加载、`build_inputs()` 和 `forward()` 从 NumPilot 主进程迁移到受限子进程。

建议新增 `SandboxRunner` 或等价组件，并满足：

1. 原始程序和迁移产物在不同受限进程执行；
2. 支持超时、异常、进程退出和内存限制；
3. 支持嵌套 ndarray/标量/容器结果的结构化序列化；
4. 不再污染主编排进程的 `sys.modules`；
5. 并发验证不会串用 backend；
6. 无限循环、恶意文件写入、异常退出和超内存具有确定性失败测试；
7. 现有 106 个测试继续通过；
8. 裸金属 elementwise 正例继续通过；
9. matmul fallback 继续在执行前拒绝性能发布。

如果隔离执行器不适合作为当次任务，则第二优先级是 profiler/kernel 证据 schema；第三优先级是选择一个真实科研网络热点缺口算子，完成 GapSpec、实现、UT、裸金属 ST、精度、性能、安装、artifact hash 和回滚的完整闭环。

## CodeArts 注意事项

已有配置位于：

- `/home/lixing/numpilot/.codeartsdoer/agents`
- `/home/lixing/numpilot/.codeartsdoer/skills`

需要在实际比赛账号验证：

- AgentTeam 是否能够加载或组合项目级自定义 Agent；
- `ProjectSkillStatus.txt` 与 Skill 是否成功启用；
- Remote SSH 是否能连接当前裸金属 NPU 环境；
- Sandbox、Hook、MCP 和权限配置；
- 任务图、对话、代码 diff、测试失败修复和流水线证据是否能完整留痕。

官方文档没有明确保证 AgentTeam 会自动把项目级 Agent 装配成固定团队，不得在未经账号实测时声称已经完成。

## 真实性原则

以下原则必须继续保持：

- shim 只证明语义，不证明 NPU；
- CPU fallback 或 unknown 不得计算或发布 speedup；
- `expected_rewrites` 必须与实际 `rewrites` 完全一致；
- `performance_valid=true` 才能引用性能比值；
- NPU/CPU 小于 1 时必须明确写“CPU 更快”；
- Python API trace 不等于 profiler/kernel trace；
- 当前 matmul 是 CPU fallback；
- CodeArts 配置文件存在不等于比赛账号已经成功加载；
- 所有结论必须能追溯到 JSON、测试或 profiler 原始证据。

## 开始工作

请依次完成：

1. 阅读所有交接材料；
2. 检查当前 `git diff`，确认已有修改；
3. 用几段话说明你理解的项目状态；
4. 给出本次具体修改计划；
5. 随后直接实施、测试并在裸金属 NPU 上验证；
6. 完成后更新 `/home/lixing/numpilot/docs/AI-HANDOFF.md`，写明做了什么、测试结果、新证据路径、剩余问题和下一步。
