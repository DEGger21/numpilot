# NumPilot 架构与信任边界

NumPilot 将可静态分析的普通 NumPy 单文件迁移为 AsNumpy 候选，并用运行时身份、来源回执、
哈希、数值验证和可选性能门决定候选能否进入后续阶段。模型或交互式 AI 负责提出候选，
确定性检查负责准入；任何证据缺失、身份漂移或语义不明都会失败关闭。

完整架构图：[中文 PNG](assets/architecture/numpilot-architecture-zh.png) ·
[English PNG](assets/architecture/numpilot-architecture-en.png)。

## 1. 支持范围

普通入口是包含顶层 `import numpy as np` 的、可静态分析且自包含的单个 Python 文件。
输入文件不需要 `build_inputs()` 或 `forward()`。分析器会检查整个文件的 NumPy 调用面；
别名重绑定、动态调用、星号实参、未建模的 ndarray method、Python 数组运算符或其他无法证明
安全的语义会直接阻断。

普通分析和迁移阶段不会 import 或执行用户源码。只有用户显式运行 `verify-script`、
`benchmark-script` 或其他执行命令时，源码与候选才会以当前用户权限在独立 Python 进程中执行。
因此，执行命令只适用于受信代码；进程隔离和 `python -I` 不是操作系统沙箱，不可信输入应放入
独立容器或虚拟机。

历史 `build_inputs + forward` 合同仅通过显式 `--mode contract` 提供。它是独立兼容通道，
不能替代普通文件的能力、来源或验证证据。`--mode contract` 会对 `build_inputs` 的叶子做
严格静态输入分类：object、字符串、日期时间、timedelta 和结构化 dtype 在分析阶段即失败
关闭，不会进入模型生成或执行。

## 2. 两种交互路线

两条路线共享同一个 CLI 执行面和 run 目录契约：正式阶段命令为 `analyze`、`migrate`、`verify`、`replay`。Terminal 和 Agent Skill 只记录不同的 `entry_lane`/`model_source`，不会产生两套迁移、验证或性能逻辑；`verify-script` 等旧名称仅作为兼容别名保留。

| 路线 | 候选由谁提出 | `llm.json` 边界 | 可进入普通迁移报告的条件 |
|---|---|---|---|
| Terminal workflow | 用户手工编辑，或明确授权的 configured model | 仅 configured-model 子运行可读取；调用前校验配置与模型身份 | configured response 或已登记 adapter 通过来源、AST 与哈希门 |
| Agent Skill workflow | IDE/侧边栏中正在运行的交互式 AI | 不得读取、搜索、探测或调用 | 必须有已登记 adapter 将 agent handoff 转成受支持的 hash-bound 报告 |

Hybrid workflow 可以在两条路线间显式交接，但必须为 configured-model 子运行重新授权、使用新的
run 目录，并分别保留 agent 与远端模型回执。没有已登记 adapter 的手工或 Agent Skill 候选只保留
为 `agent_authored_draft`/`draft-stopped`；不得伪造 configured-model 回执，也不得运行依赖正式迁移
报告的验证或性能命令。

Configured model 会收到完整输入源码，并可能产生费用。缺口生成会发送有界 GapSpec、静态语义
函数、选定的 CANNBot Skill 内容以及适用的诊断；adapter 生成只发送目标所需的源码、头文件、
overlay 和哈希清单。CANNBot 提供工作流上下文，不是代码生成模型。

## 3. 六角色主链（与架构图一致）

架构图把迁移分成 Understand → Migrate → Validate 三段。六个模型角色只出现在这些位置：
编排器负责传递上下文与汇总报告，确定性验证（API 正确性、全程序验证、ValidationReceipt）
是主干，不占角色名额。实现与图中每个框一一对应：

| 阶段 | 角色 / 主干 | 输入 → 输出 | 实现 |
|---|---|---|---|
| Understand | ① Analyzer | 源码 → AST · Program IR/DAG · capability 判定 | `agents/analysis.py`、`program_ir.py` |
| Understand | ② Planner | 分析结果 + 缺口清单 → Migration Plan（Direct / Fill API Gaps） | `agents/planner.py` |
| Migrate | ③ API Gap Filler | 缺口清单 → GapTask（official alias / 确定性下降 / 需生成） | `agents/gap_filler.py` |
| Migrate | API Correctness Check | 缺口产物 → 接受 / 拒绝 | `capabilities.py`、`gaps/qualification.py`（主干，不是角色） |
| Migrate | ④ Code Migrator | 计划 + 源码 → 迁移后 AsNumpy 代码 | `agents/migration.py`、`engines/` |
| Validate | 全程序验证 | 原程序 vs 迁移产物 → 三层数值比较 | `agents/verification.py`（主干，不是角色） |
| Validate | ⑤ Diagnostics Agent | 失败报告 → 分类 + 回退角色 | `agents/diagnostics.py` |
| Validate | ⑥ Migration Reporter | 验证证据 + 中间产物 + 输出程序 → 报告 | `report.py`、`orchestrator.py` |

实现上的三条硬约束与图一致：

1. Planner 在任何改写前给出路线；缺口不可路由时在 Migrate 阶段失败关闭，不绕过缺口改写。
2. 全程序验证失败后只回退到 Diagnostics 指定的角色；`repair` 仅对声明支持修复的引擎开放，
   最多 3 次（`MAX_REPAIR_ITERATIONS`）。
3. Reporter 引用验证证据，不重新判定通过与否。

### 3.1 翻译主线：整程序准备、任务书与模型代码格式

“AsNumpy 接口已全覆盖”的程序仍可能因为结构、别名拼写或模型格式不清楚而迁移失败。
这条翻译主线把确定性事实集中到 Planner，并把模型需要遵守的 AsNumpy 形状显式交给它：

```text
原始源码（数值 oracle，永不替换）
  └─ prepare_program → 分析/迁移表面
       └─ Analyzer → Planner → TRANSLATION_TASKBOOK.json/.md
            ├─ route=code_migrator（别名、Python 算子、静态结构、标量绑定）→ direct
            └─ 真正的 gap_required / consent_required → fill_api_gaps
```

| 组件 | 责任 | 边界 |
|---|---|---|
| `program_preparation.py` | helper 内联、静态 for/while/if 展开、解构、文档化等价拼写、Python dunder 降级与标量广播；直线化 SSA 重排单独标记 | 只改分析/迁移表面；原源码继续做数值 oracle；`operator_spec` 领域算子保持调用边界；`applied`/`source_changed`/`normalized` 必须一致 |
| `translation_taskbook.py` | 逐调用点的源/目标 API、常见 Python 算子的确定性目标、静态参数、状态、已知/未知事实、准备前后 SHA-256、验收边界 | 内容寻址、可验证、可渲染；不是能力或性能证书 |
| `asnumpy_guide.py` | 合法 `forward` 形状、`ap.*` 映射、标量 materialization、排序 `stable=True`、禁止项 | 随配置模型 system prompt 发送；Planner 任务书中的逐调用点事实优先 |
| `engines/llm.py` | `rewrite` 与 `repair` 都接收 taskbook；repair 另带确定性诊断和上一版候选 | 模型输出仍须通过表面审计；生成文本成功不等于迁移成功 |

准备记录区分语义 rewrite 与表面直线化：只做 SSA 重排时报告
`source_changed=true` / `normalized=true`，不会出现 `applied=false` 而分析源码已经
变化的矛盾记录。

Planner 只把 `code_migrator` 任务留在 direct 路线；只有真正的 `gap_required` /
`consent_required` 任务才把路线切到 `fill_api_gaps`。API Gap Filler 继续消费任务书里的
`gap_tasks`，Code Migrator 消费 `callsites`，因此“翻译”和“补缺口”共享同一份事实，
但不会互相冒充。全程序表面审计和数值对拍仍是独立的确定性门禁。

## 4. 普通文件迁移流程

```text
source
  -> 静态扫描与调用点清单
  -> runtime-bound capability 判定
  -> 五态决策
  -> 已选路线提出候选
  -> 来源 / AST / 注释 / hash 准入
  -> migrated.py + write-once migration report
  -> 可选 verify-script
  -> 可选 benchmark-script
```

静态扫描只陈述调用表面；capability 判定还要绑定实际 AsNumpy、Python wrapper/native 文件、
CANN、SoC、版本和哈希。聚合结果只有以下五种：

| 结果 | 含义 | 行为 |
|---|---|---|
| `DIRECT` | 所有调用均落在当前 runtime 可绑定的 strict 或 scope-qualified 范围 | 按已选路线生成或接收候选 |
| `FALLBACK_CONSENT` | 存在已审计的 CPU、conditional CPU 或 host-assisted 路径 | 等待用户明确接受或拒绝 |
| `GAP_REQUIRED` | API 确实缺失，或用户拒绝已审计 fallback | 生成与源码、调用点、CANN、SoC 绑定的 GapSpec |
| `CAPABILITY_AUDIT_REQUIRED` | API 可调用，但来源未知、身份漂移或资格证据不足 | 先审计/资格化，不生成同名替代算子 |
| `BLOCKED` | 动态或未建模语义无法安全处理 | 输出稳定原因并停止 |

Configured-model 产物必须绑定 configured/requested/response model identity、请求与响应哈希、
远端回执、原始源码和候选哈希。模型返回完整文件或受限 exact edit；本地 AST/机械重建只承担
准入和等价性检查，不能作为远端模型失败时的隐式生成器。

每次运行使用独立、write-once 目录。`migrated.py`、GapSpec、迁移报告、验证报告和性能报告
不得从其他 run 拼接；每个阶段都重新检查父级产物及其 SHA-256。

## 5. 验证与性能门

`verify-script` 重新绑定 source、migration report 和 migrated artifact，在两个独立解释器进程中
执行原文件与候选，并通过有界、无 pickle 的 named-output 合同比较 numeric/bool 输出。它记录
解释器、worker、NumPy/AsNumpy 模块身份和数值结果，但固定不证明 OS sandbox、strict NPU、
无 fallback 或性能：

```text
strict_npu_verified=false
performance_valid=false
os_sandbox=false
```

`benchmark-script` 是独立的 strict-NPU resident 性能门，只接受来源完整的 strict artifact 和
带显式 resident marker 的直线计算区。它验证 source/report/artifact 哈希、分段 AST、独立单核
进程、CPU BLAS 实际线程、runtime profile、AsNumpy 调用次数、NPU 同步、fallback guard 和逐样本
数值。输入准备、steady device-resident 计算和最终 D2H 分别报告；任一门失败时
`performance_valid=false` 且不发布 speedup。

详细合同见[普通文件数值验证](script-verification.md)和
[strict-NPU resident benchmark](script-benchmark.md)。

## 6. 缺口与算子边界

缺口处理固定为 official-first：

```text
GapSpec
  -> 已安装 CANN ACLNN 只读解析
  -> 与目标 CANN 绑定的 pinned cann/ops-nn 只读解析
  -> 仅在官方两层均为可绑定的 clean no_match 时，
     才允许 CANNBot Skill + 已授权边界模型生成自定义算子草稿
```

解析器不 import CANN、不加载共享库、不切换 ops-nn 工作树、不联网、不构建，也不写系统 OPP。
名称、header、符号或源码目录命中只是候选发现，不等于语义合格、已接入或已激活。installed ACLNN
通用路径最多物化为 `CANDIDATE_READY`；已审计的 MatMul existing-module 路线可以生成并绑定
`ADAPTER_CANDIDATE`。pinned ops-nn 只发布源码发现证据。

自定义生成只接受显式 `OperatorSpec` 和获准的 `custom_generation_candidate` 路由。生成结果始终
从 `generated_unverified` 开始；它不是已编译、已测试、已接入或已激活的能力。通用的
`integrate -> qualify -> activate -> rescan -> final-migrate -> verify` 自动编排不在发布接口内，
因此状态控制面会在已有回执支持的边界停止。

注册的标量 `add_one` 组合路线可尝试把 `numpy.add(array, 1)` 解析为 run-scoped
`asnumpy.ones_like -> asnumpy.add`，并要求真实 NPU 子进程资格回执。只有该回执和 runtime/profile
绑定全部通过时才能继续；接口不可用、超时、fallback、来源未知或任一门失败都会停止迁移。
并行生成的 Ascend C 草稿仍保持 `generated_unverified`，不会因组合路线存在而被编译或激活。

状态名表示已经取得的事实，而不是意图：

```text
CREATED -> ANALYZED -> GAP_PREPARED -> RESOLVED
  -> CANDIDATE_READY -> ADAPTER_CANDIDATE -> INTEGRATED_CANDIDATE
  -> QUALIFIED -> ACTIVATED -> RESCANNED -> FINAL_MIGRATED -> VERIFIED
```

候选、adapter、资格化、激活和最终迁移必须具有各自的 typed receipt。授权 profile 在首次创建
recovery run 时冻结；授权某类副作用不代表该副作用已经发生。完整约束见
[Gap 恢复状态合同](gap-recovery-state.md)和
[自定义算子流程图](numpilot-asnumpy-custom-operator-flow.png)。

## 7. 安装与组件信任

`bootstrap` 无参数或 `--plan` 时只读。`--apply` 只接受以下一种明确来源：

- 同一发行物中的 hash-bound AsNumpy component；
- 用户同时提供 wheel 文件和可信 SHA-256；
- 显式选择的锁定源码构建路径。

NumPilot 不会在 pip 安装钩子、import、doctor、migrate 或缺口命令中隐式下载依赖，也不会安装
或修改 Ascend driver、firmware、CANN、OPP 或 shell 启动文件。联网安装必须由用户显式传入
`--allow-network`；CANNBot/ops-nn 的获取、许可和来源身份与 NumPilot/AsNumpy 分别核对。

Bootstrap receipt 绑定 source/component、wheel、Python/native 模块、解释器、CANN 和 runtime
身份。它用于检测本地 lineage 与漂移，不是数字签名，也不承诺抵御同一用户或 root 同时篡改
安装文件和回执。组件发布合同见
[AsNumpy 内部组件锁定](ASNUMPY-INTERNAL-COMPONENT-LOCK.zh-CN.md)。

## 8. 兼容性边界

| 层级 | 支持范围 | 可以证明 |
|---|---|---|
| Core CLI / shim | Linux，Python 3.10+，NumPy 1.26+ | 静态分析、流程检查和数值诊断；不证明 NPU |
| Bundled NPU component | Linux AArch64、CPython 3.12、CANN 9.2.0-beta.2、兼容的 driver/firmware/OPP | 安装并核对随发行物提供的 hash-bound AsNumpy wheel |
| Strict verified profile | Ascend 910B3、CANN 9.2.0-beta.2 位于 `/usr/local/Ascend/cann-9.2.0-beta.2`、CPython 3.12.14、NumPy 2.5.4、Loguru 0.7.3、随发行物提供的 AsNumpy component | 仅在每个调用自身的 dtype/rank/layout/profile 与资格回执范围内证明 strict-NPU |

不同 SoC、CANN 版本或安装位置、Python ABI、操作系统、wrapper/native 文件、driver/OPP 或组件
哈希都不自动继承 strict 结论。此类环境必须返回 `CAPABILITY_AUDIT_REQUIRED`，直到新 profile、
组件构建与资格证据经过独立审计。通过 `doctor` 或 shim smoke 只表示基础环境可用，不会扩大
strict-NPU 能力范围。CANN、driver、firmware 和 OPP 均为外部依赖，不包含在 NumPilot wheel 中。

安装和验收步骤见[安装指南](installation.md)。

## 9. 主要组件

```text
numpilot/
├── __main__.py                  CLI 与显式用户门
├── bootstrap.py                 依赖计划、锁定安装与原子回执
├── script_migration.py          普通文件扫描、五态判定与候选准入
├── script_verification.py       双进程数值验证
├── script_benchmark.py          strict-NPU resident 性能门
├── capabilities.py              runtime-bound capability/provenance
├── backends/                    shim 与真实 NPU runtime adapter
├── gaps/                        GapSpec、官方解析、草稿、adapter、资格与状态原语
├── roles.py / agents/           六角色主链（Analyzer → Planner → Gap Filler → Migrator → Diagnostics → Reporter）
├── program_ir.py                Program IR / DAG
└── engines/ / orchestrator.py   改写引擎与六角色编排（确定性验证为主干）
```

模型身份、回执和 no-llm 规则见[模型合同](model-contract.md)；Terminal、Agent Skill 与 hybrid
交接见[交互路线指南](codearts-integration.md)。
