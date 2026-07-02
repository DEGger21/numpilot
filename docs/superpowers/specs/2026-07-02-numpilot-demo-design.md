# NumPilot 最简 Demo — 设计文档(Spec)

> 课程:高级软件体系结构 · 课程作业
> 主题:面向科学智能的昇腾 NPU 代码自动迁移智能体(NumPilot)最简可运行 demo
> 底座库:自研 AsNumpy(面向昇腾 NPU 的 NumPy 兼容库)
> 日期:2026-07-02

---

## 1. 背景与定位

NumPilot 是一个把 **NumPy 科学计算程序自动迁移到昇腾 NPU** 的多智能体系统(见 `NumPilot20260615-v1.pdf`)。完整系统包含"缺口算子自动生成"等复杂环节;**本 demo 聚焦最简可运行的迁移流水线,刻意不做缺口算子补充**,用于:

1. 作为高级软件体系结构课程作业,展示清晰的多智能体架构(编排器 + 分工智能体)、可替换构件设计(策略模式 / 端口-适配器)、契约化组件边界。
2. 作为后续参赛(华为云码道 CodeArts / 揭榜挂帅 XH-202618)的技术骨架与升级基线。

**底座 AsNumpy**:通过 pybind11 封装 CANN 算子,以 `NPUArray` 对标 `numpy.ndarray`,API 名称/签名与 NumPy 对齐。迁移的本质 = `np.*` → `ap.*` 改写 + `from_numpy/to_numpy` 的 CPU↔NPU 数据搬运。

## 2. 目标与非目标

### 目标(In Scope)
- 输入一段只使用 **AsNumpy 已覆盖算子** 的 NumPy 程序,自动产出 AsNumpy(NPU)版本。
- 多智能体流水线:分析 → 决策 → 迁移 → 验证,由编排器统筹。
- 数值一致性校验(按 dtype 定相对误差容差)+ 加速比评测 + 迁移报告。
- 迁移引擎与执行后端均**可替换**:AST 改写 ↔ LLM;numpy-shim ↔ 真实 NPU。
- 一定泛化能力:基于覆盖集自动改写任意受支持算子,不依赖 hardcode 映射表。
- 零外部依赖即可离线一键跑通(默认 AST 引擎 + numpy shim 后端)。

### 非目标(Out of Scope,本 demo 明确不做)
- **缺口算子的 Ascend C 自动生成 / 补充**。分析阶段发现不在覆盖集的算子,仅**标记并在报告中列出**,随后终止迁移,不尝试补齐。
- 真实 LLM 的联网调用(仅留 provider 无关的可插拔接口 + stub)。
- 复杂数据流图 / 精细 shape 推断 / AOT-JIT 双模。demo 用轻量静态分析即可。

## 3. AsNumpy 覆盖集(权威依据)

覆盖集从真实 `asnumpy/__init__.py` 的 `_LAZY_MAPPING` 键派生一份**快照**,存于 `numpilot/coverage.py`(含来源注释与再生成方法)。它回答"AsNumpy 支持哪些算子"。示例(节选):
`add, subtract, multiply, divide, matmul, dot, exp, sqrt, square, sum, mean, max, min, maximum, relu, gelu, softmax, tanh, sort, zeros, ones, full, linspace, ...`

判定规则:源码中出现的 `np.<name>` / `numpy.<name>`,若 `<name>` ∈ 覆盖集 → 可迁移;否则 → `UNSUPPORTED`(缺口,本 demo 终止)。

## 4. 架构总览

```
                         ┌─────────────────────────────────────────┐
   source.py ──────────► │            Orchestrator 编排智能体          │
                         │  驱动流水线 · 传递上下文 · 汇总报告            │
                         └───┬──────────┬──────────┬──────────┬──────┘
                             ▼          ▼          ▼          ▼
                        [Analysis]  [Decision]  [Migration] [Verification]
                          分析        决策         迁移         验证
                             │          │          │            │
                     AnalysisResult OffloadPlan MigrationResult VerificationReport
                             │          │          │            │
                             └──────────┴────►  contracts.py  ◄──┘
                                        (dataclass 契约集中定义)

  可替换构件:
    MigrationEngine(策略): AstRewriteEngine(默认) | LlmEngine(stub, provider 无关)
    ExecutionBackend(端口): ShimBackend(numpy 支撑, 默认) | NpuBackend(真实 asnumpy)
```

### 4.1 组件职责与依赖

| 组件 | 职责 | 输入 → 输出 | 依赖 |
|------|------|-----------|------|
| Orchestrator | 编排流水线、状态传递、汇总 `report.md`/`report.json` | 文件路径 → 报告 | 4 个 Agent |
| Analysis Agent | `ast` 解析,提取 `np.*` 算子清单 + 变量级简单数据流,比对覆盖集 | source.py → `AnalysisResult` | `coverage` |
| Decision Agent | 规则版成本模型:按张量规模阈值给每算子定 CPU/NPU | `AnalysisResult` → `OffloadPlan` | 阈值配置 |
| Migration Agent | 按 `MigrationEngine` 策略改写 `np.*`→`ap.*`、插桩数据搬运 | `OffloadPlan` + source → `MigrationResult` | `MigrationEngine` |
| Verification Agent | 跑原程序 vs 迁移程序,数值对比 + 测速 | `MigrationResult` → `VerificationReport` | `ExecutionBackend` |

每个 Agent 只认输入/输出 dataclass,内部实现可独立替换与单测——组件边界即契约边界。

### 4.2 可替换构件

- **MigrationEngine(策略模式)** — 接口 `rewrite(source: str, plan: OffloadPlan) -> str`。
  - `AstRewriteEngine`(默认):基于 Python `ast` + 覆盖集自动改写,确定性、可离线。泛化来源。
  - `LlmEngine`(stub):provider 无关,读 `NUMPILOT_LLM_{PROVIDER,BASE_URL,API_KEY,MODEL}`;OpenAI-兼容协议骨架(盘古 / GLM / DeepSeek / Claude 通用);未配置或不可用时**自动回退** AST 引擎。
- **ExecutionBackend(端口-适配器)** — 让验证阶段在不同后端运行迁移代码。
  - `ShimBackend`(默认):numpy 支撑的假 `asnumpy` 模块(`ap.add` 等转发 numpy,`ndarray.from_numpy/.to_numpy` 包装),随处可跑,数值恒等,便于离线复现与 CI。
  - `NpuBackend`:导入真实 `asnumpy`,在 NPU 上执行。通过 `NUMPILOT_BACKEND=shim|npu` 切换。

## 5. 数据流与契约(`contracts.py`)

```python
@dataclass
class OpUse:            # 一次算子调用点
    name: str          # numpy 函数名, e.g. "matmul"
    lineno: int
    supported: bool    # 是否在覆盖集
    est_elements: int | None  # 估计张量规模(用于决策), 未知为 None

@dataclass
class AnalysisResult:
    source_path: str
    op_uses: list[OpUse]
    unsupported: list[str]     # 缺口算子名(去重)
    migratable: bool           # unsupported 为空

@dataclass
class OffloadPlan:
    placement: dict[str, str]  # op name -> "NPU" | "CPU"
    threshold_elements: int

@dataclass
class MigrationResult:
    engine: str                # "ast" | "llm"
    migrated_source: str

@dataclass
class VerificationReport:
    numerically_consistent: bool
    max_rel_diff: float
    tol: float
    speedup: float | None
    per_op: list[dict]
    status: str                # "pass" | "fail" | "unsupported"
```

## 6. 错误处理

| 场景 | 处理 |
|------|------|
| 缺口算子(不在覆盖集) | Analysis 标 `UNSUPPORTED` → 报告列出 → 终止迁移,`status=unsupported` |
| 迁移后语法/运行错误 | Verification 捕获异常 → `status=fail`,报告附 traceback 摘要 |
| 数值不一致(相对误差 > 容差) | `numerically_consistent=false`,`status=fail` |
| LLM 未配置/不可用 | `LlmEngine` 自动回退 `AstRewriteEngine`,报告注明 |

容差按 dtype:fp32 → `1e-4`,fp16 → `1e-2`,整型 → 精确相等。

## 7. 测试策略(TDD)

- 单元测:`coverage`(判定)、每个 Agent、每个 Engine、每个 Backend 独立测。
- 端到端:`migrate examples/mlp_forward.py` → 断言 `status=pass` 且数值一致、报告字段完整。
- 负例:含缺口算子的样例 → 断言 `status=unsupported` 且列出正确算子名。

## 8. 目录结构(落盘 `/home/lixing/numpilot`)

```
numpilot/
├── README.md
├── pyproject.toml
├── numpilot/
│   ├── __init__.py
│   ├── __main__.py             # CLI: python -m numpilot migrate <file> [--backend shim|npu] [--engine ast|llm]
│   ├── orchestrator.py
│   ├── contracts.py
│   ├── coverage.py             # AsNumpy 覆盖集快照 + 再生成说明
│   ├── agents/
│   │   ├── analysis.py
│   │   ├── decision.py
│   │   ├── migration.py
│   │   └── verification.py
│   ├── engines/
│   │   ├── base.py             # MigrationEngine 接口
│   │   ├── ast_rewrite.py
│   │   └── llm.py              # provider 无关 stub + OpenAI 兼容骨架
│   └── backends/
│       ├── base.py             # ExecutionBackend 端口
│       ├── shim.py             # numpy 支撑的假 asnumpy
│       └── npu.py              # 真实 asnumpy 适配器
├── examples/
│   ├── mlp_forward.py          # 待迁移的 numpy 小型 MLP 前向(仅用覆盖集算子)
│   └── unsupported_demo.py     # 负例:含缺口算子
├── tests/
│   ├── test_coverage.py
│   ├── test_analysis.py
│   ├── test_decision.py
│   ├── test_migration_ast.py
│   ├── test_backends.py
│   └── test_e2e.py
└── docs/
    ├── architecture.md         # 4+1 视图 + C4(Context/Container/Component)
    ├── quality-scenarios.md    # 质量属性场景
    └── adr/
        ├── ADR-001-strategy-migration-engine.md
        ├── ADR-002-ports-adapters-backend.md
        ├── ADR-003-contract-first-agents.md
        └── ADR-004-scope-no-gap-op-gen.md
```

## 9. 课程作业交付物(高级软件体系结构)

- `docs/architecture.md`:**4+1 视图**(逻辑/进程/开发/物理/场景)+ **C4** 三层图(文本/mermaid)。
- `docs/quality-scenarios.md`:质量属性场景 —— 可修改性(换 LLM 引擎)、可移植性(shim↔NPU)、正确性(数值校验)、可测试性(契约边界)。
- `docs/adr/*`:关键架构决策记录(策略引擎、端口后端、契约先行、范围裁剪)。
- 可运行代码 + 一条命令跑通的 demo + 自动生成的迁移报告。

## 10. 架构决策摘要

| ADR | 决策 | 理由 |
|-----|------|------|
| 001 | 迁移引擎用**策略模式** | LLM 可插拔,demo 离线稳定,比赛可升级 |
| 002 | 执行后端用**端口-适配器** | shim 随处可跑,真实 NPU 无侵入替换 |
| 003 | Agent 间**契约先行**(dataclass) | 组件可独立理解/替换/单测,边界清晰 |
| 004 | **裁掉缺口算子生成** | 控制 demo 复杂度;缺口只标记不补齐 |

## 11. 非功能约束

- 语言:文档/注释中文为主,代码/路径英文。
- 运行:默认路径零外部依赖(仅 numpy),`python -m numpilot migrate examples/mlp_forward.py` 一键跑通。
- 决策阈值:默认 `1e6` 元素上 NPU,可通过参数/环境变量配置。
