# NumPilot(最简可运行 demo)

**面向科学智能的昇腾 NPU 代码自动迁移智能体** —— 把只使用 [AsNumpy](https://gitcode.com/) 覆盖集算子的 NumPy 程序,自动迁移为昇腾 NPU 版本(`np.* → ap.*`),并做数值一致性校验与加速比评测。

> 本仓库是 NumPilot 完整系统(见 `NumPilot20260615-v1.pdf`)的**最简可运行子集**,用作*高级软件体系结构*课程作业与后续参赛的技术骨架。完整系统含"缺口算子 Ascend C 自动生成"等环节;**本 demo 刻意不做缺口算子补充**——发现不在覆盖集的算子,仅标记并终止迁移。

---

## 一键运行(零外部依赖,离线可跑)

默认使用 **AST 改写引擎 + numpy-shim 后端**,无需 asnumpy / NPU:

```bash
# 正例:MLP 前向 → 迁移 + 验证
python -m numpilot migrate examples/mlp_forward.py --out /tmp/numpilot_out
cat /tmp/numpilot_out/report.md

# 负例:含缺口算子 np.fft.fft → 标记 unsupported 并终止
python -m numpilot migrate examples/unsupported_demo.py
```

正例预期输出:`状态: pass`、`数值一致: True`、`report.md` 含逐算子卸载方案。
负例预期输出:`状态: unsupported`、`缺口算子: fft`(退出码 1)。

测试:

```bash
python -m pytest -q          # 23 passed
```

---

## 架构一图

```
   source.py ─► ┌──────────── Orchestrator 编排智能体 ────────────┐
                │       驱动流水线 · 传递上下文 · 汇总报告           │
                └──┬─────────┬──────────┬───────────┬────────────┘
                   ▼         ▼          ▼           ▼
              [Analysis] [Decision] [Migration] [Verification]
                 分析       决策        迁移          验证
                   │         │          │            │
           AnalysisResult OffloadPlan MigrationResult VerificationReport
                   └─────────┴───► contracts.py ◄─────┘   (dataclass 契约)

  可替换构件:
    MigrationEngine(策略):  AstRewriteEngine(默认) | LlmEngine(provider 无关, 自动回退)
    ExecutionBackend(端口):  ShimBackend(numpy 支撑, 默认) | NpuBackend(真实 asnumpy/NPU)
```

流水线:**分析 → 决策 → 迁移 → 验证**,由编排器统筹。每个智能体只认输入/输出 dataclass 契约(`numpilot/contracts.py`),内部实现可独立替换与单测。详见 [`docs/architecture.md`](docs/architecture.md)。

---

## 后端 / 引擎切换

| 维度 | 选项 | 说明 |
|------|------|------|
| 后端 `--backend` | `shim`(默认) / `npu` | shim 用 numpy 假模块,处处可跑;npu 导入真实 `asnumpy`。也可用环境变量 `NUMPILOT_BACKEND` |
| 引擎 `--engine` | `ast`(默认) / `llm` | ast 基于 Python `ast` + 覆盖集确定性改写;llm 走 OpenAI 兼容协议,未配置或失败时**自动回退 ast** |
| 阈值 `--threshold` | 整数(默认 1_000_000) | 决策智能体按张量元素数阈值决定算子 CPU/NPU 卸载 |

```bash
python -m numpilot migrate examples/mlp_forward.py --backend shim --engine ast --threshold 500000 --out out/
```

### 启用真·LLM 引擎(`--engine llm`)

`LlmEngine` 是 provider 无关的 OpenAI 兼容客户端。启用需 **3 个必填环境变量**(第 4 个可选):

| 环境变量 | 必填 | 含义 |
|----------|------|------|
| `NUMPILOT_LLM_BASE_URL` | ✅ | OpenAI 兼容 endpoint,例:`https://.../v1` |
| `NUMPILOT_LLM_AUTH_TOKEN` | ✅ | 鉴权 token,以 `Authorization: Bearer <token>` 发送 |
| `NUMPILOT_LLM_MODEL` | ✅ | 模型名 |
| `NUMPILOT_LLM_PROVIDER` | ⬜ | 仅作标签,例:`pangu` \| `glm` \| `openai` \| `deepseek` |

```bash
# 示例:盘古 / GLM(任意 OpenAI 兼容网关)
export NUMPILOT_LLM_BASE_URL="https://your-gateway/v1"
export NUMPILOT_LLM_AUTH_TOKEN="sk-..."
export NUMPILOT_LLM_MODEL="glm-4"          # 或 pangu-* / deepseek-chat
export NUMPILOT_LLM_PROVIDER="glm"          # 可选
python -m numpilot migrate examples/mlp_forward.py --engine llm
```

> ⚠️ **凭据仅经环境变量注入,切勿写入代码或提交入库。** 三个必填变量缺任一 → `LlmEngine` 判为不可用并**自动回退 AST 引擎**,流水线仍可跑通。

---

## 目录导航

```
numpilot/
├── contracts.py         # dataclass 契约:AnalysisResult / OffloadPlan / MigrationResult / VerificationReport
├── coverage.py          # AsNumpy 覆盖集快照(来自 asnumpy _LAZY_MAPPING)
├── orchestrator.py      # 编排智能体 + report.md/report.json 落盘
├── __main__.py          # CLI:python -m numpilot migrate ...
├── agents/              # analysis / decision / migration / verification
├── engines/             # 迁移引擎策略:base / ast_rewrite(默认) / llm(stub, 可回退)
└── backends/            # 执行后端端口:base / shim(默认) / npu
examples/
├── mlp_forward.py       # 正例:MLP 前向(纯覆盖集算子)
└── unsupported_demo.py  # 负例:np.fft.fft 缺口算子
docs/
├── architecture.md      # 4+1 视图 + C4(Context/Container/Component)
├── quality-scenarios.md # 质量属性场景
└── adr/                 # ADR-001..004 架构决策记录
tests/                   # 单元 + 端到端
```

---

## 与 NumPilot PDF 的关系

`NumPilot20260615-v1.pdf` 描述的是含缺口算子生成、复杂数据流图、AOT/JIT 双模的完整愿景。本 demo 取其**迁移流水线主干**(分析→决策→迁移→验证)做最简可运行落地,聚焦**可清晰讲清的软件体系结构**:多智能体分工、策略模式、端口-适配器、契约先行。缺口算子生成被明确裁剪(见 [ADR-004](docs/adr/ADR-004-scope-no-gap-op-gen.md)),留待完整系统。
