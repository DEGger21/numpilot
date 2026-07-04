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
python -m pytest -q          # 41 passed
```

---

## 交互式迁移向导(`wizard`)

面向演示的**交互式终端向导**:启动界面 → 输入待迁移程序路径 → 选引擎 → 带进度条与实时解释跑迁移 → 输出迁移反馈(算子替换、**接入多少自定义 NPU 算子**、数值校验、**整网性能对比**)。

```bash
python -m numpilot wizard                    # 交互:提示输入路径 + 选引擎
python -m numpilot wizard <file> --engine ast --backend npu   # 非交互(带参跳过提问)
```

向导依次跑 5 步(rich 进度条 + 每步一句解释):`分析算子 → 决策卸载 → 迁移改写 → 真机校验+NPU测速 → numpy 单核基线`,最后给出反馈表 + 性能面板。性能用**公平口径**:子进程 `bench-cpu` 把 numpy 钉 1 核,向导进程内测 asnumpy@NPU,`加速比 = numpy@1核 / asnumpy@NPU`。反馈里的"自定义 NPU 算子"指迁移后命中 asnumpy 自定义算子(`relu/softmax/gelu/angle`——numpy 没有、由 NPU 提供)的数量。选 `--engine llm` 走 glm-5.1 智能体,失败按 strict 终止。

**每次迁移都会落地一份带时间戳的 Markdown 报告**到 `reports/migration_<网络>_<YYYYMMDD_HHMMSS>.md`(`wizard` 与 `migrate` 都会写;`reports/` 已 gitignore)。

---

## 真机 NPU 演示(一键脚本)

前置:CANN 环境 + 真机 `asnumpy`(本机 venv `/home/lixing/np310`)。一条命令跑完整流程:

```bash
./run_demo.sh                          # 默认演示 big_gemm(4 层 2048 宽 GEMM 网络), 引擎=AST
./run_demo.sh examples/mlp_forward.py  # 换一个网络
./run_demo.sh --full                   # 额外跑"numpy 吃满全核"对照, 展示 640 核作弊效果
./run_demo.sh --llm                    # 迁移改用真·大模型智能体(glm-5.1, 读 .env.llm), 失败即终止
```

脚本依次执行 **4 步**:环境自检 → 迁移证明 → CPU 基线(numpy 限 1 核)→ NPU 对比(算加速比)。
（非本机运行改 `NUMPILOT_PY=<你的python> ./run_demo.sh`。)

### 演示用的网络

| 文件 | 网络 | 用途 |
|------|------|------|
| `examples/big_gemm.py` | **4 层全连接(堆叠 GEMM)**:每层 2048×2048 matmul + ReLU,末层不接 ReLU;类 He 初始化 | 矩阵乘密集,NPU Cube 单元发力场景 → **性能演示** |
| `examples/mlp_forward.py` | 小型 2 层 MLP 分类器:784→256(ReLU)→10→softmax,batch 64 | 规模小(输出 640 元素)→ **正确性演示**,不比性能 |

### 四个命令各司其职

| 命令 | 作用 |
|------|------|
| `numpilot doctor --backend npu` | 环境证据:Python / numpy / CANN / NPU / asnumpy 是否就位 |
| `numpilot migrate <file> --backend npu` | 迁移证明:`np.*→ap.*` 替换计数、真机 asnumpy 命中(版本+路径)、数值一致 |
| `numpilot bench-cpu <file> --out D` | ① numpy 基线,进程内**钉 1 核**,把输出+耗时存到 `D/cpu_baseline.*` |
| `numpilot bench-npu <file> --out D` | ② NPU **满线程**跑,读 ① 基线算 **加速比 = numpy@1核 / asnumpy@NPU** |

> `bench-cpu` 与 `bench-npu` 必须用**同一 `--out` 目录**:前者写基线,后者读基线。

### 为什么性能要分两步、CPU 还限 1 核

本机 numpy = **scipy-openblas**,`matmul` 默认吃满全部 **640 核**,与单卡 NPU 严重不对等(不限核时 numpy 虚快)。限 numpy 到 1 核(单计算流 vs 单卡)后,big_gemm 上 **asnumpy@NPU 反超 ~8×**。而单进程内用环境变量限核会经 OpenMP **污染 CANN**(NPU 也被拖慢 ~10×),故必须拆成**两个独立进程**各设线程。预期头条:

```
[对比] numpy@CPU(1 核) ~1500 ms  vs  asnumpy@npu ~180 ms  → 加速比 ~8x (asnumpy 更快)
```

### 迁移引擎(当前用的是哪个)

- **`--engine ast`(默认,不经任何大模型)**:基于 Python `ast` + 覆盖集把 `np.<op>` 改写成 `ap.<op>`,并识别 `np.maximum(x,0)→ap.relu(x)`。结果确定、可复现,`run_demo.sh` 默认走它。
- **`--engine llm`(真·大模型智能体,`.env.llm` 当前配 `glm-5.1`)**:把源码交给大模型做迁移。系统提示已注入 asnumpy 硬约束(二元算子两侧须 NPUArray、`maximum(x,0)→relu`、只迁 `forward`),glm-5.1 已在真机 NPU 上跑通、数值一致。**CLI 显式选 llm 为 strict 语义:模型未配置/无响应/返回非法 → 直接终止(exit 4),不回退 AST**——保证"用智能体迁移"名副其实。`./run_demo.sh --llm` 一键体验。

> LLM 引擎相关环境变量见下节;凭据写在 `.env.llm`(已被 `.gitignore` 忽略,不入库)。

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

可选环境变量:`NUMPILOT_LLM_MAX_TOKENS`(默认 8192,部分网关强制要求)、`NUMPILOT_LLM_INSECURE`(默认 `1`,跳过自签证书校验)、`NUMPILOT_LLM_TIMEOUT`(默认 120s)。

> ⚠️ **凭据仅经环境变量注入,切勿写入代码或提交入库。** `base_url` 务必带 `/v1`。
> CLI 显式 `--engine llm` 为 **strict 语义**:未配置/无响应/返回非法 → **直接终止(exit 4),不回退 AST**。仅当以库形式 `LlmEngine()`(非 strict)调用时,才会在失败时回退 AST。

---

## 目录导航

```
run_demo.sh              # 一键演示:doctor → migrate → bench-cpu → bench-npu
numpilot/
├── contracts.py         # dataclass 契约:AnalysisResult / OffloadPlan / MigrationResult / VerificationReport
├── coverage.py          # AsNumpy 覆盖集快照(来自 asnumpy _LAZY_MAPPING)
├── orchestrator.py      # 编排智能体 + report.md/report.json 落盘
├── bench.py             # 两个独立性能程序:bench-cpu(numpy 限核基线) / bench-npu(NPU 对比)
├── wizard.py            # 交互式迁移向导(rich:启动界面→选码→进度迁移→反馈)
├── __main__.py          # CLI:wizard / migrate / bench-cpu / bench-npu / doctor
├── agents/              # analysis / decision / migration / verification
├── engines/             # 迁移引擎策略:base / ast_rewrite(默认) / llm(provider 无关, 可回退)
└── backends/            # 执行后端端口:base / shim(默认) / npu
examples/
├── mlp_forward.py       # 小型 2 层 MLP 分类器(正确性演示)
├── big_gemm.py          # 4 层 2048 宽 GEMM 网络(性能演示, NPU 发力)
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
