<div align="center">

# NumPilot

### 让普通 NumPy 科学计算脚本可信地驶向 Ascend NPU

<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.3.0a1.dev0-7B61FF?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white">
  <img alt="Ascend" src="https://img.shields.io/badge/Ascend-910B4-C7000B?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-2EA44F?style=flat-square">
  <img alt="Stage" src="https://img.shields.io/badge/stage-development--snapshot-F59E0B?style=flat-square">
</p>

<p>
  <a href="#value">项目价值</a> ·
  <a href="#results">真机结果</a> ·
  <a href="#quick-demo">快速演示</a> ·
  <a href="#architecture">系统架构</a> ·
  <a href="#boundaries">可信边界</a> ·
  <a href="#evidence-audit">证据审计</a>
</p>

</div>

> [!IMPORTANT]
> 当前源码版本是未发布的 `0.3.0a1.dev0`。本文引用的两份 Ascend 真机证据捕获时
> `git_dirty=true`，因此它们是可审计的开发态证据，不是最终 RC，也不冒充不可变发布结果。

<a id="value"></a>

## 三分钟看懂 NumPilot

NumPilot 面向科学计算团队，把普通单文件 NumPy 程序迁移到华为 Ascend NPU。它不是简单地
替换一行 `import`：系统先判断目标 API 在当前 AsNumpy/CANN/SoC 组合中是否真实可用，再由
用户配置的大模型生成迁移代码，最后用本地确定性门禁验证语义、数值、NPU 来源和性能结论。

```text
普通 NumPy 源码
    │
    ├─ 静态五态分析：当前运行时真的能迁移吗？
    │
    ├─ configured model：生成 AsNumpy 迁移产物
    │
    ├─ exact AST：模型有没有越权改变计算逻辑？
    │
    ├─ 双进程数值对拍：结果是否一致？
    │
    └─ strict-NPU benchmark：是否真的在 NPU 上、是否值得发布加速比？
```

NumPilot 的核心原则是：**模型负责生成候选，确定性系统负责决定候选能否被采用。**

| 评审关心的问题 | NumPilot 的回答 |
|---|---|
| API 是否真的存在 | 绑定 AsNumpy 版本、源码/二进制哈希与当前 runtime 的 capability 审计 |
| 模型是否改坏算法 | 对生成产物执行完整 exact AST 等价门；失败即停止，不回退伪造结果 |
| 是否悄悄回到 CPU | strict-NPU provenance、profile guard、实际调用次数和逐样本同步门 |
| 数值是否可信 | 原脚本与迁移脚本分离进程运行，按命名输出比较 shape、dtype 与数值 |
| 加速数字是否公平 | CPU/NPU 同逻辑核、CPU BLAS 实测单线程、原始样本与统计全部入回执 |
| 证据能否复核 | 源码、产物、模型回执、runtime 和统计由 SHA-256 write-once 链绑定 |

<a id="results"></a>

## 当前可核查结果

当前比赛展示绕开已知会 Host fallback 的通用 `matmul` 路径，统一使用 9,000,000 个
`float32` 元素。两条结果均由 configured `glm-5.3` 实际远程生成迁移代码，并通过 exact AST、
named-output 数值验证和 `16/16` 性能门。

| 证据层 | CPU / NPU resident 中位 | 中位加速 | 保守加速 | 一次组件和 |
|---|---:|---:|---:|---:|
| **主结果：Poisson-Ritz 源项与鲁棒残差阶段** | `458.698 / 178.658 ms` | **`2.567×`** | `2.471×` | `0.789×`，CPU 更快 |
| **优势区间：纯逐元素合成压力链** | `616.333 / 76.289 ms` | **`8.079×`** | `7.503×` | `1.220×` |

主结果的 bootstrap 95% 区间为 `[2.538×, 2.592×]`；9,000,000 个输出元素验证通过，
最大绝对误差 `1.9073486e-06`，最大相对误差 `5.6004706e-06`。合成链的 bootstrap 95%
区间为 `[7.954×, 8.147×]`，最大绝对误差 `2.3841858e-07`。

> [!NOTE]
> resident headline 统计输入已经驻留设备后的重复计算，不包含冷启动、一次输入准备和最终 D2H。
> 表中的“一次组件和”均标记为 `directly_sampled_end_to_end=false`，只是诊断账目，不是直接采样
> 的 E2E 延迟。Poisson 结果明确说明：首次单次执行仍可能是 CPU 更快。

Poisson 案例从 PDE 工作流抽取解析源项与 `SmoothL1(beta=1)` 未规约残差图，具有明确数学语义，
但它**不是完整 Poisson 求解器，不使用公开数据集，也没有真实受益方声明**。合成链只描述
大数组、设备驻留、重复复用并在最后一次回传时的优势区间，不代表行业应用性能。

完整材料：

- [Poisson 应用形态开发态证据](evidence/competition-poisson-stage-20260828/README.md)
- [逐元素合成优势区间证据](evidence/competition-ordinary-20260828/README.md)
- [比赛证据总索引](evidence/README.md)
- [比赛一页纸](docs/COMPETITION-ONE-PAGER.md)

<a id="quick-demo"></a>

## 3–5 分钟现场演示

### 1. 准备环境

目标机器需要预装 Ascend 驱动、CANN、可用 NPU、Python 3.10+、Git、GCC 11.2+，以及
`fmt`/`spdlog` 的 CMake 开发包。首次安装建议使用固定源码 bootstrap 流程：

```bash
git clone https://gitcode.com/li-xingHIT/numpilot.git
cd numpilot

./install.sh --allow-network \
  --persistent-root /absolute/persistent/numpilot-runtime
source /absolute/persistent/numpilot-runtime/activate-numpilot.sh
```

安装器不会执行 `sudo`、修改 shell 启动文件或保存模型凭据。完整的新机器步骤见
[全新 Notebook 验收路径](docs/FRESH-NOTEBOOK-QUICKSTART.md)。

### 2. 配置并探测模型

```bash
python -m numpilot configure-llm --init
# 编辑命令打印出的私有 llm.json：填写 base_url、auth_token、model
python -m numpilot configure-llm --check   # 仅检查本地配置，不联网
python -m numpilot configure-llm --probe   # 显式执行一次真实轻量请求
```

默认配置位于 `$XDG_CONFIG_HOME/numpilot/llm.json`，通常是
`~/.config/numpilot/llm.json`；目录权限应为 `0700`、文件权限应为 `0600`。Token、Base URL 和
响应正文不会写入迁移回执，真实凭据不得提交到仓库。

### 3. 运行主链

```bash
./run_demo.sh --quick
```

默认案例是 `examples/poisson_source_smooth_l1_map.py`。脚本依次执行：

1. `doctor --backend npu`：验证真实 NPU runtime；
2. `analyze --require-direct`：静态分析全部调用点；
3. `migrate --engine llm`：调用 configured model 并保存 HTTP/TLS/model 回执；
4. `verify-script`：独立执行两侧代码并做命名输出数值对拍；
5. `benchmark-script`：执行单核 CPU / strict-NPU resident 评测；
6. 渲染只读取 JSON 回执的评委一屏摘要。

`--quick` 使用 2 次预热和 7 个样本，适合现场；默认使用 3 次预热和 20 个样本；
`--full` 使用 3 次预热和 30 个样本。每次运行写入新的仓库外状态目录，可用
`python -m numpilot paths` 查看。

## 分步使用

如果希望逐步检查每一道门：

```bash
RUN=/tmp/numpilot-poisson-demo
SRC=examples/poisson_source_smooth_l1_map.py

python -m numpilot analyze "$SRC" \
  --require-direct --out "$RUN/qualified-analysis.json"

python -m numpilot migrate "$SRC" \
  --mode script --engine llm --backend npu --fallback-policy reject \
  --out "$RUN/migration"

python -m numpilot verify-script "$SRC" \
  --migration-report "$RUN/migration/script-migration-report.json" \
  --output result --out "$RUN/script-verification-report.json"

python -m numpilot benchmark-script "$SRC" \
  --migration-report "$RUN/migration/script-migration-report.json" \
  --output result --warmup 3 --samples 20 \
  --out "$RUN/script-benchmark-report.json"
```

只有 benchmark 回执中 `performance_valid=true` 时，结果才允许发布 speedup。
`bench-script` 是 `benchmark-script` 的等价短别名。

<a id="architecture"></a>

## 系统架构

NumPilot 将“可变的模型判断”与“确定性的工程事实”分开：

```text
┌──────────────────── CodeArts 协作控制面 ────────────────────┐
│ Orchestrator · Analyzer · Operator Engineer · Evidence Auditor │
└────────────────────────────┬───────────────────────────────────┘
                             │ 结构化任务 / 人工确认
┌────────────────────────────▼───────────────────────────────────┐
│                     NumPilot 确定性执行面                       │
│                                                                  │
│ ordinary：Analyze → Model → AST → Verify → Benchmark → Evidence │
│ gap：Official ACLNN → pinned ops-nn → CANNBot candidate         │
│      → build / qualify / activate（各阶段独立门禁）              │
└────────────────────────────┬───────────────────────────────────┘
                             │ hash-bound capability
                    AsNumpy / CANN / Ascend NPU
```

- ordinary 路径服务普通 `import numpy as np` 脚本，是当前推荐入口；
- legacy `build_inputs() + forward()` 合同只由显式 `--mode contract` 启用；
- 缺口路径坚持 official-first：已安装 ACLNN、固定版本 `ops-nn` 均无匹配后，才允许模型按
  CANNBot Skill 生成隔离候选；
- “生成候选”“构建通过”“数值资格化”“激活可用”是不同状态，不能互相替代。

完整设计见 [架构说明](docs/architecture.md) 与
[三页可编辑 draw.io 架构图](docs/numpilot-architecture.drawio)；面向论文/答辩的单页版本见
[中文可编辑架构 PPT](docs/numpilot-architecture-paper-style-zh-readable.pptx)。

## 为什么证据可信

### 迁移前：静态五态决策

`analyze` 不 import、不执行源码，也不调用模型。每个调用点只会进入以下一种状态：

- `DIRECT`：当前 hash-bound runtime 已证明可直接迁移；
- `FALLBACK_CONSENT`：存在已审计 fallback，必须由用户明确接受；
- `GAP_REQUIRED`：API 确实缺失，进入缺口恢复；
- `CAPABILITY_AUDIT_REQUIRED`：API 看似存在，但来源或版本尚不可信；
- `BLOCKED`：动态语义或 Python 表面超出当前安全范围。

### 迁移中：模型身份与 exact AST

ordinary 迁移必须来自显式配置的模型。回执保存实际远程调用、HTTP 状态、TLS 校验、请求和
响应模型身份、内容及 artifact 哈希；任一身份缺失或不匹配都会失败关闭。模型生成后，本地
exact AST 门只允许受控的 `numpy/np → asnumpy/ap` 命名空间变化，不允许改写计算逻辑。

### 迁移后：数值、runtime 与性能父门

`verify-script` 对 hash-bound source/artifact 执行双进程 named-output 比较。
`benchmark-script` 进一步要求：

- 迁移状态为真实 configured-model 产生的 `migrated_strict_npu`；
- CPU/NPU worker 相互独立并固定同一逻辑核；
- CPU 线程环境和实际 BLAS 线程均为 1；
- runtime identity 前后一致且 profile guard 通过；
- 实际 `ap.*` 调用数、同步数和预期完全相等；
- 最终输出仍通过数值检查。

任何父门失败，回执都固定 `performance_valid=false`，不会输出可宣传的 speedup。

<a id="evidence-audit"></a>

## 证据审计

仓库提供纯标准库只读审计器。它不执行用户代码、不导入 NumPy/AsNumpy、不联网、不访问 NPU，
但能复核目录 manifest、source/report/artifact/runtime 哈希，重算中位数、分位数、保守加速和
bootstrap 区间：

```bash
python tools/audit_script_benchmark_receipt.py \
  evidence/competition-poisson-stage-20260828 --json

python tools/audit_script_benchmark_receipt.py \
  evidence/competition-ordinary-20260828 --json

(cd evidence/competition-poisson-stage-20260828 && sha256sum -c MANIFEST.sha256)
(cd evidence/competition-ordinary-20260828 && sha256sum -c MANIFEST.sha256)
sha256sum -c evidence/MANIFEST.sha256
```

精简证据包不保存两份 36 MB 输出数组；它保存输出 SHA-256、shape/dtype/finite 元数据和数值
比较回执。因此第三方可以离线复算统计与身份绑定，但逐元素误差需要在匹配环境真机重放。

## AsNumpy 的优势区间

当前证据表明，以下组合最容易体现 NPU 优势：

- 单一 `float32` 数据类型与百万级以上连续数组；
- 多个可直接落到 ACLNN 的逐元素算子形成较长计算链；
- 输入一次 H2D 后在设备上重复复用；
- 中间过程不执行 `asnumpy`、Python 标量读取、文件 I/O 或 CPU↔NPU 往返；
- 仅在计算链末尾执行一次 D2H。

小数组、一次性调用、频繁 Host 交互、冷进程启动、规约同步或 CPU fallback 往往由固定开销
主导，CPU 可能更快。NumPilot 会保留这类反例，不会用 resident 数字掩盖端到端边界。

## Python 领域算子契约（Preview）

包级公共 API 还提供 `OperatorSpec` 与 `@operator_spec`，用于为领域 helper 声明 NumPy oracle、
目标 AsNumpy API、签名、dtype/shape、来源和许可证。静态分析器只接受字面量合同，并把未资格化
领域算子送入 GapSpec；装饰器本身不会自动构建、激活或授予 NPU 能力。

```python
from numpilot import operator_spec

@operator_spec(
    name="domain_update",
    version="1",
    numpy_oracle="domain_update_oracle",
    target_api="asnumpy.domain_update",
    provenance="custom_generation_candidate",
    route="ascendc_registry_candidate",
    signature_variants=({"positional_args": 2, "keyword_args": ()},),
    dtype_rules=("float32",),
    shape_rules=("same-shape",),
    source="competition-domain-contract-v1",
    license="Apache-2.0",
    target_soc="ascend910b",
    cann_version="9.0.0",
)
def domain_update_oracle(value, delta):
    return value - delta
```

该接口面向受控缺口恢复，不是 ordinary 脚本的必需入口。

<a id="boundaries"></a>

## 可信边界

| 当前支持 | 当前不宣称支持 |
|---|---|
| 顶层 `import numpy as np` 的普通单文件脚本 | 任意 Python、动态 import、任意 ndarray method |
| 已审计 DIRECT API 的 strict-NPU 迁移 | 未审计 API 自动等价或自动获得加速 |
| named numeric/bool 输出的显式数值验证 | OS 级用户代码沙箱与完整权限隔离 |
| marker 后直线 resident 计算区评测 | 通用 cold/warm E2E、多核 CPU 或 kernel-only profiler |
| official-first 缺口解析与候选证据链 | 通用自动 integrate → qualify → activate 全闭环 |
| 显式用户同意的 fallback 兼容迁移 | 把 fallback 结果宣传为 strict-NPU 性能 |

当前 pinned AsNumpy 基线中的通用 `matmul` 会回到 Host NumPy，因此不用于主性能结论。
仓库保留 gap 恢复控制面的实现与测试，但它们不能扩大当前 ordinary 主链声明。
详细限制见 [架构说明](docs/architecture.md)、[Gap 状态合同](docs/gap-recovery-state.md) 和
[ordinary benchmark 合同](docs/script-benchmark.md)。

## CodeArts 多智能体协作

仓库提供可随项目分发的 CodeArts 配置：

| 角色 | 责任 |
|---|---|
| `numpilot-orchestrator` | 拆解任务、串联门禁、控制状态推进 |
| `numpilot-analyzer` | 只读分析调用点、数据流与 capability |
| `numpilot-operator-engineer` | 在明确缺口后处理受控算子/adapter 候选 |
| `numpilot-evidence-auditor` | 与开发职责隔离，复核证据与宣传口径 |

对应配置位于 `.codeartsdoer/agents/`，迁移 SOP 位于
`.codeartsdoer/skills/numpilot-migration-gate/`。这些仓库文件证明配置已提供，**不等于已经在
目标 CodeArts 账号完成平台实跑**。AgentTeam 装配方式、账号能力和运行记录仍需在比赛环境验证。
参见 [CodeArts 集成说明](docs/codearts-integration.md)。

## 开发安装与测试

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

基础依赖是 `numpy>=1.26`、`loguru>=0.7`、`rich>=13`。从固定源码构建 AsNumpy 时请使用
`.[bootstrap]`；`import numpilot`、`doctor` 和普通依赖安装不会隐式下载 CANNBot 或算子源码。

```bash
python -m numpilot bootstrap --plan --json
python -m numpilot bootstrap --apply --build-asnumpy
python -m numpilot bootstrap --verify
python -m numpilot doctor --backend npu
```

`bootstrap --plan` 是只读计划。`--apply` 只有在用户明确执行后才允许网络、构建和写入用户态
状态目录；安装器不会静默换源。Gap/operator lane 还需要显式接受相应第三方许可证，具体命令
见 [新机器快速开始](docs/FRESH-NOTEBOOK-QUICKSTART.md)。

## 仓库结构

```text
numpilot/                 核心 CLI、合同、迁移、验证、benchmark 与 gap recovery
examples/                 ordinary 与 legacy 示例
tests/                    单元、集成、失败关闭和公开证据测试
tools/                    只读审计、证据与发布物检查工具
evidence/                 可公开结构化证据及 SHA-256 manifest
docs/                     架构、模型合同、验证协议与比赛材料
.codeartsdoer/            CodeArts agents 与 Skills
run_demo.sh               比赛 ordinary 五阶段现场演示
install.sh                显式、用户态、可审计安装入口
```

核心阅读入口：

- [文档总导航](docs/README.md)
- [模型配置与回执合同](docs/model-contract.md)
- [ordinary 数值验证](docs/script-verification.md)
- [ordinary strict-NPU resident benchmark](docs/script-benchmark.md)
- [Gap 恢复状态与证据链](docs/gap-recovery-state.md)
- [比赛要求映射](docs/COMPETITION-REQUIREMENTS-MAP-20260827.md)
- [比赛提交分支约定](docs/SUBMISSION.md)
- [发布检查清单](docs/RELEASE-CHECKLIST-DEMO-V1.md)

## 发布状态与下一步

当前仓库适合继续功能冻结和比赛演练，但 `0.3.0a1.dev0` 仍是 development snapshot。正式交付前
还需在干净 commit/tag 上重跑主链，构建同一 revision 的 wheel、sdist、交付 ZIP、SBOM 与逐文件
manifest，并在第二个全新 Ascend 环境复现。

最影响比赛真实性得分的后续工作是：引入具有固定来源、许可证、输入哈希和明确受益方的真实
负载；在真实 CodeArts AgentTeam 中保存任务图、对话、Diff 和执行记录；补充直接采样的 cold/E2E、
多核 CPU 对照和 exact-artifact profiler。完成前不会把这些目标写成已实现能力。

## 安全、贡献与许可证

- 安全问题与凭据处理：[SECURITY.md](SECURITY.md)
- 贡献流程与开发约定：[CONTRIBUTING.md](CONTRIBUTING.md)
- 第三方组件与许可证边界：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- 项目许可证：[Apache License 2.0](LICENSE)

NumPilot 源码采用 Apache-2.0。AsNumpy、CANN、CANNBot Skills、`ops-nn` 及模型服务遵循各自的
许可证与服务条款；仓库中的生成候选不会因为进入 NumPilot 流程而自动获得可发布资格。

<div align="center">

**NumPilot：让迁移结果不仅能运行，还能被解释、被否决、被复核。**

</div>
