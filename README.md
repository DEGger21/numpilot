# NumPilot

NumPilot 是一个面向科学计算程序的 NumPy → AsNumpy/Ascend NPU 可审计迁移工具。当前默认
用户路径已经收敛为普通纯 NumPy 脚本：输入只需包含标准的 `import numpy as np`，不再强制
编写 `build_inputs()` 或 `forward()`。NumPilot 先静态扫描整个文件并核对实际 AsNumpy
capability；满足条件后由配置的 GLM 只做 `numpy/np → asnumpy/ap` 命名空间迁移，本地再用
完整 AST 门禁证明模型没有改变计算逻辑。

原有 `build_inputs + forward` 流水线只作为显式 `--mode contract` 的 legacy 严格通道保留，
不再作为普通用户迁移的前置格式要求。`--mode auto` 仅用于兼容旧调用，不是推荐入口。

完整愿景见 `NumPilot20260615-v1.pdf`。仓库现已实现缺口 API → GapSpec → 真实 LLM Ascend C 隔离草稿的安全纵切；编译、NPU 数值校验和激活仍受独立门禁约束，不会把“生成了源码”冒充成“算子已经可用”。

开发者从 [贡献指南](CONTRIBUTING.md) 开始，评委可直接查看
[证据索引](evidence/README.md) 与 [可信边界](#当前可信边界)；完整文档导航见
[`docs/README.md`](docs/README.md)。项目源码采用 [Apache-2.0](LICENSE)，第三方工具链与
生成算子的许可证边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 当前可信边界

- `migrate` 的产品默认入口是 ordinary script。只有用户显式传入 `--mode contract` 才进入
  legacy `build_inputs + forward` 通道；`--mode auto` 只保留兼容性，不应用于新 Demo、文档或
  比赛脚本。
- script 通道要求一个顶层 `import numpy as np`，拒绝其他 import、别名重绑定、动态
  `np`、`*args/**kwargs`、不兼容调用签名，以及尚未审计的 Python 数组运算符、ndarray
  method 和嵌套 namespace；扫描期间不 import、不执行源码。
- 所有普通脚本迁移代码必须来自显式配置的 GLM provider/model。NumPilot 的本地变换只作为
  AST 期望值，不会作为失败时的替代产物；完整远程回执、模型身份、内容哈希和二次 AST
  审计缺一不可。
- 普通脚本的能力结果只使用五类：`DIRECT`、`FALLBACK_CONSENT`、`GAP_REQUIRED`、
  `CAPABILITY_AUDIT_REQUIRED`、`BLOCKED`。`DIRECT` 才直接调用迁移 GLM；
  `FALLBACK_CONSENT` 必须等待用户接受或拒绝；只有 API 确实缺失，或用户拒绝已审计
  fallback，才进入 `GAP_REQUIRED`。
- callable 但 provenance 为 `unknown`、运行时版本/哈希漂移，或路径存在但没有版本绑定审计，
  都进入 `CAPABILITY_AUDIT_REQUIRED`：先做资格/源码审计，不得盲目生成同名替代算子。
  动态语义、未建模 Python 表面及当前不能安全处理的路径进入 `BLOCKED`。
- 用户接受 fallback 后可以生成兼容性迁移，但报告固定 `strict_npu=false`、
  `performance_valid=false`。当前 AsNumpy 也不被假定具有任意缺失 API 的通用 NumPy fallback。
- script 通道只迁移和落盘，不执行用户源码，因此报告固定
  `execution_performed=false`、`numerical_verification=not_run`。产品内尚未实现 ordinary script
  的通用 benchmark；本轮普通 GLM artifact 的真实 NPU 数值/性能证据来自外部受信、与源码和
  迁移产物哈希绑定的定向 harness，不得回写成迁移阶段已经执行。
- 每次 script 迁移使用 write-once 输出目录，避免上一次的 `migrated.py` 或 gap request
  混入本次证据。省略 `--out` 时会自动建立新的 run 目录；显式目录若已有 managed 产物会拒绝覆盖。
- `check-suite`、`wizard`、`bench-cpu`、`bench-npu` 都是 legacy contract 工具，要求
  `build_inputs + forward`，不能用于证明 ordinary script 的已发布 GLM artifact。
- contract 通道仍采用 `whole_forward_npu`：只分析和改写顶层 `forward()`，`build_inputs()`
  与顶层展示代码保留在 CPU；暂不宣称支持逐算子 CPU/NPU 混合卸载。
- contract 的 AST 引擎只保留给框架离线回归，不用于本次比赛案例；比赛案例走 strict LLM。
  该旧引擎在 `forward` 入口对 NumPy ndarray 做一次幂等 H2D，设备数组和标量保持原样。
- 验证先检查嵌套结构、shape、dtype，再按实际浮点 dtype 使用 `atol + rtol`；整数/布尔精确比较，显式处理零值、NaN 和 Inf。
- 真机 capability 绑定 AsNumpy 版本和运行环境；CPU fallback、版本未知或 API 缺失时 fail closed。
- strict `forward` 禁止残留 NumPy/host escape、未建模 helper/控制流、全局预计算和丢弃的设备结果；两条验证路径都记录实际 `ap.*` callsite 次数。
- `bench-npu` 仅在 strict-NPU、数值和 CPU 基线一致性门禁全部通过后计算 speedup。
- `--allow-cpu-fallback` 只用于诊断，报告始终为 `performance_valid=false`，不会发布 NPU 加速比。
- 普通脚本的 gap 只接受两类来源：实际缺失的直接 `numpy.<api>`，或用户明确拒绝的已审计
  CPU/host fallback replacement。legacy contract 还支持通过
  `@numpilot.operator_spec` 声明、映射到公共 AsNumpy API 的领域 helper。后者使用
  `operator.<name>` 身份并始终要求资格化，不能伪装成 NumPy API；普通 helper、签名
  缺口、Python dunder 和嵌套 namespace 仍会保持阻断。
- `complete-gaps --provider llm` 在新草稿路径上真实调用配置 endpoint，失败即终止，不回退
  AST；逐次 HTTP 2xx 回执与响应 SHA-256 会进入草稿 manifest。符合当前回执 schema 的
  已有草稿只做离线复验并明确记录 `reused_existing`，不能冒充本次联网。
- 算子生成固定使用华为官方 `cann/cannbot-skills` 的已审计 commit。远端 Chat Completions
  模型无法读取本机路径，因此 NumPilot 会校验 checkout 的 origin/commit/工作树，将所需
  `SKILL.md`、指南和 910B 模板内容连同逐文件 SHA-256 真正嵌入请求，而不是只告诉模型一个
  `/home/...` 路径。
- 当前仓库已有 GapSpec、未验证 draft/repair/seal、AsNumpy adapter 生成/repair、qualification
  schema 和 capability overlay 原语，但还没有一个命令自动完成
  `integrate → qualify → activate → rescan → final migration`。任何一步失败都必须保留为可恢复
  状态；不得把 `prepared`、`generated_unverified` 或 adapter candidate 写成已经修复。

已知重要事实：通用基线审计的 AsNumpy `0.3.1.dev4+g1929a3ba4` 中，`matmul` 会回到
host 调用 NumPy。因此 `mlp_forward.py` 和 `big_gemm.py` 仍只能用于迁移/诊断展示，不能
作为 NPU GEMM 加速证据。Poisson P0 另有一个文件与二进制哈希绑定的 move wheel：它只在
float32、非空、rank 1/2 的已测范围内放行修复后的 `outer`/`matmul` ACLNN 路径，并把
私有 native 返回值以显式 one-shot move 包装成公开 `ndarray`。同一 10 轮 profiler 的
kernel 数保持 1,170，`aclrtMemcpy` 从 1,235 降到 35；这仍不能反向扩大通用基线能力声明，
也不是性能结论。随后原始 48×48 内点、49 RBF、3000 轮配置通过独立 CPU/NPU、数值、
传输、profiler 和单线程实测门；2 次预热、7 次重复的 device-resident 中位数表明 CPU
约快 `2.67×`，没有 NPU 加速。随后广播/CSE 优化把每轮调用从 117 降到 101，NPU
中位时延从 `47.414 ms` 降到 `38.069 ms`（优化版快 `1.245×`），但 CPU 仍约快
`2.879×`。详见[原始基线](docs/evidence/poisson-ritz-performance-original-scale-20260823.md)
与[优化证据](docs/evidence/poisson-ritz-broadcast-optimization-20260823.md)。

ordinary script 的定向证据例子见
[Demo v1 Case 01 精选证据](evidence/demotest-case01-20260825/README.md)：GLM 生成的
2048×2048 float32 脚本及其精确 GLM 迁移产物由外部受信、哈希绑定 harness 验证。稳态脚本体
中位数为 CPU 单线程 `809.784 ms`、NPU `4.499 ms`，即 NPU 约快 `180.004×`；冷进程端到端
则是 CPU 约快 `4.44×`。该包没有 kernel profiler，因此只声明 runtime target/profile/call-count
guard 与 ACLNN 日志证据，不声明 profiler 已验证，也不代表产品内 ordinary 通用 benchmark
已经实现。

## 快速运行

从全新目录验证固定 tag、P0 AsNumpy wheel、CANNBot、GLM-5.3 迁移和 NPU 实跑时，直接打开
[Notebook 模板](docs/NumPilot-Demo-v1-Quickstart.ipynb)，并按
[全新 Notebook 验收路径](docs/FRESH-NOTEBOOK-QUICKSTART.md)执行。该路径当前不接入 ops-nn。

安装开发版本（这是用户显式执行的开发安装，不代表 `import` 或 `migrate` 可以联网）：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

基础运行依赖为 `numpy>=1.26`、`loguru>=0.7`、`rich>=13`。Demo v1 已审计 P0 wheel 的
运行组合是 CPython 3.10/aarch64、NumPy 1.26.4、Loguru 0.7.3；可用
`python -m pip install -e '.[dev,demo]'` 固定这组 Python 依赖。该 P0 二进制 profile 不代表
其他 Python/架构组合已经资格化。使用本地预编译
AsNumpy wheel 的 bootstrap 路径不需要额外构建工具。只有准备显式
`--build-asnumpy` 时，才应先由用户安装可选 `numpilot[bootstrap]`；从本仓库
开发时可用 `python -m pip install -e '.[dev,bootstrap]'`。`bootstrap --apply` 不会
自动从 PyPI 补齐缺失的构建依赖。
`--build-asnumpy` 会执行上游 CMake/PEP 517 构建，目前没有 OS 级断网沙箱；即使源码缓存已
存在，plan 也会保守标记该动作 `network=true`。全新 Notebook 的 Demo v1 首选本地审计 wheel。

依赖准备使用显式 bootstrap 层。不传模式或显式传入 `--plan` 都是只读计划；
`--json` 可输出机器可读计划：

```bash
# 两者都只读：检查环境并展示网络/文件/构建动作，不应用
python -m numpilot bootstrap
python -m numpilot bootstrap --plan
python -m numpilot bootstrap --plan --json

# 普通 Demo 首选：安装本地 wheel 前用可信发布物/用户输入的预期 hash 校验
python -m numpilot bootstrap --apply \
  --asnumpy-wheel /trusted/path/asnumpy.whl \
  --asnumpy-wheel-sha256 "$TRUSTED_ASNUMPY_SHA256" \
  --receipt /tmp/numpilot-bootstrap-receipt.json

# 与本地 wheel 模式二选一：需先显式安装 numpilot[bootstrap]
python -m numpilot bootstrap --apply --build-asnumpy \
  --receipt /tmp/numpilot-bootstrap-build-receipt.json

# 仅在确需 CANNBot 时附加；这时才必须显式接受 CANN OSL
python -m numpilot bootstrap --apply \
  --asnumpy-wheel /trusted/path/asnumpy.whl \
  --asnumpy-wheel-sha256 "$TRUSTED_ASNUMPY_SHA256" \
  --with-cannbot --accept-cannbot-license \
  --receipt /tmp/numpilot-bootstrap-with-cannbot.json

# 安装后真实复验 AsNumpy README smoke（shape=(1000, 1000)，float32）
python -m numpilot bootstrap --verify

# 同时从已固定的本地 bundle 回答 Ascend C Skill 清单；不联网
python -m numpilot bootstrap --verify --with-cannbot --accept-cannbot-license
```

实现内锁定 official AsNumpy source commit/tree 和 official CANNBot commit/bundle 身份，
但不内置 wheel URL 或预期 wheel hash。本地 wheel 的预期 SHA-256 必须来自可信
发布物或用户输入，bootstrap 会在安装前校验并将实际 hash 写入 receipt；运行时
资格 profile 仍由后续 capability 审计绑定。`--apply` 必须在“本地 wheel + expected
SHA-256”和 `--build-asnumpy` 之间二选一。`--asnumpy-cache` 与 `--cannbot-cache`
可显式指定锁定 checkout 缓存。只有选择 `--with-cannbot` 获取或复验 CANNBot 内容时才
必须同时传入 `--accept-cannbot-license`；单独准备 AsNumpy 不要求 CANN OSL。
NumPilot 不得由 pip 安装钩子、`import`、`doctor`、`migrate` 或 `complete-gaps`
隐式下载依赖。ordinary `migrate` 为生成迁移代码会调用已配置的 GLM endpoint，这是该命令
明确的模型动作，不是依赖安装；调用必须保留回执，并只在进程内临时使用用户代理。

普通纯 NumPy 脚本的最短路径：

```bash
python -m numpilot configure-llm --check
python -m numpilot migrate /path/to/case.py \
  --backend npu --out /tmp/numpilot-case-direct
```

如果扫描发现已审计 fallback，交互终端会询问。非交互环境必须显式选择：

```bash
# 接受兼容性迁移；不会声明 strict NPU 或性能有效
python -m numpilot migrate /path/to/case.py --backend npu \
  --fallback-policy accept --out /tmp/numpilot-case-accept

# 拒绝 fallback，先准备源码绑定的缺口请求
python -m numpilot migrate /path/to/case.py --backend npu \
  --fallback-policy reject --target-soc ascend910b --cann-version 9.0.0 \
  --out /tmp/numpilot-case-gap

# 只准备 GapSpec，不调用算子生成模型
python -m numpilot complete-gaps /path/to/case.py \
  --gap-request /tmp/numpilot-case-gap/gap-request.json \
  --target-soc ascend910b --cann-version 9.0.0 \
  --provider prepare --out /tmp/numpilot-case-gap/operator-gaps
```

以下是可选 legacy contract 的离线 CI 回归；它不是 ordinary script 快速路径：

```bash
python -m numpilot migrate examples/mlp_forward.py \
  --mode contract --engine ast --backend shim --out /tmp/numpilot-shim
python -m pytest -q
```

以下裸金属 smoke 和 benchmark 同样属于 legacy contract 工具：

```bash
python -m numpilot doctor --backend npu
python -m numpilot migrate examples/elementwise_forward.py \
  --mode contract --engine llm --backend npu --out /tmp/numpilot-npu

python -m numpilot bench-cpu examples/elementwise_forward.py \
  --out /tmp/numpilot-bench --cpu-threads 1
python -m numpilot bench-npu examples/elementwise_forward.py \
  --out /tmp/numpilot-bench
```

性能结论只读取：

- `/tmp/numpilot-bench/npu_benchmark.json`
- `/tmp/numpilot-bench/benchmark_report.md`

其中必须满足 `performance_valid=true`，才可以引用 `speedup`。这些文件只证明相应
`build_inputs + forward` contract run；不能替代普通脚本已发布 GLM artifact 的哈希绑定定向
harness 证据。

默认运行产物不会写入 Git 工作树。执行以下命令查看配置、缓存、报告和运行目录：

```bash
python -m numpilot paths
```

默认状态根为 `$XDG_STATE_HOME/numpilot`（通常是 `~/.local/state/numpilot`）；可用
`NUMPILOT_STATE_HOME`、`NUMPILOT_RUNS_DIR`、`NUMPILOT_REPORTS_DIR` 覆盖。所有原有
`--out` 参数继续有效，适合 CI 和比赛评测显式指定一次性目录。详细策略见
[`docs/artifact-storage.md`](docs/artifact-storage.md)。

准备 legacy contract 批量测试集时，可先运行完全不执行用户源码的静态准入检查：

```bash
python -m numpilot check-suite /path/to/cases \
  --expect-direct 5 --expect-single-gap 5 \
  --target-soc ascend910b --cann-version 9.0.0 \
  --out /tmp/numpilot-suite-check.json
```

`check-suite` 不检查普通脚本默认入口。目录输入只扫描当前层 `*.py`；该门禁要求每个 legacy
contract 案例恰有一个无参数 `build_inputs()` 和一个只用
必需位置参数的同步 `forward()`，并拒绝嵌套 NumPy namespace、标量二元调用、未建模控制流、
多缺口、已有 API 别名伪装成缺口，以及非 unary/binary 单输出 ufunc 的复杂缺口。它只做 AST、
GapSpec 和强制改写表面审计，不会 import 或执行案例；因此 `pass` 只表示“测试集结构适合当前
  legacy contract 评测赛道”，不能替代目标 AsNumpy wheel 的 strict-NPU provenance、真机数值和自定义算子
六门资格化。

真实科学计算网络的完整一次实测见
[`docs/evidence/heat-diffusion-npu-20260813.md`](docs/evidence/heat-diffusion-npu-20260813.md)：
二维热扩散单步网络的 7 个调用全部迁移并通过 strict-NPU 与零误差门禁；报告同时保留了
计时口径和当前能力边界。

也可直接运行裸金属演示脚本：

```bash
NUMPILOT_PY=python3 ./run_demo.sh
```

脚本使用 `set -euo pipefail`；任何 doctor、迁移或 strict-NPU 门禁失败都会停止，不会继续打印“演示成功”。默认不使用 Docker。

比赛/Poisson Ritz 客户体验使用专用入口（仅在 Git 仓库 checkout 中提供，不属于精简的
wheel 安装内容）：

```bash
./run_poisson_demo.sh                 # 公开证据校验 + 实时 NPU 迁移 + 快速驻留执行
./run_poisson_demo.sh --evidence-only # 无 NPU 时只校验可公开证据
./run_poisson_demo.sh --full          # 重跑 48×48 / 49 RBF / 3000 轮完整门控评测
./run_poisson_demo.sh --large-single-core --cpu-core 0
                                      # 128×128 / 225 RBF，CPU 进程只允许一个逻辑核
```

默认会折叠 AsNumpy/CANN 的逐算子日志并保存到本次产物目录，终端只展示
分析、算子资格、strict-NPU、数值和性能门。`--full` 会明确报告当前 CPU
仍更快，不把优化前后的 NPU 改善包装成对 CPU 的加速。本机完整制品默认从仓库
同级 `numpilot-gap-runs` 发现；移动制品后可用 `NUMPILOT_POISSON_*` 环境变量显式指定。

`--large-single-core` 是实验入口：它在 Python/NumPy 启动前用 `taskset` 固定 CPU worker，
在收据中用 `sched_getaffinity` 验证仅剩所选逻辑核，并要求 BLAS runtime 实测为 1 线程。
该 runner/规模尚无哈希匹配且通过的 profiler audit，因此其测量只能用于本机诊断；在补齐
同身份 profiler 证据并通过发布门以前，`performance.published` 必须保持 `false`，README、
发布说明和比赛材料不得引用其性能比值。

## 示例

| 文件 | 用途 | strict-NPU 预期 |
|---|---|---|
| `examples/elementwise_forward.py` | square/add/sqrt/tanh 的裸金属 smoke | 当前审计无已知 fallback，仍以运行时 manifest 为准 |
| [`examples/heat_diffusion_forward.py`](examples/heat_diffusion_forward.py) | 1024×1024 二维热扩散单步网络 | 已有 [910B4 实测证据](docs/evidence/heat-diffusion-npu-20260813.md)，7 个逐元素调用 strict-NPU 通过 |
| [`examples/poisson_ritz_step.py`](examples/poisson_ritz_step.py) | 确定性 Poisson/RBF 损失与一次参数更新，含两个显式领域算子 | 两算子已通过六门资格化；显式加载 overlay 后，9 次调用 strict-NPU 单步通过（[证据](docs/evidence/poisson-ritz-p0-npu-20260821.md)） |
| [`examples/poisson_ritz_iteration.py`](examples/poisson_ritz_iteration.py) | RBF、Laplacian、损失、解析梯度和更新均在一个动态 `forward` | 117 个调用点完成 strict-NPU；原始规模 3000 轮与重复性能门通过，但 CPU 约快 `2.67×`（[证据](docs/evidence/poisson-ritz-performance-original-scale-20260823.md)） |
| [`examples/poisson_ritz_iteration_broadcast.py`](examples/poisson_ritz_iteration_broadcast.py) | 通过广播与公共子表达式复用减少 Poisson 迭代 launch | 101 个调用点完成 strict-NPU；NPU 比 117-call 基线快 `1.245×`，CPU 仍约快 `2.879×`（[证据](docs/evidence/poisson-ritz-broadcast-optimization-20260823.md)） |
| `examples/mlp_forward.py` | 两层 MLP 的迁移与正确性示例 | 因 `matmul` CPU fallback 被拒绝性能发布 |
| `examples/big_gemm.py` | 历史 GEMM 压测输入 | 只允许诊断，不是 NPU GEMM 加速证据 |
| `examples/unsupported_demo.py` | `np.fft.fft` 缺口负例 | 分析阶段 early stop |
| `examples/gap_cbrt.py` | 直接缺口 API 的 GapSpec/LLM 草稿示例 | 只生成未验证草稿，不进入 strict-NPU |

## Legacy contract 交互向导

```bash
python -m numpilot wizard
python -m numpilot wizard examples/elementwise_forward.py --engine llm --backend npu
```

`wizard` 只接受 legacy `build_inputs + forward` contract。它展示整段 NPU 岛决策、产物级
语义校验和 capability 门禁，并给出同属 contract 的 `bench-cpu`/`bench-npu` 命令；它不是
ordinary script 默认入口。

界面中的“命中 AsNumpy 扩展 API”表示复用了已有 API，不等于本次新生成了自定义算子。新生成算子数必须由独立的 OperatorArtifact、构建和测试证据统计。

## CodeArts 多智能体接入

仓库已按华为云码道项目级格式提供：

```text
.codeartsdoer/
├── agents/
│   ├── numpilot-orchestrator.md
│   ├── numpilot-analyzer.md
│   ├── numpilot-operator-engineer.md
│   └── numpilot-evidence-auditor.md
└── skills/
    └── numpilot-migration-gate/SKILL.md
```

建议让 CodeArts AgentTeam 负责角色协作、任务可视化和人工确认，让 NumPilot CLI/contract/capability registry 负责确定性执行与证据落盘。完整映射、官方资料链接和比赛演示脚本见 [`docs/codearts-integration.md`](docs/codearts-integration.md)。

## 架构

```text
NumPilot bootstrap（默认 plan）/ --apply   显式、lock-bound 依赖准备
                    │
                    ▼
NumPilot migrate（ordinary script 默认）
  静态扫描 → CapabilityManifest → 五类确定结果
    ├─ DIRECT                    → GLM 精确迁移 → AST/receipt 门 → 落盘
    ├─ FALLBACK_CONSENT          → 用户接受 / 拒绝
    ├─ GAP_REQUIRED              → GapSpec → P1 ops-nn 官方复用候选
    │                                ├─ 精确匹配 → 官方 kernel + GLM adapter
    │                                └─ 不匹配 → GLM kernel/adapter 草稿
    ├─ CAPABILITY_AUDIT_REQUIRED → 资格或源码审计；禁止盲目生成
    └─ BLOCKED                   → 原因落盘

NumPilot migrate --mode contract（显式 legacy）
  AnalysisAgent → whole_forward_npu → MigrationAgent → VerificationAgent

注意：gap 自动 integrate/qualify/activate/rescan 编排和 ordinary 通用 benchmark 尚未实现。
```

核心结构：

- `numpilot/contracts.py`：智能体之间的数据契约；
- `numpilot/script_migration.py`：普通脚本静态扫描、能力分类、GLM 精确改写审计与产物报告；
- `numpilot/capabilities.py`：可序列化 capability/provenance；
- `numpilot/comparison.py`：统一严格比较规则；
- `numpilot/engines/`：AST/LLM 迁移策略；
- `numpilot/backends/`：shim/NPU 端口适配器；
- `numpilot/bench.py`：独立进程基线与 fail-closed NPU 评测；
- `numpilot/gaps/`：缺口规格、隔离生成、哈希 manifest、离线 qualification evidence 准入
  与原子 capability overlay；`NpuBackend` 可显式加载 overlay，并以 source/callsite profile
  和每次真实调用的 dtype/rank/C-contiguous guard 双重失败关闭；
- `docs/numpilot-technical-review.md`：完整技术评审和分阶段计划。

当前保留 `--threshold` 仅为 CLI/契约兼容；`whole_forward_npu` 策略不会再用它制造不完整的逐算子混合 placement。

## LLM 引擎

推荐先生成本机私有配置模板，再用编辑器手动填写：

```bash
python -m numpilot configure-llm --init
# 打开上一条命令打印的 llm.json，填写 base_url、auth_token、model
python -m numpilot configure-llm --check
python -m numpilot configure-llm --probe
./run_demo.sh --llm
```

`--init` 会生成包含 `base_url`、`auth_token`、`model`、`provider` 的 JSON，不会覆盖已有配置。`base_url` 应包含 OpenAI 兼容 API 的 `/v1` 前缀，不要包含 `/chat/completions`。也可运行 `python -m numpilot configure-llm` 使用隐藏 Token 输入的交互模式。

`--check` 只做本地配置完整性检查，**不联网**；`--probe` 才会发起一次真实的轻量请求。
探针成功会打印 `actual_remote_call=yes`、HTTP 状态、TLS 校验状态、请求/响应模型和内容
SHA-256，但不会打印 Token、Base URL 或响应正文；未成功取得可验证响应时返回非零退出码并明确
打印失败。

普通脚本通道为了满足比赛生成约束，会额外要求配置中 `provider=glm` 且 model 名为
`glm-*`；其他 provider/model 会在请求前失败，不会回退本地 AST。旧 contract 通道的
`--engine llm` 仍遵循其原有显式选择语义。

框架默认使用 `$XDG_CONFIG_HOME/numpilot/llm.json`（通常为 `~/.config/numpilot/llm.json`），目录权限 `0700`、文件权限 `0600`。临时容器或只持久化特定目录的平台，应先把 `NUMPILOT_CONFIG_HOME` 指向仓库外的持久目录，再执行 `--init`；后续运行也要保持该环境变量一致。真实凭据不属于仓库内容，永远不得提交到开源仓库。

CI/流水线无需交互，可继续注入 `NUMPILOT_LLM_BASE_URL`、`NUMPILOT_LLM_AUTH_TOKEN`、`NUMPILOT_LLM_MODEL`；环境变量逐字段优先于用户配置。也可用 `NUMPILOT_CONFIG_HOME` 或 `NUMPILOT_LLM_CONFIG_FILE` 覆盖默认配置位置。`--engine llm` 使用 strict 语义：未配置、调用失败或返回非法代码会退出，不回退 AST。TLS 证书默认校验；只有受信内网自签网关才应显式设置 `NUMPILOT_LLM_INSECURE=1`。

## 缺口算子补全

新 kernel 生成前优先查询华为官方 [`cann/ops-nn`](https://gitcode.com/cann/ops-nn) 是一个
已确认有价值的 **P1 待定项**。它不属于默认安装，也尚未实现 resolver：必须匹配 CANN
分支/tag、显式接受 CANN OSL、逐项证明 GapSpec 语义一致、仅构建被选算子到私有 OPP，并且
所有新增 AsNumpy adapter 仍由 GLM 生成、通过原有六门资格化。没有精确匹配时才进入新
Ascend C kernel 生成。详见
[ops-nn 官方算子复用评估](docs/OPS-NN-REUSE-ASSESSMENT-20260825.md)。
复用的官方 kernel 必须保留 official provenance/hash/license，不会被标记成 GLM 生成代码；
所有为 NumPilot 新写的案例、迁移、算子和 adapter 代码仍遵守 GLM-only 约束。

依赖的首选准备入口是上文的 `bootstrap`。只有显式的
`bootstrap --apply --with-cannbot --accept-cannbot-license` 才会按实现内固定的华为官方
CANNBot Skills commit 获取并校验 bundle，同时将 CANN OSL 接受状态写入
receipt。`setup-skills` 仅保留为低层兼容
工具，不应出现在 ordinary Demo 主路径：

```bash
python -m numpilot setup-skills --accept-cannbot-license
python -m numpilot setup-skills --check
```

默认缓存位于 `~/.cache/numpilot/cannbot-skills/<commit>`，可用
`NUMPILOT_CANNBOT_CACHE` 改到其他仓库外目录。`git clone numpilot` 本身不会执行网络脚本；
只有用户显式确认的 `bootstrap --apply --with-cannbot --accept-cannbot-license`
或带同等许可证接受的低层 `setup-skills` 才允许准备固定 commit，并严格
核对官方 origin、commit 和被选文件是否有本地修改。`complete-gaps`、`import` 和 `migrate`
不得隐式获取依赖。离线缓存缺失时必须失败；已有 checkout 可显式传
`--skill-root /path/to/cannbot-skills`。

第一步只生成确定性的 GapSpec 和生成契约，不调用大模型，也不补取缺失 Skill；依赖必须先由
bootstrap 或显式低层命令准备：

```bash
GAP_RUNS="$(python -m numpilot paths --get gaps)"
python -m numpilot complete-gaps examples/gap_cbrt.py \
  --out "$GAP_RUNS/cbrt" \
  --target-soc ascend910b --cann-version 9.0.0
```

显式增加 `--provider llm` 才会调用已配置的大模型：

```bash
GAP_RUNS="$(python -m numpilot paths --get gaps)"
python -m numpilot complete-gaps examples/gap_cbrt.py \
  --out "$GAP_RUNS/cbrt" \
  --target-soc ascend910b --cann-version 9.0.0 \
  --provider llm --max-attempts 3
```

当前远端草稿协议使用 `NUMPILOT_BUNDLE_V1` 行式信封承载原始 C++/Python 内容，避免部分
OpenAI 兼容网关不能可靠强制 JSON 字符串转义的问题；路径穿越、目录白名单、文件数、总
大小、UTF-8、必需工程结构和最终哈希仍由本地校验器失败关闭。这个协议只改变传输封装，
不降低后续 build/UT/数值/NPU/activation 门禁。

对显式领域 `OperatorSpec`，NumPilot 使用固定的 17 文件逐文件生成计划，包含
`tiling_key.h`。每个阶段都重复 OpType、公共 API、签名、dtype 和 shape 不变量，减少
Host/kernel/L0/L2/test/adapter 在分阶段生成时相互矛盾。模型回复的“已完成”不是证据，
本地编译和资格化结果才能改变 gate 状态。

全新生成成功时，输出必须包含 `action=fresh_remote`、`mode=remote_llm`、配置模型名、
`fallback=disabled`、`actual_remote_call=yes` 和响应 SHA-256。复用已有产物时会明确打印
`actual_remote_call=no action=reused_existing`。生成失败会打印
`verified_new_remote_artifact=no`，不会用 AST 结果掩盖失败。`gap-report.json` 还保存每个草稿的
`current_run_actual_remote_call` 以及本次实际调用总数；`draft/operator-artifact.json` 会按每个成功 HTTP
响应保存严格白名单回执：UTC 起止、状态码、请求/响应模型、响应 ID/created、request ID、
请求体/响应体/内容 SHA-256、整数 usage 和 TLS 校验状态；Token、Authorization header、
Base URL、endpoint 指纹和原始响应不会落盘。响应中的模型名是 endpoint 自报值，不是厂商
签名 attestation。

如果符合当前严格 schema 的 draft 已存在，provider 在不加载当前凭据的情况下复验
manifest、全部 transport 回执/计数、文件全集和哈希，输出 `action=reused_existing`。
缺少逐次回执的旧格式远程草稿会被拒绝，不能补写字段后冒充新证据。原始生成动作保存在
`generation.origin_action`，本次动作写入顶层 `gap-report.json` 的
`current_run_action(s)`；两者不能混用。模型输出最多重试三轮，但这里只完成结构校验，
状态固定为 `generated_unverified`，构建、单测、数值验证和激活均为 `pending`。

隔离构建或测试失败后，可把有界诊断回传模型，并只生成指定文件的新候选：

```bash
python -m numpilot repair-draft \
  --spec /path/to/gap-spec.json \
  --draft /path/to/original/signed/draft \
  --out /path/to/new-candidate \
  --generation-request /path/to/GENERATION_REQUEST.md \
  --diagnostic-file /path/to/compiler-diagnostic.txt \
  --path op_kernel/example_arch22.cpp
```

`--draft` 始终指向不可变的原始 Artifact。第二轮起可增加
`--base-candidate /path/to/previous-candidate`；上一轮必须先通过原始 manifest、repair
report、完整文件哈希、额外文件和 symlink 校验。新 report 会绑定父 report SHA-256、完整
候选文件映射、诊断和逐次远端回执。成功与失败都会明确打印 `actual_remote_call`；HTTP
成功但正文校验失败不会被误报为“没有调用”。repair 状态仍是 `repaired_unverified`。

最终 repair 链通过本地复验后，用不联网的封存命令发布新的不可变 Artifact：

```bash
python -m numpilot seal-repair \
  --spec /path/to/gap-spec.json \
  --draft /path/to/original/signed/draft \
  --candidate /path/to/final-repair-candidate \
  --out /path/to/new-sealed-artifact
```

封存会逐级验证父 report 哈希、每轮前后文件哈希、原始签名 manifest、完整文件集合和全部
远端回执，并把 repair provenance 作为第 18 个签名文件写入新 Artifact。该动作明确打印
`actual_remote_call=no action=sealed_repair_chain`；新 Artifact 仍是
`generated_unverified`，只是已经可以作为后续 qualification 的唯一输入，不能因此跳过
构建、UT、public API、真实 NPU 数值和 activation 门禁。

输出目录包含：

- `gaps/<gap-id>/gap-spec.json`：qualified API、脱敏调用形状、目标 SoC/CANN 和 route 审计；
- `gaps/<gap-id>/GENERATION_REQUEST.md`：实际送入模型的生成契约和 Skill 内容；
- 顶层 `skill-bundle.json`：实际注入模型的官方仓库、固定 commit、许可证、bundle/逐文件哈希；
- `gaps/<gap-id>/draft/`：模型源码和带全文件哈希的 `operator-artifact.json`；
- 顶层 `gap-report.json`：prepared/generated/blocked 计数，qualified/activated 在通过门禁前始终为 0。

草稿不会自动执行 `build.sh`、写系统 CANN、修改 `~/.bashrc`、安装 OPP 或注册 AsNumpy。
adapter generation/repair、qualification schema 和 overlay activation 都是独立原语；当前没有
自动 `integrate → qualify → activate → rescan` 编排。当前机器若 `doctor --backend npu` 未同时
通过 NPU 设备与 AsNumpy 检查，只能静态审查草稿，不能宣称补全闭环完成。

## 仍未完成

- callsite/value 级 IR、动态 shape guard 和真正的 CPU/NPU 子图划分；
- OS 级用户代码隔离与完整网络/文件权限收敛；
- 通用 H2D/D2H 字节数、kernel trace 和 profiler 自动采集；Poisson P0 的定向采集、复制
  审计、消除及原始规模性能门已完成；
- 通用化的多算子 qualification runner、安装事务与自动回滚；Poisson P0 已有受信外部 harness 完成构建、UT、NPU 数值、私有安装和 overlay 激活闭环；
- ordinary GLM artifact 的产品内通用数值/性能 runner；本轮只承认外部受信、源码/产物/运行时
  哈希绑定的定向 harness；
- gap 的自动 integrate、qualify、activate、rescan 与最终 GLM migration 状态机；
- ops-nn 版本化 catalog/reuse resolver、私有 OPP 安装事务与回滚；
- 通用 ndarray 结构语义、异步 stream、内存池以及 AsNumpy 插件 ABI。

这些边界不会被 shim 通过率或“替换了多少 API”掩盖。
