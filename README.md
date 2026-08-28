# NumPilot

NumPilot 是一个面向科学计算程序的 NumPy → AsNumpy/Ascend NPU 可审计迁移工具。当前默认
用户路径已经收敛为普通纯 NumPy 脚本：输入只需包含标准的 `import numpy as np`，不再强制
编写 `build_inputs()` 或 `forward()`。NumPilot 先静态扫描整个文件并核对实际 AsNumpy
capability；满足条件后由用户配置的 OpenAI-compatible 模型只做
`numpy/np → asnumpy/ap` 命名空间迁移，本地再用
完整 AST 门禁证明模型没有改变计算逻辑。

**比赛主链已真机跑通：** configured `glm-5.3` 实际迁移，16/16 性能门通过；Ascend 910B4
上的 9M `float32` Poisson-Ritz 源项与鲁棒残差阶段，设备驻留中位加速 `2.567×`，保守结果
`2.471×`。一次输入准备 + 一次 resident + 最终 D2H 的组件和仅 `0.789×`，说明一次性调用仍是
CPU 更快；该组件和不是直接采样的 E2E 时延。另一个纯逐元素压力链给出 `8.079×` resident
优势区间，组件和为 `1.220×`，也不是直接 E2E。Poisson 阶段比合成压力链更接近 PDE 工作流，
但它**不是完整求解器，也不是公开数据负载**。评委可从[比赛一页纸](docs/COMPETITION-ONE-PAGER.md)、
[Poisson 阶段冻结证据](evidence/competition-poisson-stage-20260828/README.md)、
[合成优势区间证据](evidence/competition-ordinary-20260828/README.md)和
[只读审计器](tools/audit_script_benchmark_receipt.py)开始。

原有 `build_inputs + forward` 流水线只作为显式 `--mode contract` 的 legacy 严格通道保留，
不再作为普通用户迁移的前置格式要求。`--mode auto` 仅用于兼容旧调用，不是推荐入口。

职责上，AST 和 runtime capability 负责可复验的调用点与接口事实，大模型不充当“当前环境是否
有某个 API”的事实来源。当前 ordinary 产物仍采用“configured model 生成、exact AST 约束”；
后续只对可证明的纯机械改写增加 AST 快路径。official existing-module adapter 使用另一条受限
协议：模型决定最小语义编辑，本地只按 base hash、唯一 OLD 片段和不重叠区间做精确机械重建，
不会用 AST 替模型决定工程语义。模型的核心价值集中在复杂迁移，以及“官方算子优先、无官方
实现才用 CANNBot Skill 生成”的缺口算子与 AsNumPy 工程接入链。

当前愿景、实现状态与可信边界以[架构说明](docs/architecture.md)、
[三页可编辑架构图](docs/numpilot-architecture.drawio)和[比赛一页纸](docs/COMPETITION-ONE-PAGER.md)
为准。仓库现已实现两条安全纵切：installed
`official_binary` 可生成 reviewed、未验证的 AsNumPy adapter candidate；两层官方查询均 clean
`NO_MATCH` 时才允许真实 LLM + CANNBot Skill 生成隔离 Ascend C 草稿。编译、NPU 数值校验和
激活仍受独立门禁约束，不会把“生成了源码”冒充成“算子已经可用”。

开发者从 [贡献指南](CONTRIBUTING.md) 开始，评委可直接查看
**[比赛一页纸](docs/COMPETITION-ONE-PAGER.md)**、[证据索引](evidence/README.md) 与
[可信边界](#当前可信边界)；完整文档导航见
[`docs/README.md`](docs/README.md)。项目源码采用 [Apache-2.0](LICENSE)，第三方工具链与
生成算子的许可证边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 当前可信边界

- `migrate` 的产品默认入口是 ordinary script。只有用户显式传入 `--mode contract` 才进入
  legacy `build_inputs + forward` 通道；`--mode auto` 只保留兼容性，不应用于新 Demo、文档或
  比赛脚本。
- script 通道要求一个顶层 `import numpy as np`，拒绝其他 import、别名重绑定、动态
  `np`、`*args/**kwargs`、不兼容调用签名，以及尚未审计的 Python 数组运算符、ndarray
  method 和嵌套 namespace；扫描期间不 import、不执行源码。
- 所有普通脚本迁移代码必须来自显式配置的模型。`provider` 只是审计标签，当前最小协议合同是
  Bearer auth + Chat Completions；NumPilot 的本地变换只作为
  AST 期望值，不会作为失败时的替代产物；完整远程回执、模型身份、内容哈希和二次 AST
  审计缺一不可。
- 普通脚本的能力结果只使用五类：`DIRECT`、`FALLBACK_CONSENT`、`GAP_REQUIRED`、
  `CAPABILITY_AUDIT_REQUIRED`、`BLOCKED`。`DIRECT` 才直接调用已配置模型；
  `FALLBACK_CONSENT` 必须等待用户接受或拒绝；只有 API 确实缺失，或用户拒绝已审计
  fallback，才进入 `GAP_REQUIRED`。
- callable 但 provenance 为 `unknown`、运行时版本/哈希漂移，或路径存在但没有版本绑定审计，
  都进入 `CAPABILITY_AUDIT_REQUIRED`：先做资格/源码审计，不得盲目生成同名替代算子。
  动态语义、未建模 Python 表面及当前不能安全处理的路径进入 `BLOCKED`。
- 用户接受 fallback 后可以生成兼容性迁移，但报告固定 `strict_npu=false`、
  `performance_valid=false`。当前 AsNumpy 也不被假定具有任意缺失 API 的通用 NumPy fallback。
- `migrate` 只迁移和落盘，不执行用户源码，因此迁移报告固定
  `execution_performed=false`、`numerical_verification=not_run`。迁移后可由用户显式运行
  `verify-script`，在两个独立进程中对 hash-bound source/artifact 的明确命名 numeric/bool 输出
  做数值比较，并发布另一份 write-once 回执；它不是 OS sandbox，也不证明 strict-NPU 或性能。
  需要性能证据时再显式运行 `benchmark-script`：它只接受真实 configured-model 生成的
  `migrated_strict_npu` ordinary 产物，并在两个独立、同逻辑核、CPU BLAS 实测 1 线程的 worker
  中评测 marker 后的设备驻留直线计算区。source/report/artifact 哈希、分段 AST 等价、当前 runtime、
  profile guard、实际调用次数、逐样本同步和命名输出数值任一门失败时，回执固定
  `performance_valid=false` 且不发布 speedup。
- 每次 script 迁移使用 write-once 输出目录，避免上一次的 `migrated.py` 或 gap request
  混入本次证据。省略 `--out` 时会自动建立新的 run 目录；显式目录若已有 managed 产物会拒绝覆盖。
- `check-suite`、`wizard`、`bench-cpu`、`bench-npu` 都是 legacy contract 工具，要求
  `build_inputs + forward`，不能用于证明 ordinary script 的已发布 configured-model artifact；
  ordinary 数值验证使用 `verify-script`，ordinary strict-NPU resident 性能验证使用独立的
  `benchmark-script`（`bench-script` 是同一命令的短别名）。
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
- 算子生成固定使用华为官方 `cann/cannbot-skills` 的已审计 commit
  [`1ef6372a538f39b0b7ed34666a3e1f465fb4936a`](https://gitcode.com/cann/cannbot-skills/tree/1ef6372a538f39b0b7ed34666a3e1f465fb4936a)。远端 Chat Completions
  模型无法读取本机路径，因此 NumPilot 会校验 checkout 的 origin/commit/工作树，将所需
  `SKILL.md`、指南和 910B 模板内容连同逐文件 SHA-256 真正嵌入请求，而不是只告诉模型一个
  `/home/...` 路径。
- 当前仓库已有 GapSpec、未验证 draft/repair/seal、AsNumpy adapter 生成/repair、qualification
  schema、capability overlay 原语，以及 schema-2 typed stage receipts/write-once hash-chain。
  `recover-gaps ingest-official` 已能严格消费 migration、GapSpec 和 official-resolution 三类收据，
  幂等推进到 `RESOLVED`；installed `official_binary` 可经 fresh resolver/rehash 到
  `CANDIDATE_READY`，reviewed `numpy.matmul` 可通过模型语义编辑与本地精确重建生成 schema-4
  未验证 overlay，并以 typed receipt 到 `ADAPTER_CANDIDATE`。`candidate-generation` profile
  的实现边界止于此，且禁止 system write/import/build/execution/activation；只有创建时即冻结为
  `full-local-recovery` 才能授权后续 producer，而 `integrate → qualify → activate → rescan → final
  migration` 当前仍未自动执行。任何一步失败都必须保留为可恢复状态；不得把 `prepared`、
  `generated_unverified` 或 adapter candidate 写成已经修复。

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

当前比赛主路径已经由产品内 `benchmark-script` 真机闭环验证，展示口径分为应用形态主结果和
合成优势区间，两者都不涉及 `matmul`，都经过 configured `glm-5.3` 实际远程迁移、exact AST、
named-output 数值门和 16/16 性能门：

- **应用形态主结果：** `examples/poisson_source_smooth_l1_map.py` 从 Poisson-Ritz 工作流抽出
  解析源项与 `SmoothL1(beta=1)` 未规约逐配点残差阶段。9,000,000 个 `float32` 元素、3 次预热、
  CPU/NPU 各 20 个样本，CPU 单核 resident 中位 `458.698 ms`、NPU `178.658 ms`，中位加速
  `2.567×`，保守 `CPU p10 / NPU p90 = 2.471×`，bootstrap 95% 区间
  `[2.538×, 2.592×]`。一次输入准备 + 一次 resident + 最终一次 D2H 的组件和为 CPU
  `759.784 ms`、NPU `963.263 ms`，诊断比 `0.789×`；该值明确
  `directly_sampled_end_to_end=false`，既不是直接 E2E，也证明首次单次调用不是 NPU 优势区。
  本阶段没有外部数据集、不是完整 Poisson 求解器，不得外推到完整 3000 轮案例。冻结文件见
  [Poisson 阶段开发态证据](evidence/competition-poisson-stage-20260828/README.md)。
- **合成优势区间：** `examples/competition_power_log_exp.py` 的 9,000,000 元素纯逐元素链得到
  CPU 单核 `616.333 ms`、NPU `76.289 ms`，resident 中位 `8.079×`，保守 `7.503×`，
  bootstrap 95% 区间 `[7.954×, 8.147×]`；一次输入准备 + resident 中位 + 最终 D2H 的组件和
  为 `1.220×`，同样不是直接采样的 E2E。它只说明大数组、设备驻留、重复复用输入且最后一次
  回传时的优势上界，不宣称真实行业数据。冻结文件见
  [合成 ordinary 开发态证据](evidence/competition-ordinary-20260828/README.md)。

两份证据都保存源码、实际模型迁移回执、验证回执、20×2 原始样本和目录 manifest。可用
[标准库只读审计器](tools/audit_script_benchmark_receipt.py)在不执行用户代码、不导入
NumPy/AsNumpy、不访问网络或 NPU 的情况下独立重算哈希绑定和统计。两次捕获均绑定
`git_dirty=true`，所以这里只称开发态真机证据，不冒充不可变 RC。

历史 [Demo v1 Case 01](evidence/demotest-case01-20260825/README.md) 的 `180×` 定向 harness 数字
仅保留作归档压力案例，不再作为主宣传：其冷进程 CPU 更快、没有 exact-artifact profiler，且
不能替代当前产品 ordinary benchmark。

当前 `/home/ma-user/work/demo` 的 7 份真实 GLM 迁移产物也完成了数值核验，其中 strict
Case03/06/10 完成分层性能实测：微型稳态脚本体在 NPU 上反而慢 `2.68×–5.28×`，fresh-process 慢
`19.58×–21.85×`；但冻结的 Case06 1M/4M/9M 常驻设备四算子 adapter 快
`9.47×–11.44×`。这说明该冻结四算子链在这些规模上的设备驻留吞吐正常，固定启动、公共 wrapper 深拷贝/同步和分配
成本是当前主要问题；adapter 不是迁移脚本 E2E，不能混写。详见
[分层性能分析](docs/evidence/day4-demo-performance-analysis-20260826.md)与
[机器可读汇总](docs/evidence/day4-demo-performance-results-20260826.json)。

## 快速运行

> 发布状态（2026-08-27）：通用模型、持久激活、qualified capability、ordinary
> analyze/migrate、official-first resolver 和 ops-nn MatMul 案例已在
> `feat/framework-hardening-opsnn-rc` 集成候选中收口；尚无远端不可变发布引用。
> `v0.2.0a2` 仍只是开发基线，**不包含**这些改动；当前源码版本为未发布的
> `0.3.0a1.dev0`，也不是 RC。正式发布时必须冻结新版本并把这里替换为新的不可变 tag。

全新机器的 direct migration 默认路径只需要克隆 NumPilot。AsNumpy 由安装器拉取固定官方
源码并现场编译；CANNBot 是遇到缺口后才使用的可选 Skill bundle，不是 direct lane 依赖。
默认 bootstrap 不自动构建或激活 ops-nn；official-first resolver 和经过独立资格化的定向算子
运行时属于显式 gap lane。默认路径不使用 wheelhouse：

```bash
cd /absolute/path/to/revision-containing-this-document
./install.sh --allow-network \
  --persistent-root /absolute/persistent/root
source /absolute/persistent/root/activate-numpilot.sh
```

需要 gap/operator lane 时，先审阅 CANN OSL，再给安装器增加
`--with-cannbot --accept-cannbot-license`。标准持久 HOME 可省略 `--persistent-root`；安装器仍会
打印生成的 `activate-numpilot.sh` 路径。

若机器默认 Python 包镜像不含所需版本，由用户显式指定可信索引，例如附加
`--pip-index-url https://pypi.org/simple`。安装器不会在失败后悄悄换源。

`install.sh` 只在用户显式传入 `--allow-network` 后工作，并依次创建项目虚拟环境、安装
`.[bootstrap]`、生成并 source 无凭据 activation、展示 plan、构建/安装 AsNumpy、按显式选择
准备 CANNBot、运行 smoke 与 `doctor --backend npu`。默认 receipt 写入
`numpilot paths --get bootstrap_receipt` 所示的 XDG
状态路径，也可显式传入 `--receipt PATH`。运行时 capability 层会重新核对锁定
source commit/tree、全部已审计 Python wrapper hash、当前 `_core.so` 与 receipt 的双份
hash、模块/解释器路径和实际包版本；任一漂移都回到 `CAPABILITY_AUDIT_REQUIRED`。
该 receipt 是当前用户边界内的本地 lineage/意外漂移门，不是带签名的供应链证明，不能
宣称可对抗同一用户或 root 主动伪造。
使用安装器生成的 activation script 时，自定义 receipt 和持久根会随 source 一并恢复；手工
bootstrap 才需要在后续 shell 自行设置同一 `NUMPILOT_BOOTSTRAP_RECEIPT`。
安装器不会运行 `sudo`、修改 shell 启动文件、保存代理/凭据或安装到
系统 Python。完整步骤及手工安装入口见
[全新机器源码验收路径](docs/FRESH-NOTEBOOK-QUICKSTART.md)。

主机仍需预装 CANN/驱动/NPU、Python 3.10+ 及开发头、Git、GCC 11.2+，以及系统
`fmt`/`spdlog` CMake 开发包；缺失时安装器会在拉取/构建前明确失败，不会静默提权。
默认 pinned AsNumpy 源码 profile 中 `matmul` 仍是已知 CPU fallback，只验证安装与兼容性迁移；
只有精确匹配 ops-nn MatMul 资格化回执和 runtime hash 的二维 `float32` 合同才进入 scoped
NPU 路径。两类运行时不能混用；本案例也固定 `performance_valid=false`，不得沿用 P0 wheel
的 180× 历史结果。

安装开发版本（这是用户显式执行的开发安装，不代表 `import` 或 `migrate` 可以联网）：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

基础运行依赖为 `numpy>=1.26`、`loguru>=0.7`、`rich>=13`。源码安装必须使用
`.[bootstrap]`，其中包含 NumPy 2、CMake、PEP 517/C++ 构建工具；不要同时安装
`.[demo,bootstrap]`，因为历史 P0 wheel 的 NumPy 1.26.4 profile 与官方 AsNumpy 源码的
NumPy 2 要求不同。`bootstrap --apply` 本身不会临时从 PyPI 猜测或补齐构建依赖，
一键脚本是在调用它之前显式安装 `.[bootstrap]`。
`--build-asnumpy` 会执行上游 CMake/PEP 517 构建，目前没有 OS 级断网沙箱；即使源码缓存已
存在，plan 也会保守标记该动作 `network=true`。

依赖准备使用显式 bootstrap 层。不传模式或显式传入 `--plan` 都是只读计划；
`--json` 可输出机器可读计划：

```bash
# 两者都只读：检查环境并展示网络/文件/构建动作，不应用
python -m numpilot bootstrap
python -m numpilot bootstrap --plan
python -m numpilot bootstrap --plan --json

# 可选高级路径：安装本地 wheel 前用可信发布物/用户输入的预期 hash 校验
python -m numpilot bootstrap --apply \
  --asnumpy-wheel /trusted/path/asnumpy.whl \
  --asnumpy-wheel-sha256 "$TRUSTED_ASNUMPY_SHA256"

# 默认 direct 源码路径：需先显式安装 numpilot[bootstrap]
python -m numpilot bootstrap --apply --build-asnumpy

# 可选 gap/operator lane：增加固定 CANNBot bundle
python -m numpilot bootstrap --apply --build-asnumpy \
  --with-cannbot --accept-cannbot-license

# 手工准备源码：两个 checkout 都必须处于 plan 所列的精确 commit，origin 正确且工作树干净
python -m numpilot bootstrap --apply --build-asnumpy \
  --asnumpy-cache /path/to/asnumpy \
  --with-cannbot --cannbot-cache /path/to/cannbot-skills \
  --accept-cannbot-license

# 安装后真实复验 AsNumpy README smoke（shape=(1000, 1000)，float32）
python -m numpilot bootstrap --verify

# 同时从已固定的本地 bundle 回答 Ascend C Skill 清单；不联网
python -m numpilot bootstrap --verify --with-cannbot --accept-cannbot-license
```

以上命令默认写入 NumPilot 自己的 XDG state 目录。若高级流程需要 `--receipt PATH`，PATH
必须是绝对路径，父目录须由当前用户拥有、不可为符号链接且不可被 group/world 写入；例如先
执行 `install -d -m 700 "$HOME/.local/state/numpilot-manual"`，再把 receipt 放入该目录。
后续 shell 还需把 `NUMPILOT_BOOTSTRAP_RECEIPT` 设置为同一绝对路径。

如果用户已经完全自行编译安装 AsNumpy，可运行 `bootstrap --verify`，并用
`setup-skills --root /path/to/cannbot-skills --check` 校验手工 CANNBot checkout。这只能证明
当前安装能运行，不能追溯证明二进制来自指定源码；需要完整来源回执时应使用上面的
`--asnumpy-cache` 构建路径。

实现内锁定 official AsNumpy source commit/tree 和 official CANNBot commit/bundle 身份，
但不内置 wheel URL 或预期 wheel hash。本地 wheel 的预期 SHA-256 必须来自可信
发布物或用户输入，bootstrap 会在安装前校验并将实际 hash 写入 receipt；运行时
资格 profile 仍由后续 capability 审计绑定。`--apply` 必须在“本地 wheel + expected
SHA-256”和 `--build-asnumpy` 之间二选一。`--asnumpy-cache` 与 `--cannbot-cache`
可显式指定锁定 checkout 缓存。只有选择 `--with-cannbot` 获取或复验 CANNBot 内容时才
必须同时传入 `--accept-cannbot-license`；单独准备 AsNumpy 不要求 CANN OSL。
NumPilot 不得由 pip 安装钩子、`import`、`doctor`、`migrate` 或 `complete-gaps`
隐式下载依赖。ordinary `migrate` 为生成迁移代码会调用已配置的 endpoint，这是该命令
明确的模型动作，不是依赖安装；调用必须保留回执，并只在进程内临时使用用户代理。

普通纯 NumPy 脚本的最短路径：

```bash
python -m numpilot configure-llm --check
python -m numpilot migrate /path/to/case.py \
  --backend npu --out /tmp/numpilot-case-direct

# 这是单独的显式代码执行步骤；result 必须由两侧脚本赋值
python -m numpilot verify-script /path/to/case.py \
  --migration-report /tmp/numpilot-case-direct/script-migration-report.json \
  --output result \
  --out /tmp/numpilot-case-direct/script-verification-report.json

# 可选性能步骤：源码与 GLM 产物都必须有且仅有一个显式 resident marker
python -m numpilot benchmark-script /path/to/case.py \
  --migration-report /tmp/numpilot-case-direct/script-migration-report.json \
  --output result \
  --out /tmp/numpilot-case-direct/script-benchmark-report.json
```

`verify-script` 只接受 report 同目录、SHA-256 一致的 `migrated.py`，不会解析 stdout，也不会
使用 pickle。不同环境可分别传 `--source-python` 与 `--migrated-python`；目标依赖必须已经安装
在对应解释器环境中。完整合同、退出码和可信边界见
[ordinary script 数值验证](docs/script-verification.md)与
[ordinary strict-NPU resident benchmark](docs/script-benchmark.md)。

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
`build_inputs + forward` contract run；不能替代普通脚本已发布 configured-model artifact 的哈希绑定定向
`benchmark-script` 证据。

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
默认案例是 ordinary 大规模 `float32` Poisson-Ritz 源项与鲁棒残差阶段；脚本固定执行
`doctor → analyze --require-direct → configured GLM migrate → verify-script → benchmark-script`。
现场可用 `./run_demo.sh --quick` 运行 2 次预热 + 7 个样本，正式证据使用默认 20 个样本。

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
| [`examples/ops_nn_matmul_inference.py`](examples/ops_nn_matmul_inference.py) | `numpy.matmul` 缺口、官方候选复用、AsNumPy 接入与迁移闭环 | 仅精确资格化的二维非空 `float32` 合同为 scoped NPU；[真机证据](evidence/ops-nn-matmul-20260827/README.md)不发布性能结论 |
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

可编辑架构图：
[当前整体架构图](docs/numpilot-architecture.drawio)；
[官方算子解析与缺口恢复架构图](docs/numpilot-official-operator-resolution.drawio)；
[官方算子复用实现状态](docs/OPS-NN-REUSE-ASSESSMENT-20260825.md)。

```text
NumPilot bootstrap（默认 plan）/ --apply   显式、lock-bound 依赖准备
                    │
                    ▼
NumPilot migrate（ordinary script 默认）
  静态扫描 → CapabilityManifest → 五类确定结果
    ├─ DIRECT                    → configured model 精确迁移 → AST/receipt 门 → 落盘
    ├─ FALLBACK_CONSENT          → 用户接受 / 拒绝
    ├─ GAP_REQUIRED              → GapSpec → installed ACLNN → pinned ops-nn 只读收据门禁
    │                                ├─ 精确匹配 → 官方候选 + model semantic edits
    │                                │              → 本地精确重建未验证 adapter candidate
    │                                └─ 两层 clean NO_MATCH → CANNBot + model kernel 草稿
    ├─ CAPABILITY_AUDIT_REQUIRED → 资格或源码审计；禁止盲目生成
    └─ BLOCKED                   → 原因落盘

NumPilot verify-script（迁移后的独立显式执行）
  source/report/migrated hash 门 → 两个隔离子进程 → 明确命名输出数值比较 → write-once receipt
  固定边界：strict_npu_verified=false · performance_valid=false · os_sandbox=false

Gap recovery control plane（已实现确定性状态基础）
  source/target/runtime/policy binding → typed stage receipts → write-once SHA-256 chain
  ingest-official：migration report → GapSpec → official resolution → RESOLVED / resume
  installed official_binary：fresh resolver + rehash → CANDIDATE_READY
  reviewed MatMul：per-target hash-bound projection → model exact edits → local rebuild
                   → schema-4 receipt ingest → ADAPTER_CANDIDATE
  三种冻结 profile：receipt-only / candidate-generation / full-local-recovery
  candidate-generation 只到 ADAPTER_CANDIDATE；仅 full-local 可授权后续本地动作
  授权不等于执行；当前仍未自动 integrate / build / qualify / activate

NumPilot migrate --mode contract（显式 legacy）
  AnalysisAgent → whole_forward_npu → MigrationAgent → VerificationAgent

注意：gap 自动 integrate/qualify/activate/rescan 编排尚未实现；ordinary `benchmark-script`
首版只承诺显式 marker 后的设备驻留直线计算区，不是任意 Python 或冷启动端到端 benchmark。
```

核心结构：

- `numpilot/contracts.py`：智能体之间的数据契约；
- `numpilot/script_migration.py`：普通脚本静态扫描、能力分类、configured-model 精确改写审计与产物报告；
- `numpilot/script_verification.py`：ordinary 迁移产物的显式、hash-bound 双进程数值验证与
  write-once 回执；私有 worker 为 `_script_verification_worker.py`；
- `numpilot/script_benchmark.py`：ordinary strict GLM 产物的单核 CPU/NPU resident 性能父门、
  统计与 combined receipt；私有 worker 为 `_script_benchmark_worker.py`；
- `numpilot/capabilities.py`：可序列化 capability/provenance；
- `numpilot/comparison.py`：统一严格比较规则；
- `numpilot/engines/`：AST/LLM 迁移策略；
- `numpilot/backends/`：shim/NPU 端口适配器；
- `numpilot/bench.py`：legacy contract 基线与 fail-closed NPU 评测；
- `numpilot/gaps/`：缺口规格、隔离生成、哈希 manifest、离线 qualification evidence 准入
  与原子 capability overlay；`NpuBackend` 可显式加载 overlay，并以 source/callsite profile
  和每次真实调用的 dtype/rank/C-contiguous guard 双重失败关闭；其中 `gaps/recovery.py`
  提供不可变 binding、typed stage reducer、write-once 状态链和 route-neutral operator
  candidate 合同，`gaps/recovery_ingest.py` 负责前三段 producer receipts 的严格导入与续接；
- `docs/numpilot-technical-review.md`：完整技术评审和分阶段计划。

当前保留 `--threshold` 仅为 CLI/契约兼容；`whole_forward_npu` 策略不会再用它制造不完整的逐算子混合 placement。

## LLM 引擎

推荐先生成本机私有配置模板，再用编辑器手动填写：

```bash
python -m numpilot configure-llm --init
# 打开上一条命令打印的 llm.json，填写 base_url、auth_token、model
python -m numpilot configure-llm --check
python -m numpilot configure-llm --probe
./run_demo.sh --quick
```

`--init` 会生成 schema 2 JSON，包含 `base_url`、`auth_token`、`model`、审计用
`provider`、固定的 `protocol=openai-chat-completions` 和
`response_model_aliases`，且不会覆盖已有配置。`base_url` 应包含兼容 API 的 `/v1` 前缀，
不要包含 `/chat/completions`。也可运行 `python -m numpilot configure-llm` 使用隐藏 Token 输入
的交互模式。

`--check` 只做本地配置完整性检查，**不联网**；`--probe` 才会发起一次真实的轻量请求。
探针成功会打印 `actual_remote_call=yes`、HTTP 状态、TLS 校验状态、请求/响应模型和内容
SHA-256，但不会打印 Token、Base URL 或响应正文；未成功取得可验证响应时返回非零退出码并明确
打印失败。

配置模型必须与请求 body 的 `model` 精确一致；响应 `model` 默认也必须精确一致。网关确实
返回别名时，只能把该精确别名显式列入 `response_model_aliases`，决策和 identity hash 会写入
receipt。缺失、不匹配或被篡改时 fail closed，且不会回退本地 AST。`glm-5.3` / `provider=glm`
只是可选配置示例，不是运行时授权条件。完整合同见
[通用模型与回执合同](docs/model-contract.md)。

框架默认使用 `$XDG_CONFIG_HOME/numpilot/llm.json`（通常为 `~/.config/numpilot/llm.json`），目录权限 `0700`、文件权限 `0600`。临时容器或只持久化特定目录的平台，应先把 `NUMPILOT_CONFIG_HOME` 指向仓库外的持久目录，再执行 `--init`；后续运行也要保持该环境变量一致。真实凭据不属于仓库内容，永远不得提交到开源仓库。

CI/流水线无需交互，可继续注入 `NUMPILOT_LLM_BASE_URL`、`NUMPILOT_LLM_AUTH_TOKEN`、`NUMPILOT_LLM_MODEL`；环境变量逐字段优先于用户配置。也可用 `NUMPILOT_CONFIG_HOME` 或 `NUMPILOT_LLM_CONFIG_FILE` 覆盖默认配置位置。`--engine llm` 使用 strict 语义：未配置、调用失败或返回非法代码会退出，不回退 AST。TLS 证书默认校验；只有受信内网自签网关才应显式设置 `NUMPILOT_LLM_INSECURE=1`。

## 缺口算子补全

新 kernel 生成前的只读 official-first 解析与收据门禁已经实现：先检查已安装 CANN ACLNN，
再读取匹配 CANN 版本、固定 commit/tree 的本地
[`cann/ops-nn`](https://gitcode.com/cann/ops-nn) Git object。`resolve-gaps` 不联网、不调用模型、
不读取 CANNBot Skill、不构建或写 OPP；`complete-gaps --official-resolution-report` 会在模型
readiness 前复验 hash-bound 收据并重新盘点当前官方证据，只有两层均 clean `no_match` 且
`resolution_id` 未漂移才允许新 kernel。有 Gap 的 `--provider llm` 未提供收据时会在模型和
Skill 前失败关闭；无 Gap 时直接本地结束，不进行模型 readiness 探测。

候选仍必须逐项证明 GapSpec 语义一致。installed `official_binary` 的 route-neutral candidate、
reviewed MatMul `official_aclnn_existing_module` adapter planner/model gate 和 typed receipt 已接线；
`official_source`/`remote_llm` Recovery candidate、定向构建/私有 OPP 回滚、通用 adapter、六门
资格化和激活仍未完成。详见
[ops-nn 官方算子复用评估](docs/OPS-NN-REUSE-ASSESSMENT-20260825.md)。
复用的官方 kernel 必须保留 official provenance/hash/license，不会被标记成模型生成代码；
所有新迁移、算子和 adapter 都使用同一 configured-model identity/receipt 合同。

现有 adapter generation/repair 已修正为先只读识别目标 AsNumpy 的
`manifest_generator`、`legacy_manual` 或 reviewed `official_aclnn_existing_module` 布局。
固定 `1929a3...` 源树会使用真实的手工
C++/pybind/Python/CMake 调用链作为 hash-bound 上下文，不再硬读该版本不存在的 manifest
generator 文件；布局或 signed artifact 无效时会在任何模型请求之前停止。
生成/修复报告 schema 4 还会把 architecture、参考根目录、context hashes、完整 overlay 路径集
及 repair parent 绑定在一起；切换布局或修改参考源码后必须重新生成，不能沿用旧 adapter 修复链。

reviewed official existing-module generation 还使用 schema-2 context plan 的
`prompt_dependencies`。完整 reviewed context 继续在本地逐文件哈希，模型的每个目标请求只投影
该目标允许的 `reference`、`signed_official_header` 或先前 `generated_overlay` 依赖，并携带
content-free projection manifest 与 `projection_sha256`；远端回执的 request-body hash 继续绑定
实际请求。模型只返回 `NUMPILOT_EXACT_EDIT_V1` 的最小 OLD→NEW 语义编辑，不返回完整文件；
本地验证目标路径、base SHA-256、OLD 唯一性和区间不重叠后机械重建完整文件。schema-4 report
和后续 Recovery receipt 绑定的是重建后的完整 overlay 及其哈希，不是未经验证的模型文本。
MatMul official route 还将可编辑范围冻结为 target edit windows：`product.cpp` 只能修改
Matmul 函数，`test_product.py` 只能修改 imports/helpers 和 matmul 测试段；任一 OLD
片段跨越窗口边界都会失败关闭。真实 GLM exact-route 验收使用严格 SSE 事件聚合，
请求显式禁用 thinking 并设置 minimal reasoning；只有聚合后 `content` 为空时才把
响应预算从 8192 提升到 16384 重试。本轮得到 5 份真实成功远程收据，状态仅到
`ADAPTER_CANDIDATE`；未构建、未执行、未激活。

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
  --target-soc Ascend910B4 --cann-version 9.0.0
```

显式增加 `--provider llm` 前，必须先生成并在调用时提供 fresh 的官方算子解析收据：

```bash
GAP_RUNS="$(python -m numpilot paths --get gaps)"
OFFICIAL_RECEIPT="$GAP_RUNS/cbrt-official-resolution.json"

python -m numpilot resolve-gaps examples/gap_cbrt.py \
  --out "$OFFICIAL_RECEIPT" \
  --target-soc Ascend910B4 --cann-version 9.0.0 \
  --cann-root /absolute/path/to/cann-9.0.0 \
  --ops-nn-root /absolute/path/to/ops-nn \
  --accept-cann-osl

python -m numpilot complete-gaps examples/gap_cbrt.py \
  --out "$GAP_RUNS/cbrt" \
  --target-soc Ascend910B4 --cann-version 9.0.0 \
  --provider llm --official-resolution-report "$OFFICIAL_RECEIPT" \
  --max-attempts 3
```

对已经产出的 gap migration report、其 sibling `gap-request.json` 和 official resolution，可把
前三段真实收据导入可恢复状态链。该命令只做离线收据接线，不会重复运行 resolver，也不会调用
模型、构建或激活：

```bash
python -m numpilot recover-gaps ingest-official /absolute/path/to/source.py \
  --migration-report /absolute/private/run/script-migration-report.json \
  --gap-request /absolute/private/run/gap-request.json \
  --official-resolution-report /absolute/private/run/official-resolution.json \
  --state-dir /absolute/private/run/recovery \
  --authorization-profile candidate-generation \
  --json

python -m numpilot recover-gaps materialize-candidates \
  --state-dir /absolute/private/run/recovery \
  --out /absolute/private/run/candidates \
  --json

python -m numpilot generate-asnumpy-adapter \
  --candidate /absolute/private/run/candidates/gaps/GAP/operator-candidate.json \
  --recovery-state-dir /absolute/private/run/recovery \
  --asnumpy-reference /absolute/path/to/pinned/asnumpy \
  --out /absolute/private/run/adapter

# 生成成功会自动导入；发布后中断时可幂等重放
python -m numpilot recover-gaps ingest-adapter \
  --state-dir /absolute/private/run/recovery \
  --candidate /absolute/private/run/candidates/gaps/GAP/operator-candidate.json \
  --adapter-output /absolute/private/run/adapter \
  --json

python -m numpilot recover-gaps status \
  --state-dir /absolute/private/run/recovery \
  --json
```

默认 `receipt-only` run 不能事后升级；要继续物化/调用模型，必须在首次导入时选择命名 profile。
到达 `CANDIDATE_READY` 只证明 official candidate 已重新盘点并绑定；到达
`ADAPTER_CANDIDATE` 也只证明 schema-4 `generated_unverified` overlay 已被 typed receipt 绑定。
两者都不表示 `INTEGRATED_CANDIDATE`、`QUALIFIED` 或 `ACTIVATED`，更不表示 NPU 数值/性能成立。
上例使用 `candidate-generation`，所以该 run 不能继续执行 system write、AsNumpy import、build、
execution 或 activation；这些动作只有首次导入即选择的 `full-local-recovery` 能授权，且相应
integration/qualification/activation producer 当前仍待接线。profile 创建后不可原地升级。

当前 ACLNN catalog 只审计了精确的 `Ascend910B4` profile。`Ascend910B` 是 family 名称，
不会被推断成 B4；其他精确型号也必须先增加独立审计。若上述解析命中 installed ACLNN 或
ops-nn candidate（例如当前环境中的 `numpy.cbrt`），第二条命令会按设计在模型前停止，
不会绕过官方候选去手写 kernel。

当前远端草稿协议使用 `NUMPILOT_BUNDLE_V1` 行式信封承载原始 C++/Python 内容，避免部分
OpenAI 兼容网关不能可靠强制 JSON 字符串转义的问题；路径穿越、目录白名单、文件数、总
大小、UTF-8、必需工程结构和最终哈希仍由本地校验器失败关闭。这个协议只改变传输封装，
不降低后续 build/UT/数值/NPU/activation 门禁。
该完整文件信封用于 new-kernel 和普通 complete-file adapter route；reviewed
`official_aclnn_existing_module` generation 改用上述 `NUMPILOT_EXACT_EDIT_V1`，不能把两种协议
混写。

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
- ordinary configured-model artifact 的受限 strict-NPU resident runner 已由
  `benchmark-script` 实现；仍缺通用 cold/warm E2E、多核 CPU、自动 profiler/kernel trace
  和任意 Python 数据流性能 runner，现有 headline 不会外推到这些范围；
- installed `official_binary` candidate 与 reviewed MatMul adapter receipt 已接入；仍缺
  `official_source`/`remote_llm` candidate、其他 API/route adapter，以及 integrate、qualify、
  activate、rescan、最终 migration/verification 的执行与收据编排；
- official-source Artifact、逐字段语义审计、私有 OPP 安装事务与回滚，以及候选到
  adapter/qualification/activation 的统一闭环；
- 通用 ndarray 结构语义、异步 stream、内存池以及 AsNumpy 插件 ABI。

这些边界不会被 shim 通过率或“替换了多少 API”掩盖。
