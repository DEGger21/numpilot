# NumPilot

NumPilot 是一个面向科学计算程序的 NumPy → AsNumpy/Ascend NPU 可审计迁移工具。当前版本
强调“结果与证据可信”：它把顶层 `forward()` 作为一个完整 NPU 岛，在函数边界上传 NumPy
数组，运行迁移产物自己的 `build_inputs + forward`，并在发布性能数字前检查实际 AsNumpy
wheel 的 capability/provenance。

完整愿景见 `NumPilot20260615-v1.pdf`。仓库现已实现缺口 API → GapSpec → 真实 LLM Ascend C 隔离草稿的安全纵切；编译、NPU 数值校验和激活仍受独立门禁约束，不会把“生成了源码”冒充成“算子已经可用”。

开发者从 [贡献指南](CONTRIBUTING.md) 开始，评委可直接查看
[证据索引](evidence/README.md) 与 [可信边界](#当前可信边界)；完整文档导航见
[`docs/README.md`](docs/README.md)。项目源码采用 [Apache-2.0](LICENSE)，第三方工具链与
生成算子的许可证边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 当前可信边界

- 只分析和改写顶层 `forward()`；`build_inputs()` 与顶层展示代码保留在 CPU。
- 默认采用 `whole_forward_npu`，暂不宣称支持逐算子 CPU/NPU 混合卸载。
- AST 引擎在 `forward` 入口对 NumPy ndarray 做一次幂等 H2D；设备数组和标量保持原样。
- 验证先检查嵌套结构、shape、dtype，再按实际浮点 dtype 使用 `atol + rtol`；整数/布尔精确比较，显式处理零值、NaN 和 Inf。
- 真机 capability 绑定 AsNumpy 版本和运行环境；CPU fallback、版本未知或 API 缺失时 fail closed。
- strict `forward` 禁止残留 NumPy/host escape、未建模 helper/控制流、全局预计算和丢弃的设备结果；两条验证路径都记录实际 `ap.*` callsite 次数。
- `bench-npu` 仅在 strict-NPU、数值和 CPU 基线一致性门禁全部通过后计算 speedup。
- `--allow-cpu-fallback` 只用于诊断，报告始终为 `performance_valid=false`，不会发布 NPU 加速比。
- 缺口补全接受两类显式来源：确实不在静态覆盖集的直接 `numpy.<api>`，以及通过
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
- 定制算子的默认路线是“扫描缺口 → 依 CANNBot Skill 直接生成本地候选”；首轮不
  拉取、复制或依赖 ops-nn。外部算子仓库只是后续人工对照/回退选项。

已知重要事实：通用基线审计的 AsNumpy `0.3.1.dev4+g1929a3ba4` 中，`matmul` 会回到
host 调用 NumPy。因此 `mlp_forward.py` 和 `big_gemm.py` 仍只能用于迁移/诊断展示，不能
作为 NPU GEMM 加速证据。Poisson P0 另有一个文件与二进制哈希绑定的 v9 wheel：它只在
float32、非空、rank 1/2 的已测范围内放行修复后的 `outer`/`matmul` ACLNN 路径，不能
反向扩大通用基线能力声明。

## 快速运行

安装开发版本：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

离线 CI 使用 numpy shim，不需要 NPU：

```bash
python -m numpilot migrate examples/mlp_forward.py \
  --backend shim --out /tmp/numpilot-shim
python -m pytest -q
```

裸金属 NPU 上，建议先使用没有已知 fallback 的逐元素 smoke：

```bash
python -m numpilot doctor --backend npu
python -m numpilot migrate examples/elementwise_forward.py \
  --backend npu --out /tmp/numpilot-npu

python -m numpilot bench-cpu examples/elementwise_forward.py \
  --out /tmp/numpilot-bench --cpu-threads 1
python -m numpilot bench-npu examples/elementwise_forward.py \
  --out /tmp/numpilot-bench
```

性能结论只读取：

- `/tmp/numpilot-bench/npu_benchmark.json`
- `/tmp/numpilot-bench/benchmark_report.md`

其中必须满足 `performance_valid=true`，才可以引用 `speedup`。

默认运行产物不会写入 Git 工作树。执行以下命令查看配置、缓存、报告和运行目录：

```bash
python -m numpilot paths
```

默认状态根为 `$XDG_STATE_HOME/numpilot`（通常是 `~/.local/state/numpilot`）；可用
`NUMPILOT_STATE_HOME`、`NUMPILOT_RUNS_DIR`、`NUMPILOT_REPORTS_DIR` 覆盖。所有原有
`--out` 参数继续有效，适合 CI 和比赛评测显式指定一次性目录。详细策略见
[`docs/artifact-storage.md`](docs/artifact-storage.md)。

真实科学计算网络的完整一次实测见
[`docs/evidence/heat-diffusion-npu-20260813.md`](docs/evidence/heat-diffusion-npu-20260813.md)：
二维热扩散单步网络的 7 个调用全部迁移并通过 strict-NPU 与零误差门禁；报告同时保留了
计时口径和当前能力边界。

也可直接运行裸金属演示脚本：

```bash
NUMPILOT_PY=python3 ./run_demo.sh
```

脚本使用 `set -euo pipefail`；任何 doctor、迁移或 strict-NPU 门禁失败都会停止，不会继续打印“演示成功”。默认不使用 Docker。

## 示例

| 文件 | 用途 | strict-NPU 预期 |
|---|---|---|
| `examples/elementwise_forward.py` | square/add/sqrt/tanh 的裸金属 smoke | 当前审计无已知 fallback，仍以运行时 manifest 为准 |
| [`examples/heat_diffusion_forward.py`](examples/heat_diffusion_forward.py) | 1024×1024 二维热扩散单步网络 | 已有 [910B4 实测证据](docs/evidence/heat-diffusion-npu-20260813.md)，7 个逐元素调用 strict-NPU 通过 |
| [`examples/poisson_ritz_step.py`](examples/poisson_ritz_step.py) | 确定性 Poisson/RBF 损失与一次参数更新，含两个显式领域算子 | 两算子已通过六门资格化；显式加载 overlay 后，9 次调用 strict-NPU 单步通过（[证据](docs/evidence/poisson-ritz-p0-npu-20260821.md)） |
| [`examples/poisson_ritz_iteration.py`](examples/poisson_ritz_iteration.py) | RBF、Laplacian、损失、解析梯度和更新均在一个动态 `forward` | 117 个调用点完成 strict-NPU；同一产物已完成 250 轮设备驻留对拍，29,250 次实际调用零 profile 拒绝（[证据](docs/evidence/poisson-ritz-p0-npu-20260821.md)） |
| `examples/mlp_forward.py` | 两层 MLP 的迁移与正确性示例 | 因 `matmul` CPU fallback 被拒绝性能发布 |
| `examples/big_gemm.py` | 历史 GEMM 压测输入 | 只允许诊断，不是 NPU GEMM 加速证据 |
| `examples/unsupported_demo.py` | `np.fft.fft` 缺口负例 | 分析阶段 early stop |
| `examples/gap_cbrt.py` | 直接缺口 API 的 GapSpec/LLM 草稿示例 | 只生成未验证草稿，不进入 strict-NPU |

## 交互向导

```bash
python -m numpilot wizard
python -m numpilot wizard examples/elementwise_forward.py --engine ast --backend npu
```

向导展示分析、整段 NPU 岛决策、迁移、产物级语义校验和 capability 证据门禁。它不在交互进程内顺手计算加速比，而是给出独立 `bench-cpu`/`bench-npu` 命令。

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
CodeArts AgentTeam / 项目级 Agents / Skill
                      │
                      ▼
Orchestrator
  ├─ AnalysisAgent      只分析 forward 调用点
  ├─ DecisionAgent      whole_forward_npu
  ├─ MigrationAgent     AST（默认）或 strict LLM
  └─ VerificationAgent  artifact execution + strict comparison
                      │
       ExecutionBackend + CapabilityManifest
             ├─ ShimBackend（离线语义仿真）
             └─ NpuBackend（真实 AsNumpy/CANN/NPU）
```

核心结构：

- `numpilot/contracts.py`：智能体之间的数据契约；
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

框架默认使用 `$XDG_CONFIG_HOME/numpilot/llm.json`（通常为 `~/.config/numpilot/llm.json`），目录权限 `0700`、文件权限 `0600`。临时容器或只持久化特定目录的平台，应先把 `NUMPILOT_CONFIG_HOME` 指向仓库外的持久目录，再执行 `--init`；后续运行也要保持该环境变量一致。真实凭据不属于仓库内容，永远不得提交到开源仓库。

CI/流水线无需交互，可继续注入 `NUMPILOT_LLM_BASE_URL`、`NUMPILOT_LLM_AUTH_TOKEN`、`NUMPILOT_LLM_MODEL`；环境变量逐字段优先于用户配置。也可用 `NUMPILOT_CONFIG_HOME` 或 `NUMPILOT_LLM_CONFIG_FILE` 覆盖默认配置位置。`--engine llm` 使用 strict 语义：未配置、调用失败或返回非法代码会退出，不回退 AST。TLS 证书默认校验；只有受信内网自签网关才应显式设置 `NUMPILOT_LLM_INSECURE=1`。

## 缺口算子补全

NumPilot 固定华为官方 CANNBot Skills 的版本。可以先显式准备并查看来源：

```bash
python -m numpilot setup-skills
python -m numpilot setup-skills --check
```

默认缓存位于 `~/.cache/numpilot/cannbot-skills/<commit>`，可用
`NUMPILOT_CANNBOT_CACHE` 改到其他仓库外目录。`git clone numpilot` 本身不会执行网络脚本；
首次运行 `setup-skills` 或 `complete-gaps` 时才会获取固定 commit，并严格核对官方 origin、
commit 和被选文件是否有本地修改。离线环境使用 `--offline-skills`，缓存缺失时会失败而不是
悄悄省略 Skill。已有 checkout 也可显式传 `--skill-root /path/to/cannbot-skills`。

第一步只生成确定性的 GapSpec 和生成契约，不调用大模型（Skill 缓存缺失时可能获取上面的
固定官方 commit；完全离线时加 `--offline-skills`）：

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

草稿不会自动执行 `build.sh`、写系统 CANN、修改 `~/.bashrc`、安装 OPP 或注册 AsNumpy。当前机器若 `doctor --backend npu` 未同时通过 NPU 设备与 AsNumpy 检查，只能静态审查草稿，不能宣称补全闭环完成。

## 仍未完成

- callsite/value 级 IR、动态 shape guard 和真正的 CPU/NPU 子图划分；
- OS 级用户代码隔离与完整网络/文件权限收敛；
- H2D/D2H 字节数、kernel trace 和 profiler 自动采集；
- 通用化的多算子 qualification runner、安装事务与自动回滚；Poisson P0 已有受信外部 harness 完成构建、UT、NPU 数值、私有安装和 overlay 激活闭环；
- 通用 ndarray 结构语义、异步 stream、内存池以及 AsNumpy 插件 ABI。

这些边界不会被 shim 通过率或“替换了多少 API”掩盖。
