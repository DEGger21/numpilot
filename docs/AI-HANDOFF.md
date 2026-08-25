# NumPilot AI 续接手册

## 当前续接摘要（2026-08-25）

权威工作目录是 `/home/ma-user/work/numpilot`，当前分支是 `feat/numpilot-demo`。普通脚本
MVP 修改目前仍在工作树中，尚未提交或推送；重启后先用 `git status --short` 核对，不要使用
`git reset --hard` 或 `git checkout --` 覆盖工作树。用户已经明确要求按案例逐个开始迁移；当前只
完成第 01 例，继续前先确认没有重复处理已有 run，也不要越过用户直接批量处理后续案例。

### 2026-08-25 普通 NumPy 脚本 MVP（最新检查点）

- 用户已把主场景收敛为纯 Python + `import numpy as np` 的普通 NumPy 文件；`migrate` 的默认
  产品入口是 ordinary script，不再要求 `build_inputs()/forward()`。旧通道只有显式
  `--mode contract` 才进入；`--mode auto` 仅为旧调用兼容，新的 Demo 不得依赖自动探测。
- 新主流程已经落地：全文件惰性 AST 扫描 → 实际 AsNumpy capability 分类 → 全部可用时由
  GLM 做精确 `numpy/np → asnumpy/ap` 改写；已知 fallback 必须由用户 accept/reject；缺失
  API 或 reject 会生成源码/调用点/SoC/CANN 绑定的 GapSpec，再接现有 `complete-gaps`。
- 对外能力结果统一表述为 `DIRECT`、`FALLBACK_CONSENT`、`GAP_REQUIRED`、
  `CAPABILITY_AUDIT_REQUIRED`、`BLOCKED`。只有 missing API 或用户拒绝 audited fallback 才能
  生成 gap；callable + `UNKNOWN`、runtime/hash 漂移或存在但未审计的路径必须先做资格/源码
  审计，绝不能盲目生成同名替代算子。
- 普通脚本产物只允许 GLM provider 和 `glm-*` model。模型回复必须是单一 Python code
  block，并携带完整 14 字段远程回执；发布前会再次验证 model、内容 hash、完整 AST 和注释。
  本地 AST 只能作为预期树，永远不能作为迁移产物或失败回退。
- requested/configured/response model 必须是同一个 `glm-*`；同一硬门已覆盖算子 generation、
  repair、adapter 和旧远程草稿复用。contract 默认使用 LLM，AST 仅保留给框架离线回归。
- 普通脚本在调用 GLM 前还会 fail closed：未审计的 Python 数组运算符、ndarray method 和
  不满足现有 AsNumpy call contract 的签名都不能被误报为 direct；这是静态阻断，不会执行源码。
- 普通脚本的三个 managed 输出实行 write-once；省略 `--out` 会新建 run 目录，显式目录中
  若已存在迁移、gap request 或报告会在零修改前拒绝，不能留下互相矛盾的旧产物。
- `complete-gaps --gap-request ...` 会在任何 LLM readiness、生成或输出前重新扫描源码，
  本地重建 canonical GapSpec，并拒绝 request 注入、源码/签名/目标漂移、布尔 schema、重复
  JSON key、shim replacement 和 unknown provenance。混合缺口会保留 blocked 并返回 1。
- 当前 AsNumpy `master@1929a3b` 没有任意缺失 API 的通用 NumPy fallback；明确审计的
  fallback 只有 `inner/matmul/outer`（CPU）、`dot/vdot/floor`（conditional CPU）和
  `signbit`（host assisted）。`UNKNOWN` 不得询问成“可接受 fallback”。
- replacement gap 目前只自动完成“准备/生成未验证草稿”的安全接线。仓库另有 adapter
  generation/repair、qualification schema 和 overlay 原语，但没有统一的
  `integrate → qualify → activate → rescan → final GLM migration` 自动编排。旧 CPU fallback
  runtime 不能被 qualification overlay 直接覆盖；不能把 prepared/generated、adapter candidate
  或 schema 存在写成已修复。
- 显式依赖准备层已落地：`numpilot bootstrap` 无模式和 `--plan` 均为只读
  plan，`--json` 输出机器可读计划；只有 `--apply [--receipt PATH]` 会修改环境。
  apply 必须在 `--asnumpy-wheel PATH --asnumpy-wheel-sha256 HEX` 和 `--build-asnumpy`
  之间二选一，缓存可由 `--asnumpy-cache PATH`/`--cannbot-cache PATH` 显式定位。
- official AsNumpy source commit/tree 和 CANNBot commit/bundle 身份固定在实现中。
  实现不内置 wheel URL/hash；本地 wheel expected hash 由可信发布物/用户输入提供，
  校验后将实际 hash 写入 receipt，资格 profile 仍在 capability 层绑定。普通 Demo
  优先预编译 wheel，源码构建只有显式 `--build-asnumpy` 才允许。
  预编译 wheel 路径普通安装即可；源码构建前需用户先安装 `numpilot[bootstrap]`
  （仓库开发为 `.[dev,bootstrap]`），apply 不会自动从 PyPI 获取构建工具。
  源码构建子进程没有 OS 级断网沙箱，当前 plan 会保守把 build 标为 `network=true`；新的
  Notebook Demo 应走本地 P0 wheel，不要现场构建。
- 只有显式 `--with-cannbot` 获取或复验 CANNBot 内容时才必须同时给出
  `--accept-cannbot-license`；单独准备 AsNumpy 不要求 CANN OSL。pip 安装钩子、import、
  doctor、migrate、complete-gaps 均不得隐式下载依赖。ordinary migrate 的 GLM 请求是明确
  模型动作，仍必须保留回执并在调用进程临时注入用户代理。
- `bootstrap --verify` 已落地：AsNumpy 复验真实执行 README 中
  `ap.ones((1000, 1000), dtype=ap.float32)` 的 shape smoke，并核对版本、模块路径与 native
  hash；`--verify --with-cannbot --accept-cannbot-license` 从固定本地 bundle 回答“现在写
  Ascend C 算子有哪些能参考的 Skill？”并输出 8 项已校验清单，不联网、不调用模型。
- 2026-08-25 在 Case 01 P0 venv 和干净 CANN 9.0.0 library 顺序中实测上述组合验证通过：
  AsNumpy=`0.1.dev106+g21368abf9.d20260822`、NumPy=`1.26.4`、Loguru=`0.7.3`，
  `__init__.py` SHA-256=`4f4e8083755dde87bd6f0dc8c765da71984cb1346c97e8b6dbb97d714be34c04`，
  native `_core` SHA-256=`4465ce6e0589eb000d450a49c15488a4d2d9f211fa7867646992c56e7c040747`，
  CANNBot bundle SHA-256=`125de566277696abd1f8ec5823533adc4c24323aae29f0b048d577ffc431b4e1`。
- 基础依赖是 NumPy>=1.26、Loguru>=0.7、Rich>=13；当前 P0 wheel 的已审计 profile 仅覆盖
  CPython 3.10/aarch64、NumPy 1.26.4、Loguru 0.7.3。fresh CANN+Python 环境仍需可用包索引或
  发布方提供完整离线 wheelhouse，不能宣称 bootstrap 能在 NumPilot 自身依赖尚未安装时启动。
- 新增 P1 待定：真正 `GAP_REQUIRED` 后应先检查匹配 CANN 版本的官方
  `https://gitcode.com/cann/ops-nn`，精确匹配时定向构建官方 kernel、由 GLM 生成 AsNumpy
  adapter；不匹配才进入 CANNBot+GLM 新 kernel 生成。当前没有 resolver、没有
  `official_source` Artifact validator，也不会默认拉取/构建 ops-nn。不要把历史 Poisson
  qualification 追溯改写成官方复用；官方源码须保持 provenance/hash/license，不能冒充 GLM。
- 用户下一步会在全新 Notebook 中从远端 tag clone，亲自验证 clone → install → bootstrap
  → GLM migration → NPU execution。入口是 `docs/NumPilot-Demo-v1-Quickstart.ipynb`，详细边界
  在 `docs/FRESH-NOTEBOOK-QUICKSTART.md`；这次验收明确先不做 ops-nn 接入。Notebook 使用
  write-once 根目录、P0 wheel 精确 SHA、私有 llm.json 和进程内代理，不保存 Token/代理值或输出。
- `check-suite`、`wizard`、`bench-cpu`、`bench-npu` 都是 legacy contract 工具。产品内 ordinary
  通用 benchmark 尚未实现；本轮普通 GLM artifact 的真实 NPU 数值/性能只承认外部受信、
  与 source/artifact/runtime/harness 哈希绑定的定向证据，且不得回写成迁移阶段已经执行。
- `/home/ma-user/work/demotest/demo` 的 10 组 NumPy/AsNumpy 文件已只读了解。第 01 例已在
  `case-01-run02` 由 GLM-5.3 完成迁移并通过 source-bound AST/回执复核；未执行源码，02～10
  尚未开始。`/home/ma-user/work/.numpilot-config/llm.json` 已确认 provider=`glm`、
  model=`glm-5.3`、权限 `0600`，Token 未输出。
- 真正调用 GLM 时，必须只在调用进程临时注入
  `http_proxy=http://127.0.0.1:7897` 和
  `https_proxy=http://127.0.0.1:7897`；不得把代理或凭据硬编码进仓库。用户明确说开始测试
  后仍需逐例调用，不能批量预生成。
- 2026-08-25 当前收口工作树完整回归为 `755 passed in 16.15s`，bootstrap/CLI/Notebook/
  path/version 定向回归为 `41 passed`；`git diff --check`、draw.io XML/
  引用和精选 evidence manifest 均通过。提交或构建后仍应在干净 worktree 重新验证，不能把
  这些工作树数字冒充发布 commit 的验证结果。第 01 例发生一次真实 GLM
  请求；没有在普通迁移阶段执行 demotest 源码，也没有重新执行历史迁移或算子生成。
- 第 01 例权威产物：`/home/ma-user/work/numpilot-runs/demotest-migration-20260825/case-01-run02/`
  中 `migrated.py` SHA-256 为 `e935a38a5684238ed3f2d175a801d0d863dbecd2e5505358c272de7c2c0cbf8a`，
  报告 SHA-256 为 `6316c84f7285b00fea32996ebb485f666cb6c55b039a3f18627ae8869243f6a5`。
  状态为 `migrated_scoped_npu`，3 strict + 3 scoped、0 fallback/unsupported；报告明确保持
  `strict_npu=false`、`performance_valid=false`、`execution_performed=false` 和
  `numerical_verification=not_run`。早先的 `case-01-run01` 仅是修复前的 blocked 静态报告，
  不得误当最终产物或重复覆盖。
- 另有一个由 GLM 生成的 2048×2048 float32 Case 01 性能变体，使用外部受信、哈希绑定的
  定向 harness 执行其 exact GLM migration。精选包在
  [`evidence/demotest-case01-20260825/README.md`](../evidence/demotest-case01-20260825/README.md)：
  稳态脚本体 CPU 单线程 `809.784 ms`、NPU `4.499 ms`，NPU 约快 `180.004×`；冷进程端到端
  CPU 约快 `4.44×`。该证据未采集 kernel profiler，只能声明 runtime target/profile/call-count
  guard 与 ACLNN 日志；它不改变普通迁移报告的 `execution_performed=false`，也不表示产品内
  ordinary 通用 benchmark 已经实现。

### 2026-08-25 架构与初赛计划续接记录

- 当时已在干净工作树显式确认 `feat/numpilot-demo`；该历史提交身份和测试计数不应写入当前
  架构图，续接时以实际 `git status`、`git rev-parse HEAD` 和本轮测试为准。
- 未重新执行任何已完成迁移、缺口生成或 repair；两个 GLM demo 最终产物及其 SHA-256
  保持不变，也未人工改写任何案例代码。
- 三页可编辑 draw.io 为 `docs/numpilot-architecture.drawio`，依次说明 ordinary 默认主流程、
  gap/资格化可恢复状态、bootstrap 安装与可信边界；图中使用 lock manifest，不固化易漂移 commit。
- 新增 `docs/COMPETITION-10-DAY-PLAN-20260825.md`。比赛主线冻结为 Poisson Ritz，未来
  10 天优先完成评委路径、CodeArts 真实留痕、精简发布包、PPT 与视频；不扩张通用控制流、
  子图切分、新算子数量、OS 沙箱或通用 profiler runner。
- D2 已开始并按“逐例停顿”完成第 01 例；下一执行点只有在用户确认后才进入第 02 例，先做
  静态扫描/分类，再决定是否调用 GLM。D3 前必须在真实 CodeArts 账号验证 AgentTeam、项目
  Agents/Skills 和 Remote SSH；不能根据 `.codeartsdoer` 文件存在推断产品侧已经加载。

### 本轮可重启检查点

- 最近记录的全量与定向测试通过；本轮并行修改完成后必须重跑并记录新的实际计数，
  `git diff --check` 也必须重新执行。
- 新增 legacy contract 静态 `check-suite`、完整 AsNumpy 调用合同、`build_inputs` 非执行审计、shim 122 API、
  精确 runtime hash capability、CANNBot/operator repair 完整性约束、AsNumpy adapter
  generate/repair、内置 AsNumpy custom-op integration Skill、cbrt/Poisson qualification 与
  demo 工具。
- `complete-gaps --operator-goal performance` 是保留但尚未实现的选项；CLI 已在任何模型、
  Skill、网络或输出动作前明确返回 2，不能再宣传为已实现流程。
- 用户要求 demo 的补码和迁移代码只能由其 GLM 模型生成；NumPilot 只负责框架门禁、执行和
  记录。不要人工改写下面两个 GLM 最终产物。

两个 GLM demo 最终产物：

- `/home/ma-user/work/numpilot-runs/examples-demo-20260824-run02/case-01/final-v2-output/migrated.py`
  （SHA-256 `a093d2dba01982a5bd5d1150452819e59e5957d3e01a4979b36e9e4230e18a77`）；
- `/home/ma-user/work/numpilot-runs/examples-demo-20260824-run02/case-02/final-v2-output/migrated.py`
  （SHA-256 `051fca3b19a2299f1dc28e6111d7afe2f1c492553e02f344147caf5c43e4c9a3`）。

两者均已在 P0 move wheel 上独立运行并通过 deny-NumPy smoke，分别出现真实 ACLNN
`matmul/exp/sum/divide` 与 `sin/cos/add/absolute/amax` 日志。必须保留以下新发现：case-02
原始 `normalized.py` 使用 `np.amax(abs_signal)`；真实 AsNumpy 在 `axis=None` 时返回 Python
`float`，而 NumPy float32 输入返回 `np.float32`，因此 legacy `check-suite` 会诚实地把原始
case-02 阻断为 1 direct + 1 blocked。GLM 最终产物已把这一维输入改成
`ap.amax(..., axis=0, keepdims=False)` 并通过真机；不能让 shim 再把原始调用误报为机械直迁。

本轮真实 AsNumpy 审计依据位于：

- Python 源码：`/home/ma-user/work/asnumpy/src/asnumpy/`；
- C++/pybind 源码：`/home/ma-user/work/asnumpy/bindings/python/` 与
  `/home/ma-user/work/asnumpy/csrc/`；
- 通用审计 wheel：`/home/ma-user/work/numpilot-runs/heat-diffusion-20260813/audited-site/`；
- demo P0 wheel：`/home/ma-user/work/numpilot-runs/examples-demo-20260824-run02/runtime/venv/`。

外部资格化结果：cbrt 最终 qualification 位于
`/home/ma-user/work/numpilot-showcases/cbrt_sensor_calibration/53-final-qualification-v2-20260825`
（21 sealed artifact、4 AsNumpy、7 numerical tests 通过）；Poisson 默认实时 demo 位于
`/home/ma-user/work/numpilot-runs/poisson-demo-20260825-final`，101/101 rewrites、20 轮驻留
执行和 strict-NPU 门禁通过。仓库外结果不能自动提升为公开性能结论。

推送需使用用户指定代理：

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
git push origin feat/numpilot-demo
```

远端 GitCode 使用 HTTPS Basic 身份；若交互提示，用户会输入账号和令牌。禁止 force push。
`.git/info/exclude` 中两份本地未发布文档仍只保留本机，不要纳入：
`docs/RESTART-HANDOFF-20260824.md` 与
`docs/evidence/poisson-ritz-large-single-core-20260824.md`。

Poisson 两个缺口算子已由用户 LLM 配合固定 CANNBot Skill 完成生成、repair、封存和六门
资格化；已封存的原始规模与广播/CSE 性能结论仍以仓库内精选证据及其 manifest 为准：

- [`P0-LLM-CANNBOT-RUN-20260821.md`](P0-LLM-CANNBOT-RUN-20260821.md)；
- [`evidence/poisson-ritz-p0-npu-20260821.md`](evidence/poisson-ritz-p0-npu-20260821.md)；
- [`evidence/poisson-ritz-performance-original-scale-20260823.md`](evidence/poisson-ritz-performance-original-scale-20260823.md)；
- [`evidence/poisson-ritz-broadcast-optimization-20260823.md`](evidence/poisson-ritz-broadcast-optimization-20260823.md)；
- [`COMPETITION-GAP-REVIEW-20260821.md`](COMPETITION-GAP-REVIEW-20260821.md)。

`run_poisson_demo.sh` 提供公开证据校验、qualification、doctor、实时 strict-NPU 迁移和
驻留执行入口。`--large-single-core` 仅为实验诊断：它验证 CPU affinity 和 BLAS 单线程，
但在缺少同 runner/规模、哈希匹配且通过的 profiler audit 时必须保持
`performance.published=false`，不得发布或引用其性能比值。CodeArts 集成仍按用户决定后置。

续接时先执行：

```bash
cd /home/ma-user/work/numpilot
git status --short --untracked-files=all
python3 -m pytest -q
git diff --check
```

然后阅读 `README.md`、当前变更对应测试和上面的精选证据；不要把仓库外运行目录中的结果
直接提升为公开结论。

---

## 2026-07-18 历史 contract 基线（不得作为 Demo v1 默认流程）

> 以下正文保留早期架构、风险和实现背景。其 `/home/lixing` 路径、测试数量、capability
> 判断、未完成项和复现命令均属于 2026-07-18 快照，不是当前工作目录、当前命令或当前结论；
> 续接工作一律以上方 2026-08-25 摘要为准。

## 1. 历史项目目标与当时定位

长期目标：构建一个利用 CodeArts 多智能体，把科学计算 NumPy 网络迁移到 AsNumpy/Ascend NPU，并在遇到能力缺口时路由到复用、组合或 Ascend C 算子生成的框架。

该历史快照中的可信定位：

```text
CodeArts AgentTeam / 项目级 Agent / Skill   协作控制面
                         ↓
NumPilot CLI + contracts + capability       确定性执行/证据面
                         ↓
AsNumpy + CANN + 裸金属 Ascend NPU
```

当时的 contract demo 只承诺一个顶层同步 `forward()` 的完整 NPU 岛。该句不是当前 Demo v1
ordinary 默认流程；它只用于解释 legacy 实现背景。

“补全算子”路线仍然合理，但必须放在 Gap Router 的最后一级：

```text
已有严格 capability → ACLNN → 组合已有 NPU 原语
→ 只有热点且收益合理时生成 Ascend C → 显式 CPU fallback/unsupported
```

不能把“函数名存在”“命中已有 AsNumpy 扩展 API”或“编译成功”当作新算子已经 qualified。

## 2. 历史快照中已经完成

### 2.1 迁移边界与静态分析

- 只分析、改写一个顶层同步 `forward()`；`build_inputs()` 和顶层展示逻辑留在 CPU。
- `DecisionAgent` 固定产生 `whole_forward_npu`，保留的 `--threshold` 只为旧接口兼容。
- forward 入口仅对 NumPy ndarray 做幂等 H2D；设备数组和标量保持原样。
- 分析阶段阻断未验证的 NumPy keyword、风险标量二元签名、Python ndarray dunder 等。
- strict 迁移产物禁止 residual NumPy、host escape、未知 helper/方法、全局预计算、未建模控制流和被丢弃的 `ap.*` 结果。
- 从原始 `forward + OffloadPlan` 推导精确 `expected_rewrites`；算子名称或 callsite 数不一致即失败，不能用“数量相同的错误算子”蒙混通过。
- Python `3.9.9` 已单独验证兼容；`ast.Match` 只在 Python 3.10+ 存在时注册。

### 2.2 三层语义与运行时证据

- 原始输入只生成一次，并保存/恢复 NumPy 与 Python RNG 状态。
- 三层门禁：输入生成一致；同一规范输入上的 forward 一致；迁移产物自身 `build_inputs + forward` 一致。
- 比较器检查嵌套结构、shape、dtype、实际 dtype 的 `atol + rtol`、整数/布尔精确值、零参考、NaN 和 Inf。
- 异常 phase/type/message/traceback 结构化落盘；JSON 不写非标准 Infinity。
- `sys.modules['asnumpy']` 注入由进程内全局锁保护并恢复。
- 两条验证路径分别记录实际 `ap.*` callsite 次数；每条路径都必须覆盖全部静态 callsite。
- 已加入不可达 `ap.*`、host NumPy escape、NPU decoy + host 全局结果、错误算子同数量替换等反例。

### 2.3 Capability/provenance 与性能

- 新增版本化、JSON 可序列化的 capability manifest，绑定 AsNumpy 版本、模块路径、CANN、SoC 和 Python。
- 当前严格审计版本：`0.3.1.dev4+g1929a3ba4`，audit ID 为 `asnumpy-0.3.1.dev4+g1929a3ba4-source-1929a3ba4-v1`。
- 已知 `matmul/inner/outer` 为 CPU fallback；`dot/vdot/floor` 有条件 host/CPU 路径；`signbit` 为 host assisted；`angle` 在当前 wheel 未审计。它们不能发布 strict-NPU 性能。
- 未明确返回 `strict_npu=true`、版本漂移、API 缺失、CPU fallback 或 unknown 都 fail closed。
- `bench-cpu` 与 `bench-npu` 为独立进程，记录源码/迁移产物 hash、实际 BLAS 线程、参考输出和计时范围。
- 缺 CPU baseline 时 `performance_valid=false`、`speedup=null`、状态为 `pass_unqualified`。
- `--allow-cpu-fallback` 只诊断，不计算 speedup。
- 比值小于 1 时终端和 Markdown 都明确写“CPU 更快”，不再称为 NPU 加速。

### 2.4 CodeArts

已新增：

```text
.codeartsdoer/
├── agents/
│   ├── numpilot-orchestrator.md
│   ├── numpilot-analyzer.md
│   ├── numpilot-operator-engineer.md
│   └── numpilot-evidence-auditor.md
└── skills/
    ├── ProjectSkillStatus.txt
    └── numpilot-migration-gate/SKILL.md
```

设计原则：AgentTeam/自定义智能体负责协作、任务图、人工确认和展示；NumPilot CLI 保留确定性门禁。官方文档没有明确承诺 AgentTeam 自动把项目级自定义智能体装配成固定队伍，必须在比赛账号实测。若不支持，就分别展示 AgentTeam 动态 Teammate 与 `numpilot-orchestrator → task → subagent` 两条入口。

## 3. 历史验证状态

### 3.1 单元测试

```text
/home/lixing/np310/bin/python -m pytest -q tests
106 passed
```

系统 Python 3.9 未装 `rich`，不用于全套 CLI 测试；核心 strict surface 已用 Python `3.9.9` 单独验证。

### 3.2 裸金属环境

- 8 × Ascend 910，检查时均为 OK；
- CANN `8.5.0`；运行架构 `dav-2201`；
- Python `3.12.13`；AsNumpy `0.3.1.dev4+g1929a3ba4`；
- 没有使用 Docker。

### 3.3 strict-NPU 正例

`examples/elementwise_forward.py`：

- source SHA256：`6d4778553b5bf053b0c18b67ad9f5f37c0d5448e196a80642a1a79f00f2924bb`；
- migrated SHA256：`d903cea364357a4b2210df84e0f2a55647827929274df975ca80a9e82e595084`；
- `expected_rewrites == rewrites`，`rewrite_integrity=true`；
- kernel 与 artifact 路径都执行 `add×1, sqrt×1, square×2, tanh×1`；
- 三层语义通过，`max_abs_diff=1.1920928955078125e-07`；
- CPU 单线程基线 `0.01498 ms`；当前 NPU 口径 `0.41934 ms`；
- `performance_valid=true`，NPU/CPU=`0.035723×`，即 CPU 约快 `27.99×`。

该 NPU 口径为“输入预驻留设备、排除 H2D、每轮包含 D2H 同步”，不是 kernel-only，也不是完整端到端。单次小网络数据只用于验证证据链，不用于宣传性能优势。

证据：`/tmp/numpilot-final-bench-20260718/`。`/tmp` 可能被清理，需要长期保存时应把原始 JSON/Markdown/profiler artifact 复制到版本化外部制品库，不要只抄数字。

### 3.4 fallback 负例

`examples/mlp_forward.py` 因 `matmul` CPU fallback 在执行前退出 3：

```text
status=rejected
performance_valid=false
speedup=null
cpu_fallback_ops=[matmul]
```

证据：`/tmp/numpilot-final-matmul-rejection-20260718/`。

## 4. 历史快照中的未完成项与风险

### P0：下一步先做

1. **OS 级用户代码隔离**：当前只有进程内锁。需把 load/build_inputs/forward 放入受限子进程，限制超时、内存、文件、网络和设备，并保留结构化结果。
2. **真实 kernel/profiler 证据**：当前 runtime trace 证明 Python API 被调用，再结合版本绑定源码审计推断实现来源；仍应采集 kernel 名、时间线、stream、H2D/D2H 字节和原始 profiler artifact。

### P1：比赛差异化闭环

1. 定义 `GapSpec / OperatorArtifact / CapabilityRecord` schema。
2. 选择一个真实科研网络中的热点缺口，走通 ACLNN/组合/Ascend C 其中一种真实补全路径。
3. 把 build、UT、裸金属 ST、精度、性能、artifact hash、安装与回滚作为 hard gate。
4. 在目标 CodeArts 账号验证 AgentTeam、项目 Agent、Skill、Remote SSH、沙箱和 Hook；保存任务图、对话、diff、测试失败修复、review/流水线证据。
5. 使用真实、可追溯数据和原程序许可证完成端到端案例。

### P2：后续扩展

- callsite/value 级 IR、shape/dtype/layout 传播；
- 真实 CPU/NPU 子图划分、动态 shape guard 和成本模型；
- AsNumpy 插件 ABI、DLPack、缓存和多 CANN/SoC；
- 自动失败诊断、修复重试和长期算子知识库。

## 5. 历史建议任务（不作为当前优先级）

推荐只选一个纵向目标，不要同时扩张所有方向。

### 任务 A（首选）：隔离执行器

交付标准：

- 新增 `SandboxRunner`/子进程协议；
- 原始与迁移产物在不同受限进程执行；
- 支持 timeout、进程退出、异常和嵌套 ndarray 结果序列化；
- 不再修改主编排进程的 `sys.modules`；
- 并发验证不会串 backend；
- 恶意文件写入、无限循环、超内存至少有确定性失败测试；
- shim 全量测试和裸金属 elementwise smoke 继续通过。

### 任务 B：Profiler 证据 schema

交付标准：

- 每次 benchmark 保存 profiler 原始目录及 manifest；
- JSON 能追溯 kernel、设备、时间、传输字节和计时范围；
- 无 profiler 或 kernel 不匹配时，比赛级 `performance_qualified=false`；
- Python API trace 与 profiler trace 分字段展示，不能混称。

### 任务 C：单个真实缺口算子闭环

先选算子，再创建 GapSpec。优先满足：真实网络热点、NumPy oracle 明确、shape/dtype 边界可控、现有 AsNumpy 确实缺失、预期收益覆盖生成/维护成本。不要先追求数量。

## 6. 裸金属复现命令

```bash
cd /home/lixing/numpilot
source /usr/local/Ascend/cann/bin/setenv.bash
export ASCEND_SOC_VERSION=dav-2201
export LD_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"

/home/lixing/np310/bin/python -m numpilot doctor --backend npu
/home/lixing/np310/bin/python -m numpilot migrate \
  examples/elementwise_forward.py --backend npu --out /tmp/numpilot-resume-migrate

/home/lixing/np310/bin/python -m numpilot bench-cpu \
  examples/elementwise_forward.py --out /tmp/numpilot-resume-bench --cpu-threads 1
/home/lixing/np310/bin/python -m numpilot bench-npu \
  examples/elementwise_forward.py --out /tmp/numpilot-resume-bench

# 预期退出码 3；不要因为非零退出误判测试脚本本身坏了。
/home/lixing/np310/bin/python -m numpilot bench-npu \
  examples/mlp_forward.py --out /tmp/numpilot-resume-matmul-rejection
```

只在 `npu_benchmark.json` 同时满足以下条件时引用比值：

```text
performance_valid == true
rewrite_integrity == true
provenance.ok == true
provenance.strict_npu == true
provenance.static_forward_surface.ok == true
provenance.runtime_trace_ok == true
baseline_compatible == true
input/kernel/artifact comparisons == consistent
```

## 7. 不得破坏的真实性原则

- shim 只证明语义，不证明 NPU。
- API 存在不等于 strict NPU；CPU fallback/unknown 不得计算 speedup。
- “已有 AsNumpy 扩展 API”不等于本次新生成自定义算子。
- 当前 MLP/GEMM 的 `matmul` 是 CPU fallback，历史 12～15× 不能作为 NPU GEMM 证据。
- `NPU/CPU < 1` 必须明确写 CPU 更快。
- 当前 static/runtime Python 证据仍不替代 profiler/kernel trace。
- CodeArts 配置文件存在不等于目标账号已成功加载，也不等于 AgentTeam 自动装配固定角色。
- 不使用 Docker；本机已有裸金属 NPU/CANN 环境。

## 8. 仓库快照

- 分支：`feat/numpilot-demo`；
- 本轮开始时 HEAD：`cd32f40`；
- 当前修改尚未提交；
- 核心新增文件：`capabilities.py`、`comparison.py`、`module_injection.py`、CodeArts Agent/Skill、strict elementwise 示例、真实性测试和本文档；
- `git diff --check` 已通过；
- 最后一次目标仓库全量测试应在继续开发前重新执行并记录。

完成下一阶段后，请更新本文的日期、测试数、NPU 证据路径、未完成项和下一步，不要只在聊天里交接。
