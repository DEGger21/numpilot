# NumPilot 五日产品完善计划

> 计划基线：2026-08-25<br>
> 执行窗口：2026-08-26 ～ 2026-08-30<br>
> 当前版本：`v0.2.0a2`，HEAD `374e4da7df827bbf69668eb896e86ab1f95fa755`
> 五日目标：交付一个可重启复用、可接入用户自带 OpenAI-compatible 模型、能诚实分析/迁移/验证普通 NumPy 脚本的 RC 主通道。

## 0. 本计划的口径

产品目标已经从“只能使用 GLM”调整为：

> 用户拉取 NumPilot，填写自己的 OpenAI-compatible 模型信息，由该模型完成迁移和后续算子代码生成；GLM 只是一个配置示例。

因此，本计划在 2026-08-26～30 的执行优先级和模型口径上，替代
`COMPETITION-10-DAY-PLAN-20260825.md`、`FRESH-MACHINE-AI-PROMPT.md`、README 和旧 handoff
中的 GLM-only 描述。旧文档暂时保留作历史记录，必须在 Day 1 和 Day 5 同步更新，避免用户按旧说明操作。

五天只收敛一条可信纵向主线：

```text
用户 clone/install 一次
        │
        ▼
source 无凭据的持久激活脚本，重启后离线复用
        │
        ▼
填写任意满足已声明合同的 OpenAI-compatible 模型
        │
        ▼
analyze（不执行源码、不联网、不加载模型）
        │
        ▼
migrate（获授权后才调用用户配置模型）
        │
        ▼
显式 verify → 合格后才 benchmark → 哈希绑定报告
```

CANNBot 在这条主线之外作为 gap lane 的已校验 Skill bundle：direct migration 不依赖它；真正遇到缺口后，才由同一个用户配置模型读取 CANNBot 内容生成或修复算子。

## 1. 五日后的可验收产品状态

到 2026-08-30 收工时，以下项目必须同时成立：

1. `provider` 只作为审计标签，不再充当厂商授权门；生产路径不再要求 `provider=glm` 或 `model=glm-*`。
2. 支持包含 `/` 的常见模型 ID；配置模型、请求模型、响应模型和内容哈希有明确、不可静默放宽的身份规则。
3. 仅 `/work` 持久的平台安装一次后，可以 source 一个不含 Token/代理的脚本恢复 venv、config、state、cache、receipt、CANN 和 CANNBot 路径。
4. `numpilot analyze FILE --json` 成为 ordinary script 的独立只读入口；成功分析不再依赖启动 `migrate` 后交互退出。
5. capability 从“API 名存在”升级为“qualified name + runtime identity + call profile”；已知 fallback、host-assisted 和未资格化合同均 fail closed。
6. `/home/ma-user/work/demo` 的 10 个案例全部得到稳定、可解释的分类；不追求 10/10 迁移。
7. 当前主功能候选 `03/06/10` 能通过普通脚本分析；模型获授权后，至少 Case 06 完成迁移、数值验证和严格门禁下的单核 CPU/NPU 评测。
8. 默认 pinned runtime 中 `matmul` 作为 fallback 负例；后续若增加精确资格化的定向 runtime，
   仍不得在没有独立性能门禁时进入可发布加速比。
9. CANNBot 固定 bundle 能在 fresh shell 离线自动发现；Poisson demo 和安装器使用同一 commit-scoped 路径。
10. 全量测试、干净构建、secret scan、文档和证据 manifest 全部重新生成；不得沿用旧测试数字冒充当前结果。

## 2. 截至 2026-08-25 的最新进展

### 2.1 仓库与测试基线

- 当前仓库处于 detached `v0.2.0a2`，HEAD 为 `374e4da7df827bbf69668eb896e86ab1f95fa755`。
- 本文写入前 tracked worktree 无改动，只有未跟踪 `.venv/`；正式开发前必须从该精确 HEAD 创建功能分支，不在 detached HEAD 上累计实现。
- 2026-08-25 本轮已重新执行完整测试：`775 passed in 16.95s`。
- 与通用模型/CANNBot/bootstrap/ordinary migration 直接相关的定向集合：`175 passed in 3.80s`。
- 上述通过只能证明现有 GLM-only 实现内部自洽；其中部分测试主动固化了错误的 GLM-only 产品约束，Day 1 必须改写。

### 2.2 安装、运行时与持久化

- source-first `install.sh → bootstrap plan/apply/verify → doctor` 已能安装固定 AsNumpy 并校验 CANNBot。
- 当前机器为 Ascend 910B4 / CANN 9.0.0；固定 AsNumpy 是
  `0.3.1.dev4+g1929a3ba4`，commit
  `1929a3ba44eb532a9f544f774a249cdf1c4119be`。
- 在干净 CANN 动态库顺序中，AsNumpy smoke、NPU doctor、CANNBot 离线 bootstrap verify 均通过。
- 仅 `/home/ma-user/work` 跨重启持久；真实 config/state/cache/receipt 已放到 `/work`，但当前 `.codex-env.sh` 只设置 Codex 的 `CODEX_HOME/PATH`，没有激活 NumPilot 的四个持久路径。
- fresh shell 不显式设置路径时，CLI 会回到临时 home，导致“本地已有 checkout/config/receipt，但 auto 模式报告不存在”。
- 登录环境还可能继承多套 CANN `LD_LIBRARY_PATH`，只叠加 `set_env.sh` 不能保证与安装时一致。

### 2.3 AsNumpy 能力盘点

- NumPilot registry 有 122 个顶层名称，但固定 AsNumpy 实际只有 121 个公开计算 API；唯一多出的 `angle` 根本不存在。
- 顶层有 114 个“已知 NPU 路线名”。这个数字只证明 wrapper 路线，不证明完整 NumPy dtype/axis/rank/signature 合同。
- 必须硬剔除：
  - CPU fallback：`inner, matmul, outer`；
  - conditional fallback / 整张量 host inspection：`dot, vdot, floor`；
  - host-assisted：`signbit`；
  - unavailable：`angle`。
- nested `ap.linalg` 的 `det/inv/qr` 真机 NPU 烟测通过；`ap.random` 有 14 个 NPU/composite-NPU 分布 API。但当前 NumPilot registry/resolver 不认识这些 qualified names。
- 已发现多项调用合同缺口，包括 `cumsum/cumprod(axis=None)`、`sort(axis=None)`、创建函数 `dtype=None`、受限 `einsum` equation、random dtype 和缺少 ndarray `.astype()`。

权威本地盘点：

- `/home/ma-user/work/asnumpy-npu-api-and-demo-audit-20260825.md`
- `/home/ma-user/work/.numpilot-state/runs/api-coverage-20260825/runtime-capability-manifest.json`
- `/home/ma-user/work/numpilot-experience-notes.md`

### 2.4 demo 当前分级

| 层级 | 案例 | 当前结论 |
|---|---|---|
| 当前 NumPilot 首选 | `03, 06, 10` | 无已知 scanner/registry 缺口；前提是 pinned-source receipt 与 runtime identity 匹配。 |
| 后端可覆盖、框架待补 | `02, 04` | 02 需要资格化常量运算与 reduction 参数；04 的 `linalg.det` 已在 NPU，但 nested resolver 缺失。 |
| 需要显式等价改写 | `05, 07` | 05 要显式整数 axis；07 要 native Cast、nested random 和 reduction 合同。 |
| 五日内跳过 | `01, 08, 09` | 01 有 matmul CPU fallback；08 有 dot/floor host 路线；09 缺 arange/erf/astype 且含 SciPy。 |

后续主性能样例选择 **Case 06**；Case 03 用作最干净的数值/流程对照，Case 10 主要做功能演示。Case 02/04 在 Day 2 完成 profile/resolver 后再决定是否加入主路径。

### 2.5 性能诊断结论

- CPU 基线已经用 `taskset` 和线程环境约束为单核，不是 CPU 配置异常。
- 9M 元素上，AsNumpy 的 add、exp2、multiply、divide、mean、sort、linspace 等能够显示真实 NPU 优势；AsNumpy 整体没有失效。
- `full` 和低强度逐元素算子受每次设备分配影响；H2D/D2H 可能抵消 kernel 优势。
- `dot/vdot` 为检查 dtype 无条件 D2H 整张输入，是明确的 wrapper 性能问题。
- 官方 examples 使用最小值作为所谓“稳定指标”不可靠；未来统一报告 median、p10/p90 和样本数。
- 五日计划不修 AsNumpy allocator、dot/vdot 或 matmul 上游实现；NumPilot 的职责是诚实分类、阻断错误性能结论，并把问题留作上游任务。

### 2.6 CANNBot 状态

- checkout：`/home/ma-user/work/.numpilot-cache/numpilot/cannbot-skills/1ef6372a538f39b0b7ed34666a3e1f465fb4936a`。
- commit：`1ef6372a538f39b0b7ed34666a3e1f465fb4936a`，tracked worktree clean。
- NumPilot 选中 bundle：36 files / 272327 bytes；SHA-256：
  `11dcc3cd05139bcba870ccd540cf1e56a5430a351f27c10707d481a6544ae895`。
- `setup-skills --check` 与离线 `bootstrap --verify --with-cannbot` 已通过。
- 安装器已经支持 `--with-cannbot --accept-cannbot-license`；由于许可证确认，它必须保持显式 opt-in。
- 当前缺口是路径接线：`run_poisson_demo.sh` 仍查旧 `$WORKSPACE_ROOT/cannbot-skills` / `NUMPILOT_CANNBOT_ROOT`，没有使用 installer 的 commit-scoped cache。

### 2.7 当前四个 P0

| P0 | 现状 | 用户影响 |
|---|---|---|
| 通用模型硬锁 | HTTP transport 通用，但 migration/gap/repair/adapter/receipt 强制 GLM identity。 | 用户能配置/probe 自己的模型，却在真正生成前被拒。 |
| 重启持久化断点 | config/state/cache/receipt/CANN 激活需要每个 shell 手工拼装。 | 重启后像是需要重新安装或重新拉取 CANNBot。 |
| capability 过宽 | 114 个名字被近似成无约束 native；`angle` 虚假，nested API 缺失。 | 合法 NumPy 调用可能被误放行，真实 NPU 能力也可能被误报缺失。 |
| ordinary 证据链不完整 | 缺独立 analyze、普通脚本数值验证和可信 benchmark。 | 用户无法在产品内完成“分析→迁移→验证→性能”闭环。 |

## 3. 关键依赖和实施顺序

```text
通用模型 identity/receipt ──────────────┐
持久环境与统一路径 ─────────────────────┼──► ordinary analyze/migrate
qualified capability/profile ──────────┘              │
                                                       ▼
                                             verify / benchmark
                                                       │
                                                       ▼
                                           reboot + CANNBot RC
```

- 通用模型合同是所有远程生成、repair 和 adapter 的前置条件。
- 持久路径是 fresh-machine、receipt、CANNBot 自动发现和重启复现的前置条件。
- capability profile 是 analyze、migrate 和性能发布门禁的事实基础。
- CANNBot generation/repair 依赖通用模型与持久路径，但不能阻塞 direct lane。

## 4. Day 1：通用模型合同 + 持久环境基础

日期：2026-08-26

### 目标 A：取消 GLM 厂商硬锁，但不削弱身份审计

- [x] 引入明确的协议字段，例如 `protocol=openai-chat-completions`；`provider` 只作为用户填写的审计标签。
- [x] 将 `require_glm_configuration`、`validate_glm_remote_receipt`、`generate_script_with_glm` 等生产合同泛化；旧函数名如需保留，只能作为兼容 alias。
- [x] migration、gap generation、operator repair、adapter generation/repair 全部复用同一个通用 model client 与 identity validator。
- [x] 把 `generated_by=glm` 改成 `generated_by=configured_model`，同时记录真实 provider/protocol/model。
- [x] 配置模型必须与请求 body 的 model 精确一致。
- [x] 响应 model 默认要求精确一致；网关别名必须由用户显式配置 allowlist，缺失或不一致不得静默接受。
- [x] 模型 ID 比较使用原始值或哈希，展示脱敏单独处理；安全支持 `org/model`，不再把 `/` 改成 `?` 后做 equality。
- [x] 定义本 sprint 的最小兼容合同：Bearer auth、Chat Completions、文本 `choices[0].message.content`，以及受控的 token 参数/profile。合同之外由 probe 明确拒绝，不笼统宣称全部兼容。
- [x] `configure-llm --check`、`--probe` 与真正生成使用同一 readiness 规则。
- [x] Token、base URL 中的凭据、代理值、响应正文永不进入 receipt、日志或 Git。
- [x] GLM 只保留为 README/config 示例，不再进入运行时授权条件。

### 目标 B：建立单一持久根和无凭据激活脚本

- [x] 增加单一持久根配置，例如 `NUMPILOT_HOME` / installer `--persistent-root`；具体路径变量优先级固定为：显式细分变量 > 单一根 > XDG 默认。
- [x] installer 生成幂等、可 source 的 `activate-numpilot.sh`，只包含路径、venv 和经过校验的 CANN 激活信息，不包含 Token 或代理。
- [x] 激活脚本统一 config/state/cache/bootstrap receipt/CANNBot exact checkout，并且 source 两次不重复污染 PATH/LD_LIBRARY_PATH。
- [x] CANN 激活只清理冲突的 Ascend 路径，保留无关动态库路径；doctor 同时报告冲突顺序。
- [x] 标准持久 HOME 用户保持零额外配置；只有临时 HOME/Notebook 用户使用 persistent-root。
- [x] 安装器不修改 `.bashrc`、`.codex-env.sh` 或用户 shell；本机可由用户选择在 `.codex-env.sh` 中 source 生成的激活脚本。

### Day 1 产物

- 通用 model config/receipt schema 与兼容迁移说明。
- 统一 model validator 及 fake OpenAI-compatible endpoint 测试。
- persistent-root 路径解析、installer 参数和 activation script。
- README、Fresh Machine、Notebook 中 GLM-only 主口径的第一轮替换。

### Day 1 验收门

- `glm-*`、`gpt-*`、`deepseek-*`、`org/model` 四类模型 ID 均可通过本地 fake endpoint 的 configure/check/probe/generate 合同测试。
- 请求模型、响应模型/显式 alias、内容哈希和 receipt 篡改均能 fail closed。
- 生产执行路径不再硬锁 `provider=glm` 或 `glm-*`；只允许示例和兼容测试保留相应文本。
- 临时 HOME 清空后，source 激活脚本两次，venv/config/state/cache/receipt/CANN 路径仍全部指向 persistent-root。
- 激活脚本通过 secret scan，不含 auth token、代理或响应正文。
- direct lane 在完全没有 CANNBot checkout 时仍能工作。

### Day 1 止损线

若通用 model identity/receipt 没有通过篡改测试，禁止继续任何真实模型工作流；Day 2/3 仍可继续做纯本地 capability 和 analyze。

## 5. Day 2：Capability 真实性与 demo 黄金 profile

日期：2026-08-27

### 任务

- [x] 从 registry 删除不存在的 `angle`，增加固定 AsNumpy 公开映射与 NumPilot registry 的 CI drift check。
- [x] capability key 升级为 canonical qualified name，例如 `linalg.det`、`random.normal`；外部报告保留 `numpy.linalg.det` 等源码名称。
- [x] resolver 支持 nested namespace，不把“解析器不认识”误导成“需要写新算子”。
- [x] 将计算路线与调用合同分离：
  - `NATIVE_NPU / COMPOSITE_NPU / HOST_SCALAR_BOUNDARY`；
  - `HOST_ASSISTED / CONDITIONAL_CPU_FALLBACK / CPU_FALLBACK`；
  - `UNAVAILABLE / UNKNOWN`；
  - 独立的 dtype/axis/rank/signature/equation qualification 状态。
- [x] runtime profile 绑定 AsNumpy source、module/native hash、CANN、SoC、Python 和 bootstrap receipt；任一漂移转 `CAPABILITY_AUDIT_REQUIRED`。
- [x] 为 demo 黄金切片先建立精确 profile：`03/06/10`；随后资格化 02 的常量/reduction 调用和 04 的 `linalg.det`。
- [x] `axis=None`、`dtype=None`、受限 einsum equation、random float64→算子不兼容等合同默认 fail closed。
- [x] `inner/matmul/outer/dot/vdot/floor/signbit/angle` 维持明确的非 strict 结果。
- [x] 生成机器可读 manifest，不靠人工复制 API 名单。

### Day 2 产物

- 版本化 qualified capability manifest/schema。
- nested resolver 与 profile tests。
- 10 个 demo 的 source hash、API/callsite、decision、理由和 next action 矩阵。
- AsNumpy upstream 问题清单；不在 NumPilot 内偷偷 patch upstream 行为。

### Day 2 验收门

- 固定 checkout 得到 121 个真实顶层 API、114 个已知 NPU 路线名；`angle` 不再可用。
- pinned runtime/receipt 下 `03/06/10` 得到确定的 direct 结果；receipt 或 native hash 漂移时必须转 audit-required。
- 04 只能在 `linalg.det` qualified profile 建立后进入 direct，不能仅因 `hasattr()` 放行。
- 05/07 明确标为需要合同改写或 profile pending；01/08/09 不进入 strict 性能路径。
- matmul 的性能字段固定：`performance_valid=false`、`speedup=null`。

### Day 2 止损线

若 runtime identity 或 profile 仍不稳定，Day 3 只发布诚实的 analyze/audit-required，不允许调用模型迁移，也不开始新算子生成。

## 6. Day 3：Ordinary `analyze` 与通用 `migrate`

日期：2026-08-28

### 任务

- [x] 新增独立目标接口：`numpilot analyze FILE [--json] [--out PATH]`。
- [x] analyze 只读 AST 和 capability receipt，不执行用户源码、不联网、不加载模型。
- [x] 输出至少包括：source SHA-256、imports、qualified APIs、callsite、call profile、五类 decision、audit ID、next action、`actual_remote_call=false`、`execution_performed=false`。
- [x] 分析本身成功时退出 0；CI 用显式 `--require-direct` 决定非 direct 是否返回非零。
- [x] ordinary `migrate` 消费并重新验证 hash-bound analysis；源码漂移时零副作用失败。
- [x] 只有 `DIRECT` 或用户显式接受的已审计 fallback 才能调用用户配置模型。
- [x] fallback 在非交互环境绝不默认同意；missing/audit-required/blocked 不误调用模型或 CANNBot。
- [x] 模型只负责生成迁移代码；NumPilot 继续负责静态扫描、AST、receipt、hash 和安全门禁，不使用本地 AST 作为生成失败后的替代产物。
- [x] managed output 保持 write-once，禁止旧 artifact/report 混用。
- [x] 全部 CLI、报告与错误提示使用“configured model / OpenAI-compatible model”，GLM 只出现在示例中。

### Day 3 产物

- `analyze` CLI、JSON schema 和 human-readable summary。
- 通用模型 ordinary migration 主通道。
- `/home/ma-user/work/demo` 10/10 的无模型分析报告。
- fake model 下的 direct migration fixtures；不需要真实外部模型。

### Day 3 验收门

- 10 个 demo 在不联网、不执行源码、不加载模型的条件下全部产生确定 JSON 和 Markdown。
- `03/06/10` 在匹配 runtime profile 下为 direct；02/04 的状态完全取决于 Day 2 实际 qualification，不强行写成 direct。
- fake provider 能完成 direct 精确迁移并通过 source/response/artifact/AST 绑定。
- source drift、model receipt、content hash、qualified API 或 callsite 篡改全部 fail closed。
- ordinary 流程不再借 legacy `check-suite`、`bench-cpu`、`bench-npu` 证明自己可用。

### Day 3 止损线

若通用 migrate 仍不能保证身份与 artifact 完整性，则保留 `analyze` 为可交付成果，真实模型调用继续冻结。

### 2026-08-26 Day 2/Day 3 实际状态

- Day 2 提前完成：固定 catalog 为顶层 121、`linalg` 6、`random` 14，共 141 个 canonical key；顶层已知 NPU 路线 114。匹配运行时 identity SHA-256 为 `7d443cb390fdd8bb398e10ed00b82d1a5aed624c1ed777ca8d6652fc9cf1b124`。
- 10 个 demo 的黄金判定稳定为：`01 FALLBACK_CONSENT`、`02 DIRECT`、`03 DIRECT`、`04 DIRECT`、`05 BLOCKED`、`06 DIRECT`、`07 BLOCKED`、`08 FALLBACK_CONSENT`、`09 BLOCKED`、`10 DIRECT`。
- Day 3 提前完成真实迁移测试：`03/06/10` 为 `migrated_strict_npu`；`02/04` 为精确调用 profile 约束下的 `migrated_scoped_npu`；用户显式接受后 `01/08` 为 `migrated_with_fallback`；`05/07/09` 只发布 `blocked` 报告，`actual_remote_call=false`，且没有 `migrated.py`。
- 7 份迁移代码全部由配置的 `glm-5.3` 真实生成并通过 source/response/artifact hash、runtime identity 与 exact AST 审计；Codex 只修改框架和验收产物。Case 01 首次网关请求返回 HTTP 504，严格终止且无产物；一次干净重试成功。
- 本轮所有迁移报告均为 `execution_performed=false`、`numerical_verification=not_run`、`performance_valid=false`；尚未进入 Day 4，不发布任何新加速比。
- 机器可读索引：`docs/evidence/day3-demo-migration-results-20260826.json`；真实 run 目录位于 `/home/ma-user/work/.numpilot-state/runs/`。
- 最新全量回归：`1026 passed in 22.19s`；`git diff --check` 通过。

## 7. Day 4：显式数值验证与单核性能证据 MVP

日期：2026-08-29

### 任务

- [x] 新增显式授权的 ordinary `verify-script` MVP；`migrate` 本身继续不执行源码。产品入口只
  接受 hash-bound ordinary report/artifact 和显式 named numeric/bool outputs；定向 harness
  继续承担更强的 strict-NPU provenance 与性能证据。
- [x] source 和 migration artifact 在两个独立子进程执行，并设置 timeout、资源限制和明确的结果边界。
- [x] MVP 要求调用者显式声明可序列化输出变量或使用受审 benchmark adapter；不从任意 stdout 猜测数值。
- [x] 产品 MVP 比较显式 array/scalar leaf 的 shape、dtype、finite、NaN/Inf、整数/布尔精确值
  与浮点容差；nested container 尚未进入当前 receipt schema。
- [x] 报告绑定 source、artifact、runtime/profile、harness 四类 SHA-256。
- [x] benchmark 只支持已有可信结果边界的 direct 样例；通用任意 ordinary script runner 不是强行交付项。
- [x] CPU worker 必须在 Python 启动前用 `taskset --cpu-list N` 绑核，并实测 `os.sched_getaffinity(0)=={N}`；同时记录 BLAS/OMP effective threads=1。
- [x] NPU worker 使用同一 host affinity、单 NPU、显式同步；检查 0 fallback、0 unknown、0 host-assisted。
- [x] 分开报告 cold process end-to-end 与 steady harness；如能分解，再单列 H2D、device-resident、D2H、allocation，不混成一个“算子速度”。
- [x] 预热至少 2 次，测量至少 7 次，报告 median、p10/p90 和样本数；不再使用绝对 min 作为“稳定指标”。
- [x] 对 Case 06 在计时前冻结规模梯度和内存上限，建议从 1M、4M、9M float32 元素开始，并在可用内存 60% 之前停止；禁止看完结果后挑选有利规模。
- [x] 只有 numerical pass + strict profile match + affinity/thread gate 全部通过，才计算 speedup。

### Day 4 产物

- ordinary verify schema、runner 和反例测试。
- Case 03 数值基线；Case 06 大规模参数与 benchmark plan。
- 冷启动/稳态/传输边界明确的 evidence report schema。

### Day 4 验收门

- Case 03 和 Case 06 至少完成可信数值对照；故意制造 shape/dtype/NaN/source/artifact 漂移时必须失败。
- CPU evidence 同时证明进程 affinity 单核和线程库有效线程为 1；缺任一证据只能写“线程受限”，不能写“单核”。
- fallback、host-assisted、数值失败、profile 漂移或计时门失败时：
  `performance_valid=false`、`speedup=null`。
- Case 01 matmul 必须在 NPU performance timing 前拒绝，作为负例保留。

### Day 4 止损线

若普通脚本无法建立可信输出边界，只交付 verify-only 或外部受信、source-bound 的 Case 03/06 定向 harness；绝不为了进度发布不可复核加速比。

### Day 4 实际结果（提前执行：2026-08-26）

- 采用止损线路：新增三个 repo 内、外部受信 runner，而没有把通用源码执行能力塞进 `migrate`。真实 GLM 产物只由 runner 观测执行；框架修复仍由 Codex 完成。
- 7 个已迁移 case 完成显式输出合同验证：`01/03/04/06/08/10` 通过；Case02 的 `$['max_val']` 为 `float32 != float64`，严格失败，未归一化掩盖。
- strict 稳态 Case03/06/10 的 NPU 分别比单核 CPU 慢 `2.68x / 3.76x / 5.28x`；fallback Case01/08 没有 NPU timing，scoped Case02/04 只有诊断数据。
- Case06 冻结规模 adapter 的 device-resident 公共四算子链在 1M/4M/9M 上分别快 `9.89x / 9.47x / 11.44x`；该 adapter 不执行 GLM artifact，因此没有把它写成迁移脚本 E2E 加速。
- fresh-process strict Case03/06/10 的 NPU 分别慢 `19.61x / 21.85x / 19.58x`。初版 cold runner 的 `wait(timeout=...)` 轮询误差已修为阻塞 wait + 独立 watchdog，并以新 harness hash 重跑。
- 性能根因定位到 AsNumpy 公共 wrapper 的额外 D2D 深拷贝/同步、每算子同步、分配固定成本，以及 `clip` 标量路径未释放 ACL scalar handle；五日计划内不直接修改上游 wheel。
- 权威汇总：`docs/evidence/day4-demo-performance-results-20260826.json`；分析：`docs/evidence/day4-demo-performance-analysis-20260826.md`。原始 evidence 使用 write-once 私有路径和 SHA-256 绑定。
- Day4 收口全量为 `1062 passed in 21.07s`；`git diff --check`、JSON parse、三个 runner `py_compile` 和 11 文件定向 secret scan 均通过。

## 8. Day 5：CANNBot 接线、重启模拟与 RC 验收

日期：2026-08-30

### 任务

- [ ] 统一 installer、`numpilot paths`、`setup-skills`、`complete-gaps` 和 `run_poisson_demo.sh` 的 CANNBot exact commit-scoped 路径。
- [ ] 保留 `--with-cannbot --accept-cannbot-license` 显式许可证门；缺 bundle 时只输出可复制安装命令，不隐式 fetch。
- [ ] generation、operator repair、adapter generation/repair 全部使用 Day 1 通用 model contract。
- [ ] 用 fake endpoint 验证 CANNBot Skill 内容、provider/protocol/model、skill commit/bundle hash 进入 receipt，但不在五日主路径生成新算子。
- [ ] 模拟重启：使用临时/空 HOME 与干净 shell，source generated activation script，然后执行 paths、offline bootstrap verify、doctor、setup-skills check、analyze 和 fake migrate。
- [ ] 在用户轮换此前暴露过的 Token、确认模型配置并明确授权后，才运行一次真实 endpoint probe/migration；代理只进入该进程环境，不写配置或仓库。
- [ ] 真实主样例优先 Case 06；Case 03/10 只在时间充足且逐例授权时继续，不批量预生成。
- [ ] 完成 Case 06 数值与规模 sweep；无论 NPU 是否更快，都保存完整结果，不能以结果筛选规模。
- [ ] 更新 README、Fresh Machine Prompt、Notebook、架构图、release checklist 和旧 GLM-only handoff 的当前入口说明。
- [ ] 重新运行全量测试、构建 wheel/sdist、检查包内容、`git diff --check`、secret scan 和 evidence manifest。

### Day 5 最终验收门

- 临时 HOME 中 source 一次即可复用持久安装；全程不重新 clone/build/fetch，offline verify 仍通过。
- CANNBot 自动发现固定 commit `1ef637...`、36 个 bundle 文件和 SHA-256 `11dcc3...`；Poisson preflight 使用同一实际 checkout。
- 没有 CANNBot 时 direct analyze/migrate 仍正常；gap lane 才要求安装/许可证。
- fake endpoint 覆盖所有生成/repair 路径；真实模型步骤若未获用户授权或 endpoint 不可达，必须标为 pending，不得以 fake 回执冒充真实调用。
- Case 06 只有在 strict/numerical/affinity/profile 门全部通过时才发布 speedup；否则如实输出 null。
- repo/config/log/evidence 中没有 Token、带凭据 URL、代理值或模型响应正文。
- 当天重新记录 HEAD、版本、测试数、artifact hash、CANN/SoC 和 build hash，不复制本计划中的基线数字。

### Day 5 止损线

CANNBot 只完成路径、Skill、通用 provider 和证据接线，不扩张为新算子 sprint；任何真实模型或 NPU 硬件不稳定都不能阻塞 direct lane RC，但必须在最终报告中明确标为外部阻塞或 pending。

## 9. 测试矩阵

| 层级 | 必测内容 | 联网/模型/NPU |
|---|---|---|
| 单元 | config schema、model ID、receipt、sanitizer、path precedence、profile resolver | 全部 fake；无网络、无 NPU |
| Transport 合同 | fake Chat Completions、model alias/mismatch/missing、超时、HTTP 4xx/5xx、畸形响应 | 本地 fake server |
| 安全 | Token/URL/代理/response body 脱敏，路径权限，receipt 篡改，write-once | 无网络 |
| 安装 | persistent-root、activation 幂等、CANN 冲突检测、CANNBot license 门 | fresh shell；apply 可显式联网 |
| Capability | 121/114 固定 profile、8 个非 strict 项、nested linalg/random、runtime drift | 本地源码 + receipt |
| Ordinary | 10 demo analyze、direct/fallback/gap/audit/blocked、source/artifact drift | analyze 无模型；migrate 用 fake |
| NPU 数值 | 03/06，必要时 10；shape/dtype/NaN/Inf/容差 | 单 NPU，显式授权执行 |
| 性能 | CPU affinity=1、threadpool=1；NPU strict、同步；cold/steady 分离 | 单核 CPU + 单 NPU |
| CANNBot | fixed checkout/bundle、offline verify、Poisson path、generation receipt | fake model；不生成新算子 |
| 发布 | 全测、build、包内容、docs、secret scan、manifest、clean checkout | Day 5 执行 |

每天收工至少执行并记录实际结果：

```bash
git status --short --branch
git diff --check
python3 -m pytest -q -p no:cacheprovider
```

只有 Day 5 RC 才要求从干净 checkout 重新 build 和 fresh-shell 复验；开发环境中的旧 `dist/`、cache、receipt 或 run 不得复用为发布证据。

## 10. 可发布性能结论的固定门禁

```text
source hash matched
AND migration artifact hash matched
AND runtime/profile/receipt matched
AND strict NPU provenance = true
AND fallback/unknown/host-assisted = 0
AND numerical verification = pass
AND CPU affinity = exactly one logical core
AND CPU thread pools = 1
AND timing boundary + synchronization recorded
AND sample policy frozen before observing results
──────────────────────────────────────────────────
才允许 performance_valid=true 与 speedup=<number>
```

任一条件失败时，报告必须保留原始诊断数据，但结论固定为：

```json
{
  "performance_valid": false,
  "speedup": null
}
```

matmul 负例额外固定：

```json
{
  "classification": "FALLBACK_CONSENT",
  "cpu_fallback_ops": ["matmul"],
  "npu_timing_performed": false,
  "performance_valid": false,
  "speedup": null
}
```

## 11. 优先级、范围与五日外 backlog

### Must：五日必须完成

- 通用 OpenAI-compatible model/receipt 合同；
- 持久激活与 fresh-shell 复用；
- truthful qualified capability；
- ordinary analyze + generic migrate；
- Case 03/06 的可信数值边界；
- CANNBot 路径统一与 reboot RC；
- 文档、测试、secret、build 和证据门禁。

### Should：不阻塞 RC

- 02 的常量/reduction profile；
- 04 的 nested `linalg.det` direct classification；
- Case 10 的额外迁移/数值演示；
- 普通脚本 steady benchmark 的更通用 adapter protocol。

### 五日内明确不做（原冻结范围；2026-08-27 用户追加了 MatMul 定向例外）

- 全 114 API 的完整 dtype/rank/axis/equation 资格化；
- 通用 Python 控制流、CPU/NPU 自动切图或任意 ndarray dunder/method；
- 通用 AsNumpy matmul、dot/vdot dtype guard、allocator 或 INFO 日志的上游修复；后续只增加了
  精确二维 `float32` MatMul runtime，不外推为通用修复；
- 通用自动 integrate→qualify→activate→rescan 状态机；后续增加了只读 ops-nn resolver 和
  一个外部受信编排的 MatMul 案例，仍没有通用自动状态机；
- 多个新算子或 Case 09 的 erf/arange 补全；
- 把 CANNBot 安装成 Codex 本地可发现 Skill；NumPilot 只消费固定只读 bundle；
- 未经用户明确授权的真实模型调用；
- OS 级通用不可信代码沙箱。

## 12. 风险与应对

| 风险 | 应对与停止标准 |
|---|---|
| “OpenAI-compatible”实现差异很大 | 发布最小协议合同和 conformance probe；不满足就明确拒绝，不做静默猜测。 |
| 网关返回 model alias 或缺失 model | 默认 exact；只有显式 alias allowlist 才接受，并把 identity 状态写入 receipt。 |
| 旧 GLM receipt/schema 被破坏 | 提供只读兼容解析和 schema migration 测试；新写入统一使用新版 schema。 |
| Token 已在早先终端输出中暴露 | 任何 Day 5 真实请求前必须轮换；所有日志和 evidence 再做 secret scan。 |
| CANN 动态库顺序重启后漂移 | 生成可复现 activation；doctor 检测重复/冲突 Ascend 路径；fresh shell 做硬验收。 |
| capability 名存在但签名不可用 | 路线与合同分层；未资格化 profile 一律 audit-required。 |
| NPU 冷启动、分配或传输掩盖 kernel | 分开 cold/steady/H2D/device/D2H，不跨口径计算单一加速比。 |
| ordinary 输出边界无法安全抽取 | 要求显式输出变量/adapter；无法证明则止损 verify-only，不解析任意 stdout。 |
| 五天范围再次膨胀到算子生成 | direct lane RC 优先；CANNBot 只做接线与 fake receipt，不生成新算子。 |
| 远端 endpoint/代理不稳定 | Day 1～4 全部使用 fake transport；Day 5 真实调用是显式验收项，失败如实记为外部 pending。 |

## 13. 每日状态记录模板

每天收工在本文件底部追加一段实际记录，不回写或美化失败结果：

```markdown
### YYYY-MM-DD 实际状态

- HEAD / branch：
- 今日完成：
- 未完成与原因：
- 实际测试命令与结果：
- fresh-shell / NPU / fake model / real model 状态：
- 新增或修改的 artifact SHA-256：
- secret scan：
- 明日第一动作：
- 是否触发止损线：
```

## 14. Day 1 启动前动作

1. 从精确 `v0.2.0a2` HEAD 创建功能分支或独立 worktree，保留当前 `.venv/`，不得使用 destructive reset。
2. 把本文档提交前的 `775 passed` 保存为 T0 测试基线；后续每一天重新记录实际数字。
3. 先为 generic model contract 写失败测试，再改生产代码；开发期间只用 fake endpoint。
4. 在任何真实模型请求前由用户轮换此前暴露过的 auth token。
5. 不修改用户 `.bashrc`；Day 1 生成 activation 后，再决定是否让本机 `.codex-env.sh` 显式 source 它。

### 2026-08-26 实际状态

- HEAD / branch：`374e4da7df827bbf69668eb896e86ab1f95fa755` / `feat/day1-generic-model-persistence`；实现尚未提交。
- 今日完成：Day 1 目标 A/B、六项验收门及首轮 README/Fresh Machine/Notebook/架构文档更新均完成；未进入 Day 2。
- 未完成与原因：Day 1 无未完成实现项；Day 1 改动尚未发布，新的不可变 release tag 按计划留到 Day 5。真实 endpoint 调用按计划和用户要求未执行，后续调用前仍须先轮换此前暴露过的 Token 并取得明确授权。
- 实际测试命令与结果：T0 为 `775 passed`；最终工作树全量为 `822 passed in 18.78s`；最终模型合同复核为 `247 passed`，持久化定向集合为 `86 passed in 1.67s`，Notebook 合同为 `1 passed in 0.03s`；`git diff --check` 和 Notebook JSON 校验通过。一次全量回归先暴露 Notebook 旧断言（`1 failed, 814 passed`），同步新 conditional CANNBot 合同后通过；持久化定向回归也先暴露两个旧预期（`2 failed, 84 passed`），修正“保留无关 devlib”和安装器输出预期后通过。首次 fresh-shell 命令误写 `doctor --lane npu`，CLI 在 doctor 前拒绝该参数；改为实际合同 `--backend npu` 后通过。首次临时 secret-scan 命令因 shell 正则引号语法错误未执行，修正后通过。
- fresh-shell / NPU / fake model / real model 状态：空临时 HOME、干净 shell、激活脚本重复 source、离线 AsNumpy/CANNBot bootstrap 与 NPU doctor 均通过；fake OpenAI-compatible 模型合同通过；真实模型未调用。
- 新增或修改的 artifact SHA-256：`/home/ma-user/work/activate-numpilot.sh` = `8b5924308daa123bfa969b35ada0f07a2b947535bb87a571f54e516aac89f1d5`（mode `0600`）。
- secret scan：activation 与生产代码/文档未发现凭据、代理或私钥字面量；测试目录唯一命中是预期拒绝夹具 `user:pass@gateway.example`。
- 明日第一动作：仅在用户要求继续后进入 Day 2，先对固定 AsNumpy 做 registry/API drift 的失败测试与 capability profile；`/home/ma-user/work/demo` 已被用户指定为后续迁移测试集，但 Day 2 只做本地扫描，不提前触发模型或 Day 3 迁移。
- 是否触发止损线：否；identity、alias、content hash 与 receipt 篡改均已 fail-closed 回归，direct-without-CANNBot 正向可用性回归通过。

### 2026-08-27 集成候选实际状态

- HEAD 基线 / branch：`374e4da7df827bbf69668eb896e86ab1f95fa755` /
  `feat/framework-hardening-opsnn-rc`；正在形成提交，尚无不可变 release tag。
- 今日完成：Day 2/3 qualified capability 与 ordinary analyze/migrate、Day 4 定向 evidence、
  official-first resolver，以及超出原五日范围但由用户明确要求的 ops-nn MatMul 受限闭环。
- MatMul 路径固定 ops-nn v9.0.0 `matmul/mat_mul_v3` 来源，优先复用已安装 CANN 9.0.0
  `aclnnMatmul`，没有重建整套 ops-nn 或写系统 OPP；AsNumPy patch 保存在独立仓库和
  hash-bound wheel/qualification receipt 中。
- 实际验证：NumPilot `1198 passed`；AsNumPy `-k matmul` 为 `9 passed, 31 deselected`；
  `git diff --check`、161 项精选 evidence manifest 和 draw.io XML 通过。案例最大绝对误差
  `6.984919309616089e-10`，native call 1、host fallback 0，不发布性能结论。
- 未完成与原因：通用 official-source Artifact/私有 OPP/自动 runner 仍属后续；fresh clone
  build 与不可变 tag 尚未执行；GitCode HTTPS 凭据失效且当前浅克隆没有 upstream，推送必须
  在恢复认证并比较远端后完成，禁止 force push。
