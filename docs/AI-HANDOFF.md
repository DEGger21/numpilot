# NumPilot AI 续接手册

> **当前续接提示（2026-08-22）**：下方 `/home/lixing`、106 tests 和旧 capability 内容是
> 历史设计背景，不是当前状态。权威目录是 `/home/ma-user/work/numpilot`。Poisson 两个
> 缺口算子已由用户 LLM + 固定 CANNBot Skill 定向生成、repair、封存并通过
> 六门资格化；限定 profile 的 AsNumpy `outer/matmul` 已走 NPU；完整动态一轮 117 个调用点
> 和 250 轮设备驻留对拍均通过 strict-NPU。包装层逐返回值深拷贝也已定位并消除，同一
> 10 轮 profiler 的 `aclrtMemcpy` 从 1,235 降到 35，kernel 数保持 1,170。先读
> [`P0-LLM-CANNBOT-RUN-20260821.md`](P0-LLM-CANNBOT-RUN-20260821.md)、
> [`evidence/poisson-ritz-p0-npu-20260821.md`](evidence/poisson-ritz-p0-npu-20260821.md) 和
> [`COMPETITION-GAP-REVIEW-20260821.md`](COMPETITION-GAP-REVIEW-20260821.md)。下一优先级是
> 放大规模、预热和重复样本，不能先发布性能；当前全量回归为 `369 passed`，CodeArts
> 仍按用户决定后置。

> 最后更新：2026-07-18（Asia/Shanghai）  
> 权威工作目录：`/home/lixing/numpilot`  
> 用途：下一位 AI/开发者从当前状态继续，不重新做已经完成的全量审计。

## 0. 续接时先做什么

按顺序读取：

1. 本文；
2. [技术评审与实施复盘](numpilot-technical-review.md)；
3. [当前架构](architecture.md)；
4. [CodeArts 集成建议](codearts-integration.md)；
5. [裸金属 NPU 证据](evidence/npu-smoke-20260718.md)；
6. `README.md` 和待修改模块的测试。

先执行：

```bash
cd /home/lixing/numpilot
git status --short
/home/lixing/np310/bin/python -m pytest -q tests
git diff --check
```

当前工作树包含用户授权的、尚未提交的实现和文档修改。**不要执行 `git reset --hard`、`git checkout --` 或覆盖这些文件；不要自行提交，除非用户明确要求。** `/home/lixing/asnumpyoplogin/.numpilot_stage` 只是本轮受限环境中的暂存副本，后续以 `/home/lixing/numpilot` 为准。

## 1. 项目目标与当前定位

长期目标：构建一个利用 CodeArts 多智能体，把科学计算 NumPy 网络迁移到 AsNumpy/Ascend NPU，并在遇到能力缺口时路由到复用、组合或 Ascend C 算子生成的框架。

当前可信定位：

```text
CodeArts AgentTeam / 项目级 Agent / Skill   协作控制面
                         ↓
NumPilot CLI + contracts + capability       确定性执行/证据面
                         ↓
AsNumpy + CANN + 裸金属 Ascend NPU
```

当前 demo 只承诺一个顶层同步 `forward()` 的完整 NPU 岛。它不是通用图编译器，尚未实现真实 CPU/NPU 子图切分、缺口算子自动生成闭环或 OS 级用户代码隔离。

“补全算子”路线仍然合理，但必须放在 Gap Router 的最后一级：

```text
已有严格 capability → ACLNN → 组合已有 NPU 原语
→ 只有热点且收益合理时生成 Ascend C → 显式 CPU fallback/unsupported
```

不能把“函数名存在”“命中已有 AsNumpy 扩展 API”或“编译成功”当作新算子已经 qualified。

## 2. 本轮已经完成

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

## 3. 当前验证状态

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

## 4. 仍未完成与风险

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

## 5. 建议下一次 AI 直接执行的任务

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
