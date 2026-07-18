# NumPilot 技术评审与比赛版改进建议

> 评审日期：2026-07-18
>
> 评审范围：`/home/lixing/numpilot`、`/home/lixing/asnumpy`、`/home/lixing/asnumpyoplogin`，以及比赛方案和 NumPilot 设计文档。
>
> 文档性质：技术评审 + 2026-07-18 实施复盘。第 4～7 节保留最初问题证据；完成状态以第 1.1、14、16 节为准。

## 1. 执行摘要

NumPilot 的方向合理，也具备比赛辨识度。当前仓库已经形成了一个结构清楚的“契约化、可插拔迁移流水线骨架”：

- `AnalysisResult → OffloadPlan → MigrationResult → VerificationReport` 的 dataclass 契约明确；
- AST/LLM 迁移引擎和 shim/NPU 执行后端可替换；
- strict LLM 模式不会用 AST fallback 掩盖模型失败；
- 环境检查、迁移报告、负例早停和终端向导适合演示。

但当前实现还不能支撑设计文档中的完整表述。它尚未形成真正的数据流图、CPU/NPU 混合卸载、缺口算子生成、AOT/JIT、自动修复或“越用越全”的算子知识库。更准确的当前定位是：

> NumPilot 是一个面向 NumPy/AsNumpy 的混合智能体迁移工作流原型，而不是已经完成的自主多智能体异构迁移系统。

“补全算子”路线是合理且必要的，但应当是缺口处理中的最后一级能力，不应当成为遇到 unsupported API 时的默认动作。项目的核心竞争力应表述为：

> 智能体依据真实能力与成本划分 CPU/NPU，对热点缺口自动选择复用、组合或生成 Ascend C，并在编译、真机差分和性能门禁通过后注册回 AsNumpy。

本次评审发现三个必须优先解决的 P0 问题：

1. 现有 `big_gemm` 12～15× 报告不能证明 NPU 矩阵乘加速；实际使用的 AsNumpy wheel 会把 `matmul` 拉回 CPU，而 NPU 基准进程又未限制 CPU/BLAS 线程。
2. 验证器存在严重假阳性：原输出全零、迁移输出错误时仍可能判定 `pass`；验证流程还绕过了迁移产物自己的 `build_inputs()`。
3. `OffloadPlan` 当前只控制名称替换，没有设备值、边界和数据搬运，因此不能实现报告中声称的混合 CPU/NPU 卸载。

在这些问题修复前，不应继续放大性能、智能卸载和自动算子生成方面的宣传。

### 1.1 本轮实施结论

本轮没有扩张“支持算子数量”，而是先把 demo 收敛到可证伪的可信边界：

- 迁移范围收敛为顶层同步 `forward()` 的完整 NPU 岛，入口只对 NumPy ndarray 做幂等 H2D；不再宣称已实现逐算子混合卸载。
- 验证器改为三层门禁：输入生成一致、同一规范输入的 `forward` 一致、迁移产物自身 `build_inputs + forward` 一致；零参考、shape/dtype、NaN/Inf、嵌套输出和异常都有回归。
- 新增与具体 AsNumpy 版本、模块路径、CANN、SoC 绑定的 capability/provenance；已知 CPU fallback 或来源 unknown 默认 fail closed。
- CPU/NPU benchmark 拆成独立进程，记录实际 BLAS 线程、源码 hash、参考输出和明确计时范围；只有 `performance_valid=true` 才计算 speedup。
- 加入 CodeArts 项目级 Agent/Skill 配置，明确 CodeArts 为协作控制面，NumPilot 为确定性执行和证据面。

仍未解决的核心 P0 是用户代码的 OS 级隔离；当前全局锁只解决进程内 `sys.modules` 注入竞争。缺口算子生成、值级 IR 和真实子图划分仍属于 P1/完整版本。

## 2. 目标框架原理

建议将完整系统主线明确为：

```text
NumPy 项目/程序
    ↓
语义分析 + 值级数据流 IR（op、shape、dtype、layout、device）
    ↓
查询与目标 AsNumpy/CANN 版本绑定的 Capability Registry
    ↓
按计算、搬运、同步、内存成本划分 CPU/NPU 子图
    ↓
Gap Router：复用 ACLNN / 组合已有算子 / 生成 Ascend C / CPU fallback
    ↓
生成可独立运行的迁移产物和部署清单
    ↓
NumPy 差分验证 + 真机 ST + 端到端性能验证
    ↓
门禁通过后注册、缓存；失败则诊断、修复或回滚
```

这条主线比“`np.* → ap.*` 文本替换”更能体现复杂软件工程价值，也更符合比赛对需求、架构、开发、测试、部署全流程闭环的要求。

### 2.1 当前实现与目标的差距

| 环节 | 当前实现 | 与目标的差距 |
|---|---|---|
| 分析 | 只分析顶层 `forward`，并阻断若干未验证签名/dunder | 无完整 qualified name、值级 DFG、shape/dtype/layout 传播 |
| 决策 | 明确采用 `whole_forward_npu` | 尚无调用点级成本模型和真实 CPU/NPU 子图划分 |
| 迁移 | AST 或 strict LLM；forward 入口建立单一 H2D 边界 | 无结构化 patch、动态 guard 和项目级迁移 |
| 缺口处理 | unsupported 后早停 | 无 GapSpec、算子生成、编译、注册和重跑闭环 |
| 验证 | 三层产物语义比较 + 结构化异常 + capability provenance | 尚无 OS 级隔离、kernel trace 和完整 transfer 统计 |
| 能力知识 | 静态 coverage + 版本化 runtime capability manifest | 缺少完整签名、shape/layout 约束和 OPP artifact hash |
| 多智能体 | NumPilot 确定性 Agent 类 + CodeArts 项目级 Agent/Skill | AgentTeam 账号实测、失败再规划和 MCP 尚待接通 |

相关代码：

- [编排器](../numpilot/orchestrator.py)
- [Agent 契约](../numpilot/contracts.py)
- [分析 Agent](../numpilot/agents/analysis.py)
- [决策 Agent](../numpilot/agents/decision.py)
- [AST 迁移引擎](../numpilot/engines/ast_rewrite.py)
- [验证 Agent](../numpilot/agents/verification.py)
- [静态覆盖集](../numpilot/coverage.py)

## 3. 值得保留的设计

以下设计应继续保留并扩展，不建议推倒重来。

### 3.1 契约先行

`contracts.py` 已经给 Agent 之间建立了明确数据边界。下一步应扩展这些契约，而不是让 Agent 通过非结构化自然语言互传全部状态。

### 3.2 策略和端口-适配器

`MigrationEngine`、`ExecutionBackend` 使 AST/LLM、shim/NPU 可以独立替换，适合继续增加：

- `CapabilityProvider`；
- `PlacementPlanner`；
- `GapResolver`；
- `OperatorBuilder`；
- `ArtifactStore`；
- `SandboxRunner`。

### 3.3 确定性组件与 LLM 混合

只有存在语义不确定性的迁移、修复和算子生成环节需要 LLM。解析、编译、测试、精度验收和性能门禁应尽量确定性执行。不要为了“多智能体”而把可靠规则替换成大模型。

### 3.4 strict LLM 语义

CLI 显式选择 LLM 时失败即暴露，是正确做法。后续应进一步要求 LLM 输出结构化 patch 和声明，而不只是检查返回文本中是否包含 `ap.`。

## 4. P0：现有性能证据失真

> 状态：已建立替代门禁。历史 12～15× 结论仍判定为口径失效；新 benchmark 会在 `matmul` 执行前因 CPU fallback 拒绝，并令 `speedup=null`。

历史报告 `reports/migration_big_gemm_20260704_112406.md` 记录：

- AsNumpy 版本：`0.3.1.dev4+g1929a3ba4`；
- NumPy 单核：约 1513 ms；
- “AsNumpy@NPU”：约 100 ms；
- 宣称加速约 15.2×。

但该实际 wheel 的 `matmul` 实现为：

```python
def matmul(x1, x2):
    return _to_asnumpy_array(np.matmul(_as_host_array(x1), _as_host_array(x2)))
```

也就是：

```text
NPUArray → D2H → NumPy matmul@CPU → H2D → NPUArray
```

本机证据路径：

```text
/home/lixing/np310/lib/python3.12/site-packages/asnumpy/linalg/direct.py:32-73
```

与此同时，[bench.py](../numpilot/bench.py)没有限制 NPU 对比进程中的 CPU/BLAS 线程，而 CPU baseline 只用一个核。`big_gemm` 的主要计算是四次矩阵乘，因此现有结果更接近：

```text
NumPy 单核
vs
AsNumpy 内部未限线程的 NumPy + NPU ReLU + 数据搬运
```

它不能归因于 NPU 矩阵乘加速。

### 4.1 立即处理

- 将现有 12～15× 数字标记为“历史结果、口径失效”，不再作为 PPT 头条。
- 为 AsNumpy 增加 `strict_npu` 执行模式；发现 CPU fallback 时基准直接失败。
- 每个算子记录 `backend=aclnn/custom/composite/cpu`、fallback 原因、H2D/D2H 次数和字节数。
- 使用 profiler 或执行 trace 证明热点 kernel 确实下发到 NPU。

### 4.2 新性能报告至少包含四个口径

1. NumPy 单核基线；
2. NumPy 固定多核基线，并记录实际 BLAS 线程数；
3. NPU kernel-only：输入常驻设备，不含 H2D/D2H；
4. NPU end-to-end：包含输入搬运、计算和结果取回。

所有口径应记录：

- CPU 型号、核数和 BLAS；
- NPU 型号、CANN、AsNumpy commit；
- shape/dtype；
- warmup 和迭代次数；
- 中位数、P90/P95，而不只展示最好一次；
- 是否同步；
- 峰值内存；
- CPU fallback 和传输统计。

验收标准：在 `strict_npu` 下无 CPU fallback，profiler 能看到目标 NPU kernel，再讨论 NPU 加速比。

## 5. P0：验证器存在假阳性并绕过交付产物

> 状态：已修复并加入补偿性输入、零参考、结构/dtype、NaN/Inf、嵌套输出、异常和产物自身输入构造回归。以下保留最初反例和修复依据。

### 5.1 零分母假阳性

[verification.py](../numpilot/agents/verification.py)当前逻辑近似为：

```python
denom = np.max(np.abs(out_orig))
max_rel = diff / denom if denom > 0 else 0.0
```

当原输出全零时，无论迁移输出是否错误，`max_rel` 都会被设为 0。评审中构造“原输出全零、迁移输出全一”的例子，仍得到 `status=pass`。

### 5.2 验证没有运行迁移后的 `build_inputs()`

当前验证流程：

1. 从原始源码调用 `build_inputs()`；
2. harness 将所有输入统一 `to_device`；
3. 只从迁移 namespace 中取 `forward()`；
4. 比较两个 `forward()` 输出。

AST 引擎却会改写整个文件，而不是只改写 `forward()`。因此，迁移后的 `build_inputs()` 可能同时生成 NPUArray 和 NumPy 数组，交付的 `migrated.py` 独立执行会失败，但验证仍能通过。

### 5.3 修复要求

验证至少分成两层：

- kernel-only：外部 harness 构造相同输入，仅验证迁移计算主体；
- artifact end-to-end：在隔离子进程中直接执行交付产物自己的输入、计算和输出流程。

比较规则：

- 先检查输出结构、shape 和 dtype；
- 浮点按实际输出 dtype 使用 `atol + rtol`；
- 整数、布尔精确比较；
- 显式处理零值、NaN、Inf、复数；
- 支持 tuple/list/dict 等嵌套输出；
- 捕获异常类型、消息和 traceback；
- 报告实际使用的阈值与失败索引。

应新增以下回归测试：

```text
test_verifier_rejects_zero_reference_mismatch
test_verifier_checks_shape_and_dtype
test_verifier_handles_nan_inf_and_nested_outputs
test_migrated_artifact_runs_its_own_build_inputs
test_migrated_runtime_exception_is_reported
```

验收标准：故意把迁移实现改成全零、全一、错 shape、错 dtype 时，验证必须稳定失败。

## 6. P0：OffloadPlan 尚未实现真实混合卸载

> 状态：未实现完整混合卸载；demo 已选择下述短期路径 `whole_forward_npu`，并停止产生误导性的 mixed placement。值级 IR/子图划分仍待实现。

[contracts.py](../numpilot/contracts.py)中的 `OffloadPlan` 只有：

```text
op name → CPU | NPU
```

[decision.py](../numpilot/agents/decision.py)按算子名聚合所有调用点；[ast_rewrite.py](../numpilot/engines/ast_rewrite.py)只把标记为 NPU 的名字替换为 `ap.*`。系统没有：

- 值属于 CPU 还是 NPU 的信息；
- producer/consumer 边；
- CPU/NPU 边界；
- `from_numpy()` / `to_numpy()` 插入；
- 同一个算子不同调用点的独立 placement；
- 连续 NPU 岛和传输合并。

例如：

```python
y = np.zeros((2, 2))       # plan: CPU
return np.add(x, y)        # plan: NPU
```

当前产物近似为：

```python
y = np.zeros((2, 2))
return ap.add(x, y)
```

`y` 仍是 NumPy 数组，原生 AsNumpy 二元算子可能因此报错。需要特别区分：当前版本的 `matmul` 会接收类似混合输入，但原因是它把数据拉回 host 走隐藏 CPU fallback，并不代表混合卸载已经成立。

### 6.1 两种可选路径

短期 MVP：

- 将整个 `forward` 作为一个 NPU 岛；
- 所有输入在函数边界一次性 H2D，输出一次性 D2H；
- 暂时删除“逐算子智能混合卸载”宣传。

完整版本：

- 建立 callsite/value 级 IR；
- 节点记录 op、inputs、outputs、shape、dtype、layout、device；
- placement 目标函数同时包含计算、launch、同步、传输和内存成本；
- 按子图而不是按单个算子决定设备；
- 对动态 shape 生成 guard 和运行时分支；
- 由 codegen 显式生成并合并 transfer。

验收标准：报告里的每个 placement 都能在生成代码和执行 trace 中找到对应设备和 transfer 证据。

## 7. P0：Capability Registry 不能只是算子名集合

> 状态：已实现第一版版本化 runtime manifest 和 fail-closed provenance；签名、shape/layout、OPP hash 与 profiler trace 仍需继续补齐。

[coverage.py](../numpilot/coverage.py)目前是静态 `frozenset`，而“函数名存在”并不等于“目标输入能在目标 NPU 上正确高效执行”。至少需要描述：

```yaml
numpy_symbol: numpy.matmul
signature: matmul(x1, x2)
implementation: aclnn | custom | composite | cpu
supported_dtypes: [float16, float32]
rank_constraints: ...
shape_constraints: ...
broadcast: ...
layout_constraints: ...
soc: Ascend910B
cann_version: ...
asnumpy_commit: ...
semantic_status: verified
performance_status: qualified
fallback_reason: null
artifact_hash: ...
```

评审时，静态 snapshot 与默认 `/home/lixing/np310` wheel 已发生漂移；snapshot 还包含 wheel 没有的 `angle`。因此 registry 必须：

- 从实际目标 wheel/build 生成；
- 绑定 AsNumpy commit、CANN、SoC 和自定义 OPP manifest；
- 启动时检测漂移并 fail-fast；
- 区分 API 存在、语义通过、真 NPU 执行和性能合格；
- 区分 native、custom、composite 和 CPU fallback。

应新增：

```text
test_capability_registry_rejects_version_drift
test_strict_npu_rejects_cpu_fallback
test_capability_matches_signature_dtype_and_shape
```

## 8. 自定义算子路线评审

### 8.1 当前 `asnumpyoplogin` demo 设计上覆盖什么

当前 demo 没有保存提供 `aclnnAngleV2*` 接口的 `angle_v2` 算子工程实现源码，例如 `op_host`、`op_kernel`、Tiling 和 `op_api`；仓内保存的是调用该接口的 AsNumpy adapter。构建脚本会浅克隆 `ops-math`，编译其已经存在的 `angle_v2`。

从代码设计看，它旨在支持以下链路；只有补齐可复现构建日志、产物 manifest 和真机测试报告后，才能将其作为已经验证的能力：

- 已有 ops-math 算子可以构建为自定义 OPP；
- `libcust_opapi.so` 可以链接到 AsNumpy；
- Python 可以暴露 `ap.angle`；
- NumPy oracle 可以用于最外层结果比较。

即使上述链路完成，它仍不能证明：

- LLM 生成了 Ascend C；
- 自动生成了 OpDef、InferShape、Tiling、Kernel 和 ACLNN；
- 自动完成了 UT/ST；
- 不重编译 AsNumpy 即可热插拔；
- 算子库已经“越用越全”。

比赛文案应把这一步称为：

> 现有自定义算子的发现、复用、打包与 AsNumpy 接入。

不要称为“AI 从零生成自定义算子”。

### 8.2 合理的五级 Gap Router

遇到缺口时按以下顺序处理：

1. AsNumpy 已有且满足签名约束：直接复用；
2. CANN 已有 ACLNN、AsNumpy 未封装：生成 adapter；
3. 可由已有 NPU 原语组合或融合：生成 composite/fused 实现；
4. profiler 证明是热点且前三者不可用：生成 Ascend C；
5. 冷门或收益不足：显式 CPU fallback，记录原因和搬运。

### 8.3 新算子晋升门禁

一个生成算子只有完成以下链路，才能注册回能力库：

```text
GapSpec
→ 语义、签名、shape/dtype 规则和 reference cases
→ OpDef / InferShape / Tiling / Kernel / ACLNN
→ 编译
→ Host/Tiling/Kernel UT
→ 真机 ST
→ NumPy 差分、边界、随机和变形测试
→ 性能与资源门禁
→ 版本化打包、签名、注册和缓存
```

失败时必须保留日志，回到诊断/修复 Agent；达到重试上限后回滚或显式 fallback，不能静默注册。

建议增加三个核心契约：

```text
GapSpec
  - qualified op、签名、属性、shape/dtype、语义来源、reference inputs

OperatorArtifact
  - 源码树、构建产物、target SoC/CANN、hash、测试证据、性能证据

CapabilityRecord
  - 适用约束、实现类型、artifact 引用、验证状态、fallback 策略
```

### 8.4 构建安全与可复现性

当前脚本按分支而非 commit 克隆、修改外部源码、可能安装到共享 OPP 路径，测试缺失时还可能跳过后继续成功。自动生成阶段必须改为：

- 固定源码 commit 和工具链版本；
- 每次任务使用独立工作区和独立 OPP 安装目录；
- 生成代码在无网络、非 root、限时限资源的容器中构建；
- SDK 只读挂载，禁止任意 shell 和宿主机安装；
- build、UT、ST、精度任一失败即 hard fail；
- artifact 使用内容 hash 和 manifest；
- 原子激活，失败可回滚；
- 不直接污染 `/usr/local/Ascend` 或共享开发环境。

## 9. AsNumpy 作为底座的改进顺序

AsNumpy 适合作为比赛原型的 eager NPU 适配层，但当前还不是完整的 ndarray 运行时。补大量冷门 API 前，应优先修复执行底座。

### 9.1 P0：消除额外设备深拷贝

Python `ndarray(_core_result)` 会调用 C++ 深拷贝构造；`from_numpy` 也会先 H2D，再因 Python 子类包装触发 D2D 深拷贝和同步。应提供 move/adopt 或直接暴露统一 pybind 类型。

### 9.2 P0：取消逐算子全设备同步

通用 unary/binary executor 每次调用后都 `aclrtSynchronizeDevice()`，会破坏网络级流水和并发。应改成 stream/event 依赖，只在 `to_numpy()` 或显式同步点等待。

### 9.3 P0：修复数据安全和正确性

- float16 D2H 路径存在输出缓冲区大小风险；
- 非连续 NumPy 输入没有处理 strides；
- CPU fallback 不透明；
- 导入包即初始化 device 0，退出时无条件 reset/finalize。

### 9.4 P1：补结构型 ndarray 能力

科学计算程序更依赖：

- `array/asarray/arange`；
- reshape、transpose、view；
- 索引、切片；
- astype、copy；
- concatenate/stack；
- 算术 dunder；
- DLPack 或稳定的设备互操作协议。

这些通常比继续补一批冷门逐元素函数更重要。

### 9.5 P1：建立可观察 fallback

建议至少提供：

- `strict_npu`：没有 NPU 实现立即失败；
- `allow_fallback`：允许 CPU，但必须记录 provenance 和传输；
- `analyze_only`：仅返回无法下沉的原因和能力缺口。

## 10. 多智能体闭环建议

当前 Orchestrator 只是顺序调用四个组件；Wizard 和 bench 又各自复制了一套编排逻辑。建议统一为一个状态机/事件流：

```text
ANALYZE
→ PLAN
→ RESOLVE_GAPS
→ MIGRATE
→ BUILD
→ VERIFY
→ BENCHMARK
→ PROMOTE | REPAIR | FALLBACK | FAIL
```

每个阶段：

- 输入输出使用结构化契约；
- 产物写入统一 ArtifactStore；
- 有 checkpoint、状态码和可恢复重试；
- Wizard、CLI、Web 和 bench 只订阅同一事件流；
- 验证失败生成 `FailureDiagnosis`，反馈给迁移或算子 Agent；
- 确定性 verifier 决定是否晋升，LLM 无权自行宣布成功。

准确的“多智能体”价值不在 Agent 数量，而在不同职责、工具权限和验收权的隔离。

## 11. 执行安全

当前验证器会在主进程直接 `exec` 用户源码和迁移源码；LLM 返回只做很弱的字符串检查后也可能进入执行。真实产品或云端演示必须：

- 在隔离子进程或容器运行用户代码；
- 禁止网络或使用域名 allowlist；
- 限制 CPU、内存、磁盘、进程数和执行时间；
- 使用只读输入和临时输出目录；
- 不挂载宿主凭据；
- 校验 AST/import/系统调用策略；
- TLS 默认验证证书，不能默认 insecure；
- 保存审计日志和 artifact hash。

## 12. 比赛版最小可信闭环

在 2026-09-05 前不建议同时追求 AOT、JIT、多芯片和几十个生成算子。应集中完成一个真实纵向用例。

推荐用例：真实遥感栅格、二维热传导或有限差分程序。程序包含一个真实缺口，例如 `np.diff`、gradient/stencil 或适合融合的邻域计算。

完整演示：

1. 上传或选择原始 NumPy 程序和真实数据；
2. 展示值级图、shape/dtype 和能力匹配；
3. 识别一个热点缺口并产生 `GapSpec`；
4. Gap Router 说明为何不能复用/组合，决定生成 Ascend C；
5. 展示 CodeArts 辅助生成多文件 patch；
6. 在隔离环境完成 build、UT、ST、NumPy 差分和性能门禁；
7. 注册回 capability registry；
8. 原程序自动重跑，从 unsupported/CPU fallback 变为连续 NPU 岛；
9. 报告展示执行 backend、transfer、数值、性能和 artifact provenance；
10. 演示失败算子不会晋升，并可回滚。

这比展示多个“自定义算子标签”更有说服力。

## 13. 比赛材料建议

### 13.1 CodeArts 证据

赛题完整性占 50 分，且要求展示 CodeArts 在需求分析、代码生成、知识问答、单元测试、Codebase 索引、规范驱动开发和部署中的使用记录。

当前运行时调用 GLM 不等于已经满足 CodeArts 要求。从现在起应保存：

- CodeArts 需求/规格文档；
- Codebase 索引和问答记录；
- 生成或改造的核心 diff；
- 单测生成和失败修复记录；
- PR、review、流水线、部署记录；
- 关键决策和人工验收证据。

最好让同一个真实缺口算子贯穿这些材料，而不是零散截图。

本仓库现已增加 `.codeartsdoer/agents/` 和 `.codeartsdoer/skills/`。具体角色映射、官方能力边界、Remote SSH/NPU 演示流程及 MCP/Hook 后续路线见 [CodeArts 集成建议](codearts-integration.md)。建议采用“CodeArts AgentTeam 协作控制面 + NumPilot 可审计执行面”，不要把普通 Python 类包装成多个 Agent 后就宣称已经完成 CodeArts 多智能体接入。

### 13.2 真实性

当前 toy MLP 和纯 GEMM 更适合技术冒烟，不足以支撑高分真实性。需要：

- 真实或可追溯数据；
- 明确科研任务和使用方；
- 原始程序来源和许可证；
- 迁移前实际痛点；
- 端到端效果和失败边界；
- 华为云上的可复现部署或演示环境。

### 13.3 知识产权表述

AsNumpy 使用 Apache-2.0，并有团队、社区和华为相关贡献。PPT 中“本团队自研、具有自主知识产权”应以实际权属材料为准。更稳妥的表述是：

> 团队主导或参与开发的 Apache-2.0 开源项目，并明确列出 upstream、贡献者和本次自研增量。

同时对 NumPy、AsNumpy、ops-math、模型和其他依赖做完整第三方披露。

### 13.4 用户体验

终端向导可以保留，但建议增加一个轻量 Web 页面，重点展示：

- 上传源码；
- 数据流图和设备划分；
- Agent/状态机时间线；
- 迁移 diff；
- 算子生成与门禁过程；
- 原始/迁移结果对比；
- backend、transfer 和性能报告；
- artifact 下载与复现命令。

## 14. 分阶段任务清单

### P0：先让正确性和证据可信

- [x] 修复零分母、shape/dtype、NaN/Inf 和嵌套输出验证。
- [x] 直接运行迁移产物自己的 `build_inputs + forward`，并增加同一规范输入门禁。
- [x] 在 README/新报告中标记历史 12～15× 口径失效；新流程不会复用旧结论。
- [x] 增加 `strict_npu` 和逐算子静态 provenance；kernel trace 仍待 profiler 接入。
- [x] 将 MVP 明确成全 `forward` NPU 岛，不再声称已实现混合 transfer。
- [x] Capability Registry 与目标 wheel/CANN/SoC 绑定并检测漂移。
- [ ] 用户源码和生成代码在隔离子进程/容器中执行。
- [x] 将“自定义 NPU 算子数”改成“命中已有 AsNumpy 扩展实现数”；本次新生成数单列。

### P1：完成比赛差异化闭环

- [ ] 引入 `GapSpec / OperatorArtifact / CapabilityRecord`。
- [ ] 完成一个真正生成的 Ascend C 算子纵向闭环。
- [ ] build、UT、ST、精度和性能全部作为 hard gate。
- [ ] 验证失败反馈给迁移/算子 Agent 并自动重试。
- [ ] 统一 Orchestrator、Wizard 和 bench 三套流程。
- [ ] 使用一个真实科学计算任务和可追溯数据。
- [ ] 保存完整 CodeArts 开发证据并部署到华为云。

### P2：比赛后或有余力再做

- [ ] 调用点级图划分和学习型成本模型。
- [ ] 动态 shape guard 和按 profile 运行时选择。
- [ ] 真正的 JIT 构建缓存。
- [ ] 多 SoC、多 CANN 版本适配。
- [ ] AsNumpy 稳定插件 ABI、DLPack 和不重编译核心的热插拔。
- [ ] 多项目知识库、算子去重和长期质量回归。

## 15. 建议的验收指标

比赛报告不要只展示“替换了多少 API”，应至少包含：

| 指标 | 含义 |
|---|---|
| 语义通过率 | shape、dtype、数值、异常均一致的用例比例 |
| 真 NPU 覆盖率 | 实际在 ACLNN/custom kernel 执行的热点占比 |
| CPU fallback | 次数、原因、耗时占比 |
| H2D/D2H | 次数、总字节、耗时占比 |
| 独立产物成功率 | `migrated.py` 或部署包可独立运行的比例 |
| 算子生成成功率 | 从 GapSpec 到全部门禁通过的比例 |
| 修复成功率 | 首次失败后自动诊断、修复并通过的比例 |
| 端到端加速 | 对合理 CPU 基线的整体收益 |
| Kernel 加速 | 输入驻留设备时目标算子的收益 |
| 复现成功率 | 在干净环境按 manifest 重建成功的比例 |

## 16. 测试复现记录

本轮已按上述 P0 计划修改实现并直接在裸金属 NPU 上复核，没有使用 Docker。

- 环境：8 张 Ascend 910 均为 OK；CANN `8.5.0`；运行架构 `dav-2201`；AsNumpy `0.3.1.dev4+g1929a3ba4`；Python `3.12.13`。
- 单元测试：`/home/lixing/np310/bin/python -m pytest -q tests` → **106 passed**。系统 Python 3.9 缺 `rich`，但已单独验证核心 strict forward 审计兼容 Python `3.9.9`。
- strict-NPU 正例：`elementwise_forward.py` 的 `add/square/sqrt/tanh` 通过版本化 capability 和三层语义门禁；最大绝对差 `1.192e-07`。
- 独立评测：CPU BLAS 实际线程数 1，CPU `0.01498 ms`；设备输入驻留且每轮 D2H 同步的 NPU 口径 `0.41934 ms`，`performance_valid=true`，NPU/CPU 比值 `0.0357×`，即 CPU 约快 `27.99×`。这个负收益结果被如实保留。
- fallback 负例：`mlp_forward.py` 在执行前因 `matmul` 是已知 CPU fallback 被拒绝，退出码 3、`performance_valid=false`、`speedup=null`。
- 证据路径：`/tmp/numpilot-final-bench-20260718/`、`/tmp/numpilot-final-matmul-rejection-20260718/`；仓库内复现摘要见 [NPU smoke 证据](evidence/npu-smoke-20260718.md)。
- 尚未验证/实现：OS 级隔离、真实混合 transfer、profiler kernel trace、缺口算子自动生成闭环。

## 17. 最终建议

项目不需要推倒重来。最值得做的是把已有模块化骨架升级成“证据驱动的迁移闭环”：

1. 先修可信验证和执行 provenance；
2. 再让 placement 变成真实设备图；
3. 用 Capability Registry 驱动 Gap Router；
4. 做成一个真正生成算子的真实科学场景纵向闭环；
5. 用 CodeArts 开发证据、华为云部署和可复现报告完成比赛叙事。

只要这条纵向闭环真实跑通，即使暂时只有一个生成算子，也比宣称几十个尚未被严格验证的能力更有竞争力。
