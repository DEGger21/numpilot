# NumPilot 参赛差距审计与 Poisson Ritz 主线建议

> 审计日期：2026-08-21 UTC  
> 参赛题目：XH-202618《基于华为云码道（CodeArts）代码智能体解决复杂软件工程问题》  
> 提交截止：2026-09-05  
> 审计范围：
>
> - `/home/ma-user/work/numpilot`
> - `/home/ma-user/work/numpilot-gap-runs`
> - `/home/ma-user/work/numpilot-runs`
> - `/home/ma-user/work/numpyexample/poisson_ritz_example`
> - 参赛手册与 NumPilot 原始方案 PDF

## 1. 总结论

> P0 更新（2026-08-21）：Poisson 动态迭代契约、两个 `OperatorSpec`、官方 CANNBot Skill
> 固定版本注入以及用户 `llm.json` 的两算子真实远端直接生成已经完成；
> 两份最终 repair 链已封存为新的不可变 Artifact，并通过纯 NumPy oracle、Host/API、
> Ascend910B kernel 编译、私有 OPP 安装、public AsNumpy API 和真机 NPU 数值对拍。研发过程
> 保留 73 份内容哈希绑定的成功远端回执，两条最终谱系绑定其中 57 份。联合真机测试
> `28 passed`。两算子随后从封存 Artifact 重新构建并通过正式 schema v2 六门资格化，
> v9 wheel 对 `outer`/`matmul` 的限定 float32 rank-1/2 NPU 路径已做源文件/二进制哈希
> 审计，v5 capability overlay 已原子发布。显式加载后，包含 RBF、Laplacian、损失、解析
> 梯度和更新的一轮完整动态 `forward` 共 117 个调用点通过 strict-NPU；同一迁移产物又完成
> 250 轮参数驻留设备对拍，实际 29,250 次调用零 profile 拒绝，最终 CPU/NPU 最大绝对误差
> `4.76837158203125e-07`。这证明确定性缩小案例的动态迭代与驻留执行，不证明原始大配置的
> 比赛级性能收益。详见
> [`P0-LLM-CANNBOT-RUN-20260821.md`](P0-LLM-CANNBOT-RUN-20260821.md)。按当前决策，
> CodeArts 接入暂时后置，不属于这一轮 LLM 链路的完成条件。

NumPilot 已经具备较扎实的“可信迁移与证据门禁”基础，代码质量和测试状态良好；当前距离参赛交付的主要差距，不是继续增加零散功能，而是完成一条真实、可复现、由 CodeArts 留痕的纵向闭环。

建议把 `poisson_ritz_example` 确定为主参赛案例，热扩散案例作为已有 strict-NPU 证据，`cbrt` 作为缺口生成研发支线。Poisson 主线按当前设计定位为“模型扫描接口缺口，基于官方 CANNBot Skill 定向生成本地算子候选，再经六门资格化激活”。ops-nn 不是首轮依赖，只作为后续可选对照或回退来源。

最终应形成如下闭环：

```text
Poisson Ritz 科学计算需求
→ NumPilot 分析一个完整动态训练轮
→ 识别两个领域算子缺口
→ 用户配置的 LLM 读取已验证 CANNBot Skill 并定向生成本地候选
→ 隔离构建、将诊断回传模型并修复 Host/Kernel/API/test/adapter
→ 私有 OPP 构建和六门资格化
→ Capability overlay 原子激活
→ 迁移、数值验证、Profiler 和真实性能报告
→ CodeArts 过程记录、PPT、视频与可复现代码包
```

## 2. 上次实际修改进度

“上次修改”应按 2026-08-13～14 的工作区、交接文档和外部证据判断，不能按仓库最后一次 Git 提交判断。

### 2.1 NumPilot 仓库

- 当前分支：`feat/numpilot-demo`。
- 已跟踪文件约有 4,500 行净新增尚未提交，另有缺口补全、资格化、LLM 配置、测试和证据文档尚未跟踪。
- 当前全量测试结果：`369 passed`。
- `git diff --check` 通过。
- `bash -n run_demo.sh` 通过。
- Python `compileall` 通过。
- `heat_diffusion_forward.py` 的 shim 迁移重新验证通过，7 个 NumPy 调用全部改写且三层数值比较为零误差。

### 2.2 热扩散真机证据

`/home/ma-user/work/numpilot-runs/heat-diffusion-20260813` 中的关键证据仍完整，文件哈希与仓库内留档一致。

主结果：

- Ascend 910B4，CANN 9.0.0；
- `add×4`、`multiply×2`、`subtract×1` 全部通过 strict-NPU；
- 无 CPU fallback；
- 三层数值比较最大绝对误差均为 `0.0`；
- CPU 中位时延 `15.552298980765045 ms`；
- NPU 中位时延 `14.424319961108267 ms`；
- `performance_valid=true`；
- NPU/CPU 比值 `1.07819980579314×`。

该结果证明当前可信迁移链可以工作，但负载只是二维热扩散的单步网络，不是完整时间积分求解器，性能优势也较小，不能外推为通用加速结论。

详见 [`docs/evidence/heat-diffusion-npu-20260813.md`](evidence/heat-diffusion-npu-20260813.md)。

### 2.3 `cbrt` 缺口补全进度

`/home/ma-user/work/numpilot-gap-runs` 中已完成：

- 直接 NumPy API 缺口识别；
- GapSpec 与隔离生成契约；
- 真实远程 LLM 调用回执与失败关闭；
- qualification evidence schema；
- capability overlay、CAS/revision/parent hash；
- profile-aware runtime guard；
- 私有 AsNumpy ACLNN 复合 `cbrt` 真机探针。

复合 `cbrt` 使用：

```text
Abs → PowTensorScalar(1/3) → Sign → Mul → EqScalar(0) → SWhere
```

真机 oracle 共 200,015 个 float32 case，特殊值、符号位和边界长度通过。但它不是单个 Ascend C custom kernel，也没有进入完整 qualification，因此 `numpy.cbrt` 仍应保持 `generated_unverified`。

`cbrt-qualification-20260813/operator-project` 中的 Ascend C kernel 仍是 `TODO`，目标架构和 tiling 也未完成，不能声称自定义算子闭环已完成。

### 2.4 当前机器状态

普通开发 Shell 中：

- `npu-smi` 会因动态库路径污染出现 `undefined symbol: drvSetDeviceInfo`；
- 默认 Python 找不到 `asnumpy`；
- `python -m numpilot doctor --backend npu` 因此失败。

使用清洁环境、固定 CANN 9.0.0 和审计 AsNumpy wheel 后：

- `npu-smi` 能识别 NPU ID 7；
- 真实 AsNumpy `sqrt` 在 NPU 上执行并通过 NumPy oracle。

结论是硬件和 NPU 运行链仍可用，当前缺陷主要是环境封装和可复现部署不足。

## 3. Poisson Ritz 案例审计

主案例位于：

```text
/home/ma-user/work/numpyexample/poisson_ritz_example/
```

它使用 RBF 网络和配点 Ritz/PINN 方法求解二维 Poisson 方程，并引入带异常值的 SmoothL1 鲁棒损失。两个补充算子为：

| 算子 | 用途 | 原案例来源 | NumPilot 当前路线 |
|---|---|---|---|
| `smooth_l1_loss` | PDE 与边界残差鲁棒损失 | 原案例随附的补充算子样例 | LLM + CANNBot Skill 直接生成本地候选 |
| `apply_gradient_descent` | 参数梯度下降更新 | 原案例随附的补充算子样例 | LLM + CANNBot Skill 直接生成本地候选 |

AsNumpy 已包含 Python API、pybind、C++ ACLNN adapter、测试和构建脚本，案例文档还记录了 3 个规模下的历史结果，最高声称 `3.09×`。

### 3.1 原始完整程序仍不能直接交给 NumPilot

实际执行：

```bash
python -m numpilot migrate \
  /home/ma-user/work/numpyexample/poisson_ritz_example/poisson_ritz_example/poisson_ritz_example/poisson_ritz_numpy.py \
  --backend shim --skip-env-check
```

结果为：

```text
0 处 numpy 调用
0 个算子
无法迁移
退出码 1
```

原因如下：

1. 源程序没有 NumPilot 当前要求的唯一顶层 `forward()`；
2. 求解主体位于 `solve()` 循环中；
3. 数值过程拆分成多个 helper；
4. 包含切片、Python 运算符、循环和控制流；
5. 两个补充算子是普通 Python helper，不是 `numpy.<api>`；
6. 原文件没有当前 NumPilot 需要的显式 `OperatorSpec`。

因此不应直接扩张成通用循环/任意 helper 编译器。仓库先增加了
`examples/poisson_ritz_step.py` 验证两算子链路，随后增加
`examples/poisson_ritz_iteration.py`：RBF、Laplacian、损失、解析梯度与参数更新全部位于
唯一的 straight-line `forward`，外部 runner 只负责多轮调用和设备状态接力。

### 3.2 两个领域算子契约已实现

`examples/poisson_ritz_step.py` 已用 `@operator_spec` 显式声明：

- 稳定语义名称和版本；
- NumPy oracle；
- 目标 AsNumpy API；
- positional/keyword signature variants；
- dtype、rank、shape、contiguous 约束；
- reduction、beta 等属性约束；
- 来源、许可证、目标 SoC/CANN；
- `direct_generation` 主路由及后续复用/对照扩展点。

AnalysisAgent、AST rewrite、GapSpec 和 qualification 已能保留 `operator.<name>` 身份，而不是把 helper 伪造成 NumPy API。编译/运行诊断回传、逐文件 repair、父候选哈希接力、不可变 Artifact 封存、kernel 编译、私有 OPP 安装、public AsNumpy 接线、正式六门 evidence、overlay 激活、完整动态一轮、250 轮驻留设备对拍，以及缩小案例的 profiler/包装层复制闭环均已完成。当前未完成的是接近原始规模的预热、重复性能样本与发布门禁。

### 3.3 当前 Poisson 性能结论存在真实性风险

现有 `BENCHMARK_RESULTS.md` 不能直接作为比赛级性能证据，原因包括：

1. 历史 benchmark 没有绑定当前已修复的 v9 wheel；当前修复只在限定 float32、非空、
   rank 1/2 profile 内获得 strict-NPU 准入，不能替历史数据补证；
2. `poisson_ritz_asnumpy.py` 每轮至少回传 `loss_pde`、`loss_bc` 和 `grad_norm`；
3. 代码允许 AsNumpy 导入失败后自动回退 NumPy，但 benchmark 仍会把第二段标成 NPU；
4. benchmark 未强制检查 `has_asnumpy=true`；
5. CPU 线程数量未固定和实测，环境文档只写“大概等效 24 核，不准确”；
6. 缺少 warmup、重复样本、中位数、原始 JSON 和源码/产物哈希；
7. 中等规模的 CPU/NPU L2 结果差异明显，没有硬性一致性门禁；
8. 缺少 kernel/profiler、传输字节和 fallback 证据；
9. “超过 50K 点预计 5～10×”没有实测依据，应删除。

在这些问题修复前，只能称它为“AsNumpy 混合路径历史测量”，不能称为 strict-NPU 性能结论。

## 4. 参赛手册逐项差距

参赛手册要求提交完整的“技术方案 + 可运行产品”，评审维度为作品完整性 50 分、应用创新性 25 分、应用真实性 25 分。

### 4.1 作品完整性

| 手册要求 | 当前状态 | 需要补齐 |
|---|---|---|
| 明确场景和核心痛点 | 有科学计算迁移定位，但案例叙事分散 | 统一成 Poisson Ritz 主案例与两个算子缺口 |
| 需求分析与 AI 原生设计 | 有架构文档和早期 PDF | 更新为当前真实能力，去除未实现承诺 |
| CodeArts 全流程使用记录 | 只有 `.codeartsdoer` 配置 | 保存真实需求、问答、代码生成、测试、调试、Diff、任务图和部署记录 |
| 代码生成记录 | 无比赛账号中的完整留痕 | 后续核心改动必须在 CodeArts 内完成并截图/导出 |
| 研发知识问答 | 未形成正式材料 | 留存 CANNBot/CANN/AsNumpy 检索问题、来源和决策；ops-nn 仅标注为可选对照 |
| 单元测试生成 | 测试很多，但无 CodeArts 生成证据 | 用 CodeArts 生成并接受关键正反测试 |
| Codebase 索引 | 未验证，当前 Python 还受官方语言限制 | 向赛方确认；必要时对 TypeScript/Go MCP 控制层使用索引 |
| 规范驱动开发 | 有 ADR、contract 和 GapSpec | 在 CodeArts 中用正式 spec 驱动 Poisson 闭环并留痕 |
| 技术架构与部署方案 | 有架构，无可交付部署 | 提供私有 OPP、AsNumpy wheel、NumPilot、CodeArts 的部署图和脚本 |
| 测试报告 | 有测试和零散证据 | 生成正式 JUnit/coverage/NPU/profiler/数值/性能报告 |
| 演示视频及说明 | 审计范围内未发现 | 制作完整演示脚本和视频 |
| 完整程序代码 ZIP | 尚未形成发布包 | 固定版本、依赖、许可证、制品哈希和复现命令 |

### 4.2 应用创新性

当前真正有竞争力的创新点是：

- CodeArts 多智能体控制面与 NumPilot 确定性证据面分离；
- 不把 API 存在、shim 通过或 Python wrapper 调用冒充真实 NPU；
- 缺口算子经 GapSpec、资格化证据和 capability overlay 才能激活；
- CPU fallback、unknown provenance、数值失败时禁止发布加速比；
- 能展示小规模 CPU 更快、中大规模 NPU 可能获益的真实性能交叉点。

当前两个 Poisson 算子已经由 NumPilot 自动发现、定向生成、资格化并在迁移产物中激活。当前削弱创新性展示的主要问题变成：资格化执行 harness 仍是 Poisson 专用工具，完整迭代/性能闭环和 CodeArts 过程证据尚未完成。

### 4.3 应用真实性

现有优势：

- 有真实 Ascend 910B4、CANN、AsNumpy wheel 和真机执行证据；
- 有原始 JSON、哈希和失败关闭记录；
- 没有把热扩散的小幅收益或 elementwise 的负收益包装成普遍加速。

现有弱点：

- Poisson 真解、配点和异常值均为人工生成；
- 暂无明确实际使用方或受益方；
- 暂无华为云上的可访问演示环境；
- 现有 Poisson benchmark 缺少比赛级真实性门禁。

建议在保留可复现解析 Poisson 基准的同时，增加一个来自实验室、课题组或公开可追溯工程问题的边界条件/网格/观测数据版本，并写明数据来源、许可证、使用方和实际收益。

## 5. CodeArts 状态与修改建议

当前已有：

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

这些路径和 Frontmatter 与当前官方项目级自定义智能体格式一致：

- <https://support.huaweicloud.com/usermanual-codeartsagent/codeartsagent_ug_0051.html>

但本机当前没有安装 `codearts` CLI，也没有发现真实 CodeArts 运行记录或保存后的 AgentTeam 模板。

当前官方 AgentTeam 支持 Task List、任务概览、调用上下文、任务产物以及 `/save-team`；保存的角色和场景位于 `.codeartsdoer/agent-team/roles` 与 `.codeartsdoer/agent-team/scene`：

- <https://support.huaweicloud.com/intl/zh-cn/usermanual-space/codeartsagent_space_0003.html>

后续应在真实比赛账号完成：

1. 导入 NumPilot 项目并验证四个项目智能体被正确识别；
2. 用 AgentTeam 执行 Poisson 迁移与算子补全任务；
3. 保存 Team、角色与 scene；
4. 截取 Task List、任务依赖图、失败重试和最终产物；
5. 展示 analyzer、operator engineer、evidence auditor 的职责隔离；
6. 展示 Skill/MCP 调用和人工确认；
7. 保存接受/拒绝代码 Diff 的记录；
8. 保存单元测试生成、失败诊断和修复过程；
9. 保存部署与真机运行日志。

当前 CodeArts CLI 的仓库索引文档只列出 Java、JavaScript、TypeScript 和 Go，不包含 Python：

- <https://support.huaweicloud.com/intl/en-us/usermanual-cli/codeartsagent_cli_0002.html>

因此不能虚构 Python Codebase 索引记录。应先向赛方确认；若该评分材料必须提供，可以增加一个薄的 TypeScript 或 Go MCP/证据控制层，并对该部分实际创建 Codebase 索引。

## 6. 必须修改的早期方案承诺

`NumPilot20260615-v1.pdf` 是早期愿景，不是当前可直接提交的最终 PPT。以下能力当前没有完整实现，必须标成“路线图”，不能写成“现有能力”：

- AOT/JIT 双模迁移；
- 通用数据流图；
- 真正的 CPU/NPU 子图划分；
- 成本模型自动卸载；
- 自动生成、构建、安装、注册、回滚 Ascend C 算子的完整闭环；
- Web 控制台；
- 完整 Python API；
- IDE 插件；
- 已经全流程使用 CodeArts 开发；
- 已经部署到华为云；
- 性能不退化保证。

最终 PPT 应明确区分：

1. 已实现并有证据；
2. 已实现但仍需真机/CodeArts 复验；
3. 研发原型；
4. 后续路线图。

## 7. 修改优先级

### P0：直接影响参赛资格与高分档

1. **冻结 Poisson 主线**  
   确定 Poisson 为主案例、热扩散为辅助证据、`cbrt` 为研发支线，不再同时扩张多个通用能力。

2. **实现 NumPilot 兼容的 Poisson 单步契约（已完成）**  
   `examples/poisson_ritz_step.py` 已提供 `build_inputs + forward`。

3. **实现领域算子规格与映射（已完成）**  
   两个 helper 已以 `operator.<name>` 身份进入 GapSpec、route 和 qualification。

4. **完成直接生成候选的自动修复、私有构建与算子真机段（已完成）**  
   编译/测试/运行诊断已回传用户 LLM；repair 按指定文件生成新候选，并绑定固定 CANNBot commit、CANN、SoC、诊断、请求/响应和父 repair 哈希。最终父链已封存成新 Artifact；两个算子的 Host/API、Ascend910B kernel、package、私有 OPP、public API 和真机数值对拍均已通过。package 只安装到运行目录的私有路径，未写系统 OPP。

5. **完成两个算子的六门资格化和 Poisson 单步激活（已完成）**  
   外部可信 harness 从封存 Artifact 重新构建，在新私有 OPP 中执行 artifact、build、UT、NumPy oracle、AsNumpy public API 和真实 NPU runtime 门，生成 schema v2 evidence 并原子发布 overlay。旧预计算梯度单步的 9 次调用已通过；当前完整动态一轮的 117 个调用点及 250 轮累计 29,250 次调用也已通过 runtime trace/profile/目标身份和数值门。

6. **消除当前 Poisson 路径的已知 Host fallback 与包装层深拷贝（已完成限定范围）**
   `outer`、`matmul` 已在哈希绑定的 v9 wheel 中走 ACLNN；完整动态一轮和 250 轮设备状态
   接力已通过。随后用哈希绑定的 move wheel 重新资格化并采集同一 10 轮 workload：
   1,170 条 kernel 保持一致，`aclrtMemcpy` 从 1,235 降到 35。当前证据仅覆盖声明的
   float32/rank profile，且不构成性能结论。

7. **重做真实性能测试**  
   使用独立 CPU/NPU 进程、固定线程、warmup、重复样本、中位数、原始样本、数值门禁、源码哈希、fallback 和 profiler。

8. **在 CodeArts 中完成并记录最后一轮核心开发**  
   不能只提交配置文件，必须有可验证的需求、任务、代码、测试、调试和部署过程。

9. **形成正式发布包**  
   把核心未提交文件纳入版本控制或明确发布清单；整理外部证据，不能把完整 venv 和临时目录直接塞入 ZIP。

10. **补齐 PPT、视频、测试与部署材料**  
    所有性能和功能结论必须能回溯到结构化证据。

### P1：显著影响创新性和真实性

- 增加真实工程数据与明确受益方；
- 增加 CodeArts MCP 控制层；
- 增加只读证据看板或轻量 Web 展示；
- 将缩小案例已经保存的 kernel timeline、传输字节和原始 profiler artifact 扩展到更大
  规模、通用采集器与重复样本；
- 增加第三方依赖清单、SBOM、许可证和来源说明；
- 增加 OS 级不受信代码隔离。

### 暂不优先

比赛提交前不建议同时实现：

- 通用控制流迁移；
- 任意 helper 内联；
- 完整动态图/子图切分；
- AOT/JIT 双模；
- 多种新算子生成；
- 大型 Web 平台。

这些方向会扩大风险，却不能替代 Poisson 主闭环、CodeArts 记录和提交材料。

## 8. 推荐执行顺序

### 阶段 A：主案例接线（已完成 P0 缩小案例）

- 已建立 `poisson_ritz_step` 与完整动态 `poisson_ritz_iteration` 规范和两个领域 OperatorSpec；
- 已增加分析、改写、报告与失败路径测试；
- shim 与 strict-NPU 均保持精确比较。

### 阶段 B：算子资格化

- 已用 17 文件、跨阶段一致性协议完成 fresh 直接生成；
- 已自动隔离构建并将 Host/Kernel/API/test 诊断回传模型修复；
- 已把最终 repair candidate 重签为 18 文件不可变 Artifact，并让 provenance 严格绑定 repair 链；
- 已在私有路径安装本地 OPP package，并构建固定 AsNumpy wheel；
- 已运行两个算子的 UT、NumPy oracle、public API 和真机边界用例，联合结果 `28 passed`；
- 已生成 qualification schema v2 evidence 并发布 capability overlay；
- 已显式激活两算子并运行完整动态一轮 strict-NPU。

### 阶段 C：真机正确性已完成，性能仍待完成

- 已修复并哈希绑定本例限定 profile 的 `outer`/`matmul` NPU 路径；
- 已以 250 轮设备状态接力移除逐轮显式 D2H；
- 已保存 117×250 次调用、profile、传输阶段和 CPU/NPU 收敛数值证据；
- 已保存缩小案例修复前后 profiler、原始数据库、复制审计与数值报告；
- 为更大规模保存预热和重复原始样本；
- 不满足门禁时明确显示 `performance_valid=false`。

### 阶段 D：CodeArts 与交付

- 在 CodeArts AgentTeam 中实际执行核心修改；
- 保存自定义智能体、Skill、任务图、问答、Diff、测试和部署记录；
- 更新最终 PPT；
- 录制完整演示视频；
- 生成代码 ZIP、依赖/部署/运行说明、测试报告和第三方声明。

## 9. 发布与合规清单

当前 NumPilot 仓库还应增加或整理：

- 顶层 `LICENSE`；
- `THIRD_PARTY_NOTICES.md`；
- AsNumpy、ops-nn、NumPy、Rich、pytest、CANN 模板等来源和许可证；
- AI 生成/修改代码的范围说明；
- 固定依赖与可复现环境清单；
- release manifest 与 SHA-256；
- 一键环境准备、doctor、shim、NPU 和 benchmark 命令；
- 结构化测试报告；
- 精简后的证据目录；
- 已审核通过且信息一致的报名表；
- 最终 PPT 和演示视频。

`/home/ma-user/work/numpilot-gap-runs` 约 112 MB，`/home/ma-user/work/numpilot-runs` 约 32 MB。两者不能原样作为项目源码提交，应选取：

- 关键 JSON/Markdown；
- profiler 原始制品；
- 必要 wheel/OPP 制品或下载与构建清单；
- SHA-256 manifest；
- 复现命令；
- 许可证和来源说明。

## 10. 当前可信边界

- NumPilot 离线回归只能证明代码路径未退化；Poisson 是否迁移以独立 strict-NPU 报告为准。
- 热扩散证据证明一个逐元素单步网络通过 strict-NPU，不证明通用科学网络迁移。
- Poisson 历史 `3.09×` 不能作为 strict-NPU 结论。
- 两个 Poisson 算子有真实 LLM 生成/repair 回执、封存 Artifact、正式六门 evidence 和显式 overlay 激活记录，可以称为“已资格化并在受约束 profile 内激活的本地能力”。这不等于所有 dtype/shape/signature 都受支持。
- 完整动态 Poisson 训练轮已经 strict-NPU 运行，117 个调用在 kernel/artifact trace 中精确匹配，最大绝对误差 `9.5367431640625e-07`。
- 同一产物的确定性缩小案例已运行 250 轮，29,250 次调用精确匹配，循环前 30 次 H2D、循环后 5 次 D2H，最终 CPU/NPU 最大绝对误差 `4.76837158203125e-07`；这仍不证明原始大配置的性能收益。
- `cbrt` 复合探针通过不等于 Ascend C custom OPP 已资格化。
- `.codeartsdoer` 文件存在不等于 CodeArts 已实际加载或全流程使用。
- 普通环境运行失败说明当前发布包尚不可由评委直接复现。
- 所有最终宣传必须能够回溯到代码、结构化报告、哈希、CodeArts 记录或 profiler 原始制品。

## 11. 下一项建议任务

下一轮应直接实施：

> 在缩小案例复制闭环已经通过的基础上，选择接近原始案例的规模，固定 CPU 线程与运行
> 身份，执行预热和重复原始样本，并同时守住数值、fallback、传输与 profiler 门，决定
> 能否发布性能结论。

这是当前距离比赛级真实性能闭环最近的修改方向；CodeArts 仍按用户决定后置。
