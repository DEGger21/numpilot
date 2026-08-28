# 比赛要求与 NumPilot 产品验收映射（2026-08-27）

本文把赛题文件
`XH-202618_基于华为云码道（CodeArts）代码智能体解决复杂软件工程问题.pdf`
转换为 NumPilot 的工程验收项。页码按该 PDF 的 13 页版本记录。

本阶段按用户决定先完善 NumPilot 迁移框架，不实现 CodeArts 接线。但 CodeArts 只能后置，
不能从最终参赛范围删除：赛题在第 5～6 页把完整 CodeArts 开发流程列入完整性最高档条件，
未基于 CodeArts 的作品存在落入完整性低档的明确风险。

## 1. 冻结的产品场景

NumPilot 解决的具体问题是：将可审计的普通 NumPy 科学计算脚本迁移到
AsNumPy/Ascend NPU；当目标运行时存在 API、签名或执行来源缺口时，优先复用已安装 CANN
和固定版本华为官方算子源码，再进入模型生成候选，并以失败关闭的资格化证据控制激活。

当前支持边界必须如实表述：

- 默认入口是单文件 ordinary script，而不是任意 Python 项目；
- 静态分析和迁移不执行用户源码；执行必须由单独的显式验证命令发起；
- `DIRECT`、`FALLBACK_CONSENT`、`GAP_REQUIRED`、
  `CAPABILITY_AUDIT_REQUIRED`、`BLOCKED` 是不同结果，不能折叠成“迁移成功”；
- 已生成 kernel/adapter candidate 不等于已构建、已资格化或已激活；
- 数值一致不自动证明 native NPU provenance，也不产生性能结论；
- 一个限定 dtype/shape/runtime 的 MatMul 成功案例不能外推为通用 `numpy.matmul` 支持。

## 2. 评分要求到验收项

| PDF 页码 | 赛题要求摘要 | NumPilot 可执行验收 |
|---|---|---|
| 1～3 | 明确复杂工程场景、边界和核心痛点 | 固定 NumPy→AsNumPy/Ascend 场景；记录支持面、阻断面、人工流程和跨 Python/AsNumPy/ACLNN/ops-nn/CANN 的工程链 |
| 3～4 | 完整技术方案和可运行产品 | 干净环境完成安装、`doctor`、`analyze`、`migrate`、gap 恢复、显式数值验证和结构化报告；失败返回非零状态并可从证据检查点恢复 |
| 3～4 | 背景、需求、架构、开发调试、挑战、测试、效果和扩展性材料 | 所有 PPT 结论绑定源码 revision、机器可读报告和 SHA-256；draw.io 图只画已实现能力，目标能力使用不同视觉状态 |
| 4 | 演示视频或可访问环境；在线环境须在华为云 | 冻结一条成功演示和一条失败关闭演示；若提供 URL/账号，做华为云外部访问验收，不以本机截图冒充云部署 |
| 4 | 全源码、依赖、部署和运行说明，可复现 | 从干净 checkout 构建 wheel/sdist/最终 ZIP；第二环境只按文档复现；固定 Python、CANN、SoC、AsNumPy 和 ops-nn 身份 |
| 4 | 第三方模型、组件、代码需注明来源和许可证 | 维护第三方声明和 SBOM；区分 NumPilot 自研、模型生成、AsNumPy patch、官方源码复用，保存 origin/commit/tree/file hashes |
| 4～6 | 完整性 50 分 | 产品主线必须形成真实闭环；不能用案例专用外部 harness 替代 ordinary 产品功能，也不能把独立原语描述成自动编排 |
| 6～7 | 创新性 25 分 | 展示 official-first、模型补缺、六门资格化、显式激活和证据链；量化迁移时间、人工步骤、支持 API 和失败原因 |
| 7～8 | 真实性 25 分 | 至少一个固定来源、许可证、数据哈希的真实工程负载；明确实际受益方；保留真实 Ascend/华为云环境身份和原始证据 |
| 8～10 | 2026-09-05 前提交并接受代码核查 | 功能冻结后统一 commit/tag；PPT、视频、源码包、运行证据和报名材料绑定同一不可变版本并做回读验收 |

评分权重为完整性 50、创新性 25、真实性 25。当前继续增加零散 API 的收益低于补齐产品闭环、
真实案例和发布复现。

## 3. 当前能力与证据边界

截至本文日期，框架已有：

- runtime/hash-bound capability 与五类确定性判定；
- ordinary `analyze` 和 configured-model `migrate`；
- ordinary `verify-script` 的 hash-bound 双进程基础数值验证；回执记录实际解释器和
  NumPy/AsNumPy 模块身份，同时固定不宣称 strict-NPU、性能或 OS sandbox；
- ordinary `benchmark-script` 的受限 strict-NPU resident 性能门：真实 configured-model
  strict artifact、分段 AST、独立单核 worker、CPU BLAS 实测、runtime/profile/call-count、同步、
  最终 D2H 与数值证据进入同一 write-once receipt；不外推为冷启动 E2E 或任意 Python benchmark；
- source/target/runtime/policy 绑定的 schema-2 typed-receipt/write-once hash-chain，以及三类
  候选来源的 route/license/provenance/content 合同；`recover-gaps ingest-official` 已消费
  migration/GapSpec/official-resolution 收据并推进到 `RESOLVED`，但不执行候选后的恢复动作；
- installed ACLNN → pinned ops-nn 的 official-first 只读 resolver 与收据门禁；
- GapSpec、draft/repair/seal、AsNumPy adapter、qualification schema 和 capability overlay 原语；
- Ascend 910B4 上限定二维非空 `float32` MatMul 的官方复用、AsNumPy 接入和封死宿主回退的
  真机数值纵切；
- 可编辑三页整体架构图和三页官方算子恢复图。

ordinary benchmark P0 基础切片已关闭；这些能力仍有三个主要 P0 产品缺口：

1. 消费既有收据并驱动 `integrate → qualify → activate → rescan → final migration → verify`
   的执行编排；可恢复状态链基础已经实现；
2. 固定版本、真实数据、明确受益方的多模块科学工程负载；
3. 干净发布、第二环境复现、SBOM、华为云演示和最终材料一致性。

MatMul 小例和现有 Poisson/热扩散证据可用于证明方法与真实性，但不能单独代表任意 NumPy
项目或通用算子覆盖。性能结论必须继续与数值、runtime identity、执行来源、计时边界和原始样本
绑定。

## 4. 当前 P0 顺序

1. 已完成独立 `verify-script` 与 `benchmark-script` 基础切片：前者显式执行原始与迁移脚本，绑定
   source/report/artifact hash，通过无 pickle、有界 named-output 合同比较 shape、dtype、
   NaN/Inf 和数值；后者仅对 strict GLM artifact 的 marker resident 区增加
   runtime/profile/call-count/deny-fallback、单核/线程/同步与性能证明。下一步增加直接采样的
   cold/E2E、多核 CPU 对照和 profiler 补充证据，而不是扩大当前 headline。
2. 已完成 gap 状态控制面第二切片：每次 phase `+1` 强制新增 parent/binding/coverage/kind
   精确绑定的 typed receipt；migration/GapSpec/official-resolution 已接入并可幂等续接到
   `RESOLVED`。下一步从三类 candidate materialization 开始，接 adapter/integration/
   qualification/activation/rescan/final verification，并用 `verify-script` 作为最终基础
   correctness 门。
3. 选取一个真实工程负载并明确受益方，固定代码、数据、许可证和环境；保留合成小例作为快速回归，
   不把它当成唯一真实性案例。
4. 增加直接采样 cold/E2E、默认 CPU 配额对照、规模/复用交叉点和 exact-artifact profiler，
   让 resident 优势区间可解释而非只展示单点峰值。
5. 功能冻结后才整理提交、构建发布物、做第二环境复现并录制演示；CodeArts 接线随后按赛题最高档
   标准补回。

按当前产品决策，MatMul linkage-v2 保持历史恢复链检查点，不再占用比赛主线；除非上述真实性、
发布与 CodeArts P0 已关闭，否则不继续请求模型或推进该 adapter。

## 5. 发布阻断条件

出现以下任一情况，不得把当前工作树标记为参赛 RC：

- 全量测试、`git diff --check`、secret/license 检查或证据 manifest 未通过；
- Git 作者身份未确认、远端不可认证、未形成不可变 commit/tag；
- tracked、staged、unstaged 和 untracked 的案例/patch/证据没有按提交边界完整纳入；
- draw.io XML 未通过结构校验，或图中把目标能力画成已实现；
- wheel/sdist/ZIP 未在第二个干净环境复现；
- ordinary 数值验证、gap 自动恢复、真实工程案例或最终 CodeArts 证据仍被外部手工步骤替代却未明确披露。
