# ADR-004:裁掉缺口算子自动生成(仅标记并终止)

- 状态:已接受，2026-07-18 修订
- 日期:2026-07-02
- 相关:[quality-scenarios.md](../quality-scenarios.md) QA-5、`NumPilot20260615-v1.pdf`

## 背景(Context)

完整 NumPilot 愿景包含"**缺口算子的 Ascend C 自动生成**":当待迁移程序用到不在 AsNumpy 覆盖集的算子时,系统调用另一套 kernel 生成流水线(设计→编码→验证→优化)补齐该算子。这一环节复杂度极高(涉及 Ascend C / Triton 代码生成、编译、NPU 上数值验证与性能调优),远超"最简可运行 demo"与课程作业的边界。

## 决策(Decision)

**本 demo 明确不做缺口算子生成**。分析智能体基于覆盖集快照和当前已建模调用签名判定:

- 顶层同步 `forward` 的调用全部通过名称与签名门禁 → `migratable=True`，进入迁移。
- 覆盖集外 API、未验证 keyword/标量签名、Python ndarray dunder 或无法建模结构 → `unsupported` 携带原因；编排器立即返回，**不迁移、不验证、不伪造补齐结果**。

## 后果(Consequences)

**正向**
- 系统边界清晰、可讲清,聚焦迁移流水线主干(分析→决策→迁移→验证)。
- 缺口早停即一条健壮性质量场景(QA-5),负例 `np.fft.fft` 可确定性复现。
- 零外部依赖、离线可跑,契合课程与 CI。

**负向 / 权衡**
- 对含缺口算子的真实程序,demo 只能拒绝而非迁移——功能覆盖有限。
- 与完整系统的能力差距需在文档中显式说明,避免误解 demo 即全貌。

## 备选方案(Alternatives)

- **接入真实缺口算子生成**:功能完整但复杂度、依赖、不确定性远超 demo 目标;留待完整系统与后续参赛。
- **静默跳过缺口算子**:会产出不完整/错误的迁移代码,破坏正确性契约,不可接受。

## 升级路径(Future)

缺口算子生成可作为 Migration 阶段之外的**新智能体**接入。下一步应把简单字符串升级为 `GapSpec`，至少包含 qualified API、callsite、签名、shape/dtype/layout、目标 SoC/CANN、参考实现和验收规则；只有 `OperatorArtifact` 通过 build、UT、真机 ST、性能和版本化注册门禁后，才能回流 capability registry。
