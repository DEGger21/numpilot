# 任务：NumPilot 缺口算子补全

## 目标
按照 NumPilot20260615-v1.pdf 的功能设想，为 NumPilot 增加可验证的缺口算子补全链路；以 numpyexample 为 AsNumpy 自定义算子接入模板，并遵循 cannbot-skills 的 Ascend C registry-invoke 规范。

## 待办事项
- [x] 提取 PDF 的目标流程与验收边界
- [x] 审计 NumPilot 当前缺口检测、Agent 编排和产物模型
- [x] 审计 numpyexample 的 Ascend C → ACLNN → AsNumpy 接入链路
- [x] 锁定最小可运行架构与安全边界
- [x] 实现缺口算子补全的未验证草稿纵切及 CLI/报告接入
- [x] 补充单元、集成和失败路径测试
- [x] 更新使用文档并完成环境内验证

## 进度
7/7（本轮未验证草稿纵切）

## PDF 完整闭环的下一阶段

- [x] 离线 qualification evidence 准入契约、严格绑定与失败关闭校验
- [x] 原子 Capability overlay、CAS/revision/parent hash 与发布后状态报告
- [x] 可信 build/UT/NumPy oracle/AsNumpy public API/NPU runtime 实际证据
- [x] 私有 OPP 与 AsNumpy public API 接入，不修改系统安装
- [x] 将通过完整证据复验的 overlay 接入 profile-aware runtime capability 解析
- [x] qualified + activated 后完成迁移、完整动态一轮和 250 轮数值回归
- [x] Poisson 缩小案例的 profiler、已知方向探针、包装层复制定位与消除后重采
- [ ] 放大规模、预热与重复性能样本通过后再发布性能结论

## 当前证据边界

- Poisson 的 `apply_gradient_descent` 与 `smooth_l1_loss` 已通过 move wheel 对应的 v6 六门资格化和显式
  overlay 激活；这个结论只覆盖证据中声明的 profile，不代表任意输入都合格。
- 修复后同一 10 轮 workload 保持 1,170 条 kernel，`aclrtMemcpy` 从 1,235 降为 runner
  边界传输对应的 35 次；这是复制闭环证据，不是耗时或加速比结论。
- 2026-08-13 已在隔离 AsNumpy wheel 上完成真实 910B4 ACLNN **复合** `cbrt`
  探针：200,015 个 float32 NumPy oracle case、特殊值/signbit 与边界长度通过；这不是
  Ascend C custom OPP，也不是 qualification 六门报告，不能把 `cbrt` 提升为
  `CUSTOM_NPU` 或激活 overlay。
- `numpy.cbrt` 因此仍为 `generated_unverified`；生成的 custom-op scaffold 仍是错误架构的
  TODO kernel，未构建、未安装、未在 NPU 执行。
- 2026-08-12 最终 NumPilot 回执代码发出的全新 nonce 请求收到 HTTP 200，请求/响应模型
  均为 `glm-5.2`，且响应包含 nonce；但配置
  endpoint 是私有网关，缺少网关服务端日志或签名 attestation，客户端回执不能独立证明
  官方 GLM 后端身份。
- 同日完整算子草稿生成确实走了远程路径：早期 3 次请求失败；最终代码再做 1 次新目录
  尝试并明确收到 HTTP 504。两次都失败关闭，没有生成带当前 transport receipt 的新草稿。
