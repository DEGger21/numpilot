# 变更记录

## Unreleased

- 将默认报告、benchmark、向导和缺口补全产物移到仓库外的 XDG 状态目录；
- 新增 `numpilot paths` 与 `NUMPILOT_STATE_HOME`、`NUMPILOT_RUNS_DIR`、
  `NUMPILOT_REPORTS_DIR` 覆盖；
- 新增不执行案例源码的 `numpilot check-suite` 静态准入门，区分直接迁移、单个简单缺口和
  阻断案例，并可严格核对 5+5 测试集构成；
- 静态准入门绑定真实 AsNumpy Python/C++ 签名与返回语义，检查创建器 dtype、axis、
  scalar operand、einsum 形式、只读 ndarray metadata 和 `build_inputs` 参数类型；
- 新增模型生成/修复 AsNumpy adapter 的独立命令及内置 custom-op integration Skill；
  adapter、operator repair 与 seal/reload 均绑定完整 Skill、诊断、父链和文件身份；
- 未实现的 `--operator-goal performance` 现在会在联网、读取 Skill 或写输出前明确拒绝；
- 整理可公开的 heat-diffusion 与 Poisson strict-NPU 精选证据；
- 增加 CANN memcpy 方向探针、Poisson profiler 复制审计和哈希绑定的 AsNumpy move wheel
  capability；同一 10 轮 workload 的 `aclrtMemcpy` 从 1,235 降至 35；
- 增加 Poisson 原始 48×48/49 RBF/3000 轮独立 CPU/NPU 性能 runner、OpenBLAS 线程实测、
  同规模 profiler 发布门与精选证据；合格结果为 CPU 约快 `2.67×`，不宣称 NPU 加速；
- 增加 Poisson 广播/CSE 优化变体和哈希独立的性能 runner；每轮调用从 117 降到 101，
  原始规模 NPU 中位时延下降 `19.71%`，3000 轮最终输出与基线字节级相同；
- 增加 `run_poisson_demo.sh` 客户体验入口：先校验 135 份公开证据，再展示固定
  CANNBot Skill、算子 qualification、实时 strict-NPU 迁移与驻留执行；
- 修复新版 `npu-smi` 仅显示 `910B4` 而无 `Ascend` 字样时的 doctor 设备误报；
- 增加 Poisson 单核 CPU 亲和性实验 runner：CPU 子进程在 Python 启动前固定到一个逻辑核，
  收据同时验证 affinity 与 BLAS 单线程；在缺少同身份 passing profiler audit 时明确保持
  `performance.published=false`，不发布或引用实验性能比值；
- 增加开源许可证、贡献、安全、第三方依赖和证据存储说明。

## 0.1.0

- NumPy → AsNumpy AST/LLM 迁移原型；
- shim 与真实 NPU 后端、capability provenance、严格数值比较和 benchmark 门禁；
- 缺口算子 GapSpec、LLM 草稿、repair、封存、qualification overlay 与 Poisson P0 纵切。
