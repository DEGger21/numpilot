# 变更记录

## Unreleased

- 将默认报告、benchmark、向导和缺口补全产物移到仓库外的 XDG 状态目录；
- 新增 `numpilot paths` 与 `NUMPILOT_STATE_HOME`、`NUMPILOT_RUNS_DIR`、
  `NUMPILOT_REPORTS_DIR` 覆盖；
- 整理可公开的 heat-diffusion 与 Poisson strict-NPU 精选证据；
- 增加 CANN memcpy 方向探针、Poisson profiler 复制审计和哈希绑定的 AsNumpy move wheel
  capability；同一 10 轮 workload 的 `aclrtMemcpy` 从 1,235 降至 35；
- 增加 Poisson 原始 48×48/49 RBF/3000 轮独立 CPU/NPU 性能 runner、OpenBLAS 线程实测、
  同规模 profiler 发布门与精选证据；合格结果为 CPU 约快 `2.67×`，不宣称 NPU 加速；
- 增加 Poisson 广播/CSE 优化变体和哈希独立的性能 runner；每轮调用从 117 降到 101，
  原始规模 NPU 中位时延下降 `19.71%`，3000 轮最终输出与基线字节级相同；
- 增加开源许可证、贡献、安全、第三方依赖和证据存储说明。

## 0.1.0

- NumPy → AsNumpy AST/LLM 迁移原型；
- shim 与真实 NPU 后端、capability provenance、严格数值比较和 benchmark 门禁；
- 缺口算子 GapSpec、LLM 草稿、repair、封存、qualification overlay 与 Poisson P0 纵切。
