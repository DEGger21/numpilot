# NumPilot 质量属性场景

以下场景只描述当前可测行为。

## QA-1 可修改性：更换迁移引擎

| 要素 | 内容 |
|---|---|
| 刺激 | 新增 AST/LLM 之外的迁移策略 |
| 响应 | 实现 `MigrationEngine` 并注册；继续复用统一静态、三层语义与 provenance 门禁 |
| 度量 | 既有引擎测试零回归；显式选择的新引擎失败时不静默换成另一引擎 |

## QA-2 可移植性：shim 与真实 NPU

| 要素 | 内容 |
|---|---|
| 刺激 | 同一产物需要在离线 CI 与 Ascend NPU 上验证 |
| 响应 | `ExecutionBackend` 切换 shim/NPU；真实后端额外提供版本化 capability manifest |
| 度量 | shim 单测无需 NPU；shim 永不发布性能；NPU 版本或来源未知时 fail closed |

## QA-3 正确性：防止迁移假阳性

| 要素 | 内容 |
|---|---|
| 刺激 | 错误 `forward` 试图通过修改 `build_inputs` 补偿输出 |
| 响应 | 分别验证输入生成、同一规范输入 forward、产物自身 build_inputs + forward |
| 度量 | 任一层结构、shape、dtype、数值、NaN/Inf 或异常不一致即 `status=fail`；回归测试包含补偿性输入反例 |

## QA-4 可审计性：性能数字来源

| 要素 | 内容 |
|---|---|
| 刺激 | 用户请求 NPU/CPU speedup |
| 响应 | 独立 CPU/NPU 进程记录源码 hash、参考输出、实际 BLAS 线程、wheel/CANN/SoC、逐算子 provenance 和计时范围 |
| 度量 | 只有 `performance_valid=true` 才存在 speedup；fallback/unknown/基线漂移时 speedup 为 `null` |

## QA-5 健壮性：缺口与危险签名早停

| 要素 | 内容 |
|---|---|
| 刺激 | `np.fft.fft`、未验证 ufunc keyword、Python ndarray dunder 或风险标量二元调用 |
| 响应 | 分析阶段生成带 callsite/reason 的 unsupported 结果，不产出误导性迁移文件 |
| 度量 | CLI 非零退出；报告给出最小缺口原因；负例有单测覆盖 |

## QA-6 并发与隔离

| 要素 | 内容 |
|---|---|
| 刺激 | 多智能体并行验证不同 backend/产物 |
| 响应 | 当前进程内模块注入由全局锁串行化并恢复；正式版进入独立受限子进程 |
| 度量 | 并发回归不串用 `asnumpy` 模块且退出后恢复原模块；OS 级文件/网络/设备权限隔离仍标记为未完成 P0 |
