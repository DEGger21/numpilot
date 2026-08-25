# Demo v1 Case 01：普通脚本迁移与 NPU 性能

## 结论

在同一段 `2048 × 2048`、`float32` 脚本计算范围内，**NPU 稳态中位数为
4.499 ms，CPU 单线程中位数为 809.784 ms，NPU 加速 180.004×**。

| 范围 | CPU | NPU | 胜者 |
| --- | ---: | ---: | --- |
| 稳定脚本体 | 809.784 ms | 4.499 ms | NPU，约 180.00× |
| 冷进程端到端 | 1,140 ms | 5,056 ms | CPU，约 4.44× |

冷进程范围包含 Python 启动、模块导入、CANN 与设备初始化，只描述启动成本，不参与稳定
计算加速比。无歧义的机器可读结论见
[`performance/demo-performance-summary.json`](performance/demo-performance-summary.json)。

## 测试对象

- [`source.py`](source.py) 是 GLM 生成的 Case 01 大规模性能变体，只导入 NumPy；
- [`migration/migrated.py`](migration/migrated.py) 是 NumPilot 调用同一 `glm-5.3` 生成并经
  精确 AST 门禁发布的 AsNumpy 代码；
- 工作负载为 `full × 2 → matmul → exp → sum → divide → sum × 2`；
- 源码 SHA-256：`eb07e1aa682d882f83c59f02b85c52a2ecc8a5054b9d856ae82760326f86fe2b`；
- 迁移产物 SHA-256：`f58979c0a16a0ba7414621337cb998f122c7363d5f361190c5fd665bf895dd91`。

源码生成和迁移两次请求均记录 `requested_model=glm-5.3`、
`response_model=glm-5.3`、HTTP 200、TLS 校验启用；调用时通过用户指定的
`http://127.0.0.1:7897` HTTP/HTTPS 代理。仓库不包含 Token、Base URL 或完整私有配置。

## 公平口径与正确性

- CPU 使用 OpenBLAS 1 线程；CPU/NPU worker 都固定在逻辑 CPU 168；
- 预热 3 次，测量 10 次；计时包含数组分配和最终 `.to_numpy()` 输出边界，不包含终端 I/O；
- 数值比较覆盖 20,973,569 个元素，在 `atol=1e-5, rtol=1e-4` 下通过；
- 最大绝对误差为 `0.000244140625`，最大相对误差为 `5.602423698292114e-06`。

## NPU 真实性边界

- 设备为 NPU 7 / Ascend 910B4，采样时健康状态为 `OK`；
- AsNumpy wheel SHA-256 为
  `afe845c45d17e7f5cd619831f555fba3c2344296ffe02f1d251017b04cf0282d`；
- 严格 NPU runtime guard、版本绑定 capability profile、目标模块完整性和逐算子调用次数均通过；
- ACLNN 日志确认 Matmul、Exp、Div 与 ReduceSum 调用；
- 本轮没有采集 kernel profiler，因此不能表述为“profiler 已验证”。当前证据是运行时
  target/profile/call-count guard 加 ACLNN 日志。

## 精选证据边界

本目录是适合 Git 发布的精选证据，不是完整运行树。两个各约 81 MiB 的 CPU/NPU 数值数组
没有提交；它们的 SHA-256 和比较结果保留在 JSON 报告中。完整 write-once 运行树位于仓库外，
发布材料不得把这里的精选副本描述成可独立复算全部数值的归档。

关键文件：

- [`generation/receipt.json`](generation/receipt.json)：GLM 测试源码生成回执；
- [`generation/ast-gate.json`](generation/ast-gate.json)：源码规模、dtype、API 和来源门禁；
- [`migration/script-migration-report.json`](migration/script-migration-report.json)：迁移分类、
  GLM 回执与源码/产物绑定；
- [`performance/cpu-benchmark.json`](performance/cpu-benchmark.json)：CPU 原始采样；
- [`performance/npu-benchmark.json`](performance/npu-benchmark.json)：NPU profile、调用跟踪与采样；
- [`performance/npu-standalone-aclnn.log`](performance/npu-standalone-aclnn.log) 与
  [`performance/npu-benchmark-aclnn.log`](performance/npu-benchmark-aclnn.log)：实际 ACLNN 调用日志；
- [`performance/performance-comparison.json`](performance/performance-comparison.json)：数值与稳态比较；
- [`performance/cold-start-comparison.json`](performance/cold-start-comparison.json)：独立冷启动口径；
- [`MANIFEST.sha256`](MANIFEST.sha256)：本目录全部发布文件的完整性清单。

原始比较文件中的 `cpu_over_npu_ratio=180.004...` 是 CPU 时间除以 NPU 时间，即 NPU
加速 180.004×；它不是“CPU 更快”。Demo 汇总因此显式使用 `winner=NPU` 和
`npu_speedup=180.004...`。
