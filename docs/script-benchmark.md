# Ordinary script strict-NPU resident benchmark

`benchmark-script` 是 ordinary 主流程的独立性能门。它消费已经由 configured model 发布的
`script-migration-report.json` 与同目录 `migrated.py`，不会再次迁移、调用模型或修改产物。

```bash
python -m numpilot benchmark-script SOURCE.py \
  --migration-report RUN/script-migration-report.json \
  --output result \
  --out RUN/script-benchmark-report.json
```

`bench-script` 是同一命令的短别名。`--output` 可重复；每个名字都必须在源码和迁移脚本的
全局命名空间中解析为 numeric/bool scalar 或 array。

## 首版作用域

源码和迁移产物必须各自恰好包含一次：

```python
# NUMPILOT_DEVICE_INPUTS_END
```

marker 前是只执行一次的输入准备，marker 后是被重复计时的设备驻留直线计算区。marker 只是
性能合同，不是 ordinary 迁移语法要求。计时区只允许简单赋值及 NumPy/AsNumpy namespace
调用；import、函数、控制流、helper/ndarray method、in-place、`out=`、`array/asarray`、
`from_numpy/to_numpy/item` 都会在执行前失败关闭。源码与迁移产物的 prefix 和 suffix 分别做
namespace-normalized AST 等价比较，模型不能借移动 marker 改变口径。

首版 headline 是 steady device-resident compute，不是 kernel-only profiler，也不是冷启动或
任意 Python 端到端 benchmark。回执另外记录：

- prefix/input preparation；
- 每次 resident 原始样本；
- 每个最终输出唯一一次 D2H；
- 完整 worker 进程 wall time；
- `prefix + resident median + final output transfer` 的暖运行单次组件和。

组件和明确标记 `directly_sampled_end_to_end=false`，不能冒充直接采样的端到端时延。

## 发布门

只有以下事实全部为真时才发布 `median_speedup`：

- report 是 schema-v2 ordinary `migrated_strict_npu`；
- configured-model 真实远程调用、HTTP 200、TLS 与精确模型身份回执通过；
- source/report/migrated SHA-256 和 report-sibling 路径闭合；
- qualified decision 为 `DIRECT`，全部调用点可用于性能声明；
- source/migrated 分段 AST 等价，timed suffix 无 Host escape；
- CPU 与 NPU 是不同 PID，且在用户代码/NumPy 导入前固定到同一逻辑核；
- 四项线程环境固定为 1，CPU BLAS runtime 实测为 1 线程；
- 当前 NPU runtime identity 在执行前后稳定并与迁移报告匹配；
- runtime profile/target guard 通过，实际 `ap.*` 调用次数与 AST 期望完全相等；
- prefix 完成后同步一次，每次 NPU warmup/sample 前后各同步一次，次数精确；
- 所有显式命名输出的结构、shape、dtype、NaN/Inf 和数值容差一致。

任一门失败时命令返回 1，发布 write-once 失败回执，`performance_valid=false` 且 speedup 为
`null`。输入/report/marker/路径等前置准入失败返回 2，通常不发布回执。成功返回 0。

## 协议与统计

默认使用当前 affinity 的第一个逻辑核、3 次预热和 20 次有序样本；最少允许 7 个样本。
可显式设置：

```bash
python -m numpilot benchmark-script SOURCE.py \
  --migration-report RUN/script-migration-report.json \
  --output result --out RUN/benchmark.json \
  --core 0 --warmup 3 --samples 20 --timeout 600
```

回执保留 CPU/NPU 全部纳秒样本、中位数、p10、p90、median speedup、保守
`CPU p10 / NPU p90`，以及以 source hash 为确定性随机种子的 median speedup bootstrap 95%
区间。CPU 单核结果不是默认多核 NumPy 结论；公开材料必须同时说明 CPU 配额、数据规模、dtype、
marker 边界和传输口径。

## 执行与安全边界

两个 worker 使用显式 Python、隔离进程组、私有 HOME、移除模型 Token 等敏感环境变量，并以
bounded JSON + `npy(allow_pickle=False)` 传递证据。它们是可终止的进程隔离，不是 OS sandbox；
运行用户源码仍需显式授权。最终 JSON 为 `0600` write-once 文件，并绑定 NumPilot runner/worker
SHA-256、版本、可用时的 Git commit 和 dirty 状态。

