# 混合程序的基础 API 资格

生成的缺口算子通过验证后，原程序通常还会调用 `add`、`mean` 等已有 API。
这些调用同样需要证据。基础 API 证书将固定 AsNumpy 源码、隔离构建、实际调用和
NumPy 对照绑定起来，使它们能与官方 adapter 一起用于原程序验证。

证书只覆盖已经验证的精确调用 profile。它不会把整个运行时改为 `matched`，
也不会让一个成功的 dtype/shape 推广到 API 的所有用法。

## 验证什么

1. 固定源码归档和 211 文件 manifest，与候选源码逐文件比对；只允许已证明的
   adapter overlay。继承 API 的实现及依赖必须保持一致。
2. 构建事务、wheel、运行时身份，以及实际加载的全部 AsNumpy Python/动态库文件。
3. 从未修改的原始 benchmark `forward` 捕获输入和调用 metadata；保存可重放的
   `.npy` 输入、NumPy 期望值、设备实际值和准确 profile。
4. 独立 NumPy 重算、shape/dtype、NaN/Inf、数值容差及零的符号检查。非创建类 API
   还验证每种实际 profile 的特殊值和正负零。
5. 每个调用的原生 C++ 入口计数、禁止 fallback 的记录，以及独立进程产生的
   `msprof` 设备 kernel CSV。按 API 绑定 worker，避免用其他算子的 trace 充数。

单个 API 的任一 probe 失败，该 API 整体不发布；失败日志继续保留。
证书是当前软件/硬件组合下的验证记录，不是任意输入值的数学证明。

## 当前接口范围

首批 producer 验证 `absolute`、`add`、`divide`、`full`、`maximum`、`mean`、
`minimum`、`multiply`、`not_equal`、`square` 和 `tanh`。实际发布清单以
`production-receipt.json` 为准，不能仅凭此列表宣称某个 API 已通过。

- tensor 为 float32、C 连续、正维度、rank 1–3，且 shape/参数签名与证书完全匹配。
- `mean` 仅接受明确的 Python 整数 axis 和可选布尔 keepdims；不支持本证书之外的
  `axis=None`、tuple axis、dtype 或 out。当前 `axis=None` 实现有主机标量路径。
- `full` 使用 shape、fill 两个位置参数和显式 float32 dtype。profile 记录实际创建
  输出及填充值，不伪造输入 tensor。
- `not_equal` 此批只验证数组与数组的比较及布尔输出。

`copysign`、`full_like`、`power` 和 `subtract` 未在本批范围内。
特别是当前 `copysign` 对第二个参数为零的情况存在语义缺陷，不能借数值容差放行。
原始 cbrt lowering 的限制仍需独立修复或替换。

## 产生证书

在已配置 CANN 和单张可见 NPU 的环境中，使用构建 adapter 时的 Python。
`ADAPTER_RUN` 含已有 `integration-v2` 和 `qualification-v1`；`BASE_MANIFEST`
指向该官方资格所匹配的原始能力 manifest；`BASELINE_RUN` 必须是新目录。
producer 不调用模型、不修改共享 AsNumpy 安装。

```bash
PYTHONPATH="$PWD" python tools/qualify_baseline_apis.py \
  --integration-dir "$ADAPTER_RUN/integration-v2" \
  --official-evidence-root "$ADAPTER_RUN/qualification-v1" \
  --official-base-manifest "$BASE_MANIFEST" \
  --out "$BASELINE_RUN"
```

首次发布的原始样例是 G001、D1005 和 I003。producer 保存源码 SHA，并只捕获
`forward` 内的计算；CPU `build_inputs` 不会被算作设备计算。扩展到别的样例时，
应增加真实 profile 和 probes，重新产生完整证书，不能手改既有证书扩大范围。

## 与官方资格共同使用

先按[范围限定 benchmark](scoped-benchmark.zh-CN.md)选择候选 runtime。
同时传入以下两个参数，缺一会报错：

```bash
python -S -m numpilot.benchmark_suite \
  --root benchmark/cases --backend npu \
  --capability-overlay "$ADAPTER_RUN/qualification-v1/capability-overlay.json" \
  --qualification-evidence-root "$ADAPTER_RUN/qualification-v1" \
  --baseline-api-certificate "$BASELINE_RUN/baseline-api-certificate.json" \
  --baseline-api-evidence-root "$BASELINE_RUN" \
  --cases G_partial_dedup/G001_concat_three_views \
          D1_partial_low_kernel/D1005_feature_join \
          I_gap_model/I003_nan_classification \
  --out "$BATCH_RUN/mixed" --json

python tools/check_operator_artifacts.py \
  --run-root "$BATCH_RUN/mixed" --expected-total 3 --json
```

加载顺序固定：官方 overlay 先对原始 manifest 验链，再组合基础 API 证书。
原始官方 manifest、overlay 和全局 runtime 身份均保持不变。
混合样例保存 v2 scope，绑定两个资格来源、原始/分析/生成源码、实际 profile、
guard 和程序数值对拍。旧 v1 scope 仍可验证。

性能证据是独立结论。不同基础证书不能复用同一性能结果；通过数值和 scope 验证
不代表加速，更不代表 150 个 benchmark 已全部迁移。

## 交接与恢复

保存整个 `BASELINE_RUN`、benchmark run、官方资格目录和 adapter 构建目录。
私有实验放在开发日志目录；仓库只保存实现、测试和可复现说明。不得把密钥或
`llm.json` 凭据放入公开测试 fixture。跨机器恢复需要重定位证据根目录并重验全部
hash；不能改写已经绑定的回执。完整运行结果与局限记录在本阶段开发日志中。


## Stage82 实测结果（2026-11-10）

11个基础API、46个真实设备probes通过；G001/D1005/I003原始混合程序正式迁移通过。
完整35例single-gap复测为4迁移成功、28缺口停止、3执行前runtime资格拒绝；后3例均涉及cbrt组合。
全局audit_required、原始150例及作者源码不变，没有新增性能或加速宣称。
本次复用官方concat和既有基础API，自定义Ascend C/ASC生成路线仍需单独完成验收。
