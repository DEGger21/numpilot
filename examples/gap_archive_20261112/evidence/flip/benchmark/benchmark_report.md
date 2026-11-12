# NumPilot 性能评测报告 · flip_contract

- 源程序:`flip/source/flip_contract.py`
- 源码 SHA256:`d89f11b95c02dc443cb12edbb853bf769856286eed87ac667a8d602bfe84d9d0`
- 迁移产物 SHA256:`69c3b83eccdeebd728fa40d6ca62dd79b2dc0bb699ecb4062f21280149c45d8f`
- 状态:**pass**
- AsNumpy 版本:`0.3.1`
- AsNumpy 路径:`flip/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
- 允许 fallback 逃生口:False
- 实际使用逃生口:False
- 迁移产物自身 build_inputs 已执行:True

## 执行来源门禁
- Audit 接口:NpuBackend.prepare_profile_guard
- Audit ID:unaudited-asnumpy-version
- 结论:严格 NPU ✓
- 迁移后算子:flip
- strict forward 静态数据流:通过 ✓
- 静态 ap.* callsite:flip×1
- 运行时 callsite 门禁:通过 ✓
  - kernel: flip×1
  - artifact: flip×1
- 模块路径:`flip/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
- CANN:9.2.0-beta.2
- SoC:Ascend910B3

## 数值门禁
- 输入生成一致性:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=1.000e-05, rtol=1.000e-04
- 同一规范输入上的 forward:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=1.000e-05, rtol=1.000e-04
- 迁移产物 build_inputs + forward:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=1.000e-05, rtol=1.000e-04

## 线程与计时证据
- CPU 基线 BLAS 线程:1 (来源=threadpoolctl)
- NPU 评测进程 BLAS 线程:64 (来源=threadpoolctl)
- CPU 时延:0.007 ms
- NPU 时延:1.756 ms
- CPU 原始样本:warmup=3, repeats=10, samples_ms=[0.010509975254535675, 0.008460134267807007, 0.007380731403827667, 0.006940215826034546, 0.006619840860366821, 0.007109716534614563, 0.006780959665775299, 0.007539987564086914, 0.007399357855319977, 0.007010065019130707]
- NPU 原始样本:warmup=3, repeats=10, samples_ms=[1.7692288383841515, 1.7597787082195282, 1.7573880031704903, 1.7561381682753563, 1.7451075837016106, 1.7461683601140976, 1.7580082640051842, 1.752127893269062, 1.7422987148165703, 1.745368354022503]
- NPU 计时口径:forward 输入预先驻留设备；每轮计时包含递归 D2H 回传同步
- 口径分类:kernel-only=False, end-to-end=False, 含 H2D=False, 每轮含 D2H 同步=True

## 性能结论
- **合格口径的 NPU/CPU 比值:0.0042×（CPU 快 237.94×）**
- 数值、provenance 与 CPU 基线一致性门禁均已通过。
