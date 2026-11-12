# NumPilot 性能评测报告 · pad_contract

- 源程序:`pad/source/pad_contract.py`
- 源码 SHA256:`d14c80a52866f48e3188b71dfe039df27645432e420e1b2ccc02c2494ed5d956`
- 迁移产物 SHA256:`f0bd65e1862a09fe7187254e723090800ddfd25557127fb020309c089a22f130`
- 状态:**pass**
- AsNumpy 版本:`0.3.1`
- AsNumpy 路径:`pad/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
- 允许 fallback 逃生口:False
- 实际使用逃生口:False
- 迁移产物自身 build_inputs 已执行:True

## 执行来源门禁
- Audit 接口:NpuBackend.prepare_profile_guard
- Audit ID:unaudited-asnumpy-version
- 结论:严格 NPU ✓
- 迁移后算子:pad
- strict forward 静态数据流:通过 ✓
- 静态 ap.* callsite:pad×1
- 运行时 callsite 门禁:通过 ✓
  - kernel: pad×1
  - artifact: pad×1
- 模块路径:`pad/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
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
- CPU 时延:4.117 ms
- NPU 时延:1.724 ms
- CPU 原始样本:warmup=3, repeats=10, samples_ms=[5.555015057325363, 4.14911936968565, 4.1123684495687485, 4.330730065703392, 4.128898493945599, 4.104078747332096, 4.107308574020863, 4.076899029314518, 4.079057835042477, 4.11652959883213]
- NPU 原始样本:warmup=3, repeats=10, samples_ms=[1.7243782058358192, 1.7377976328134537, 1.72449741512537, 1.726428046822548, 1.718968152999878, 1.7266683280467987, 1.7223982140421867, 1.7108581960201263, 1.7177481204271317, 1.7159786075353622]
- NPU 计时口径:forward 输入预先驻留设备；每轮计时包含递归 D2H 回传同步
- 口径分类:kernel-only=False, end-to-end=False, 含 H2D=False, 每轮含 D2H 同步=True

## 性能结论
- **合格口径的 NPU/CPU 比值:2.3873×（NPU 快 2.39×）**
- 数值、provenance 与 CPU 基线一致性门禁均已通过。
