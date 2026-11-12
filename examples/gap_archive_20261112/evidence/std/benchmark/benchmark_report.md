# NumPilot 性能评测报告 · std_contract

- 源程序:`std/source/std_contract.py`
- 源码 SHA256:`b3526fd459981a66b39f98a77be9964fcd9372f332849b4872c080131c1baf94`
- 迁移产物 SHA256:`74826011928fc6176f63b987e35573f6d6763f8cf00b419a1428632a9bc48c85`
- 状态:**pass**
- AsNumpy 版本:`0.3.1`
- AsNumpy 路径:`std/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
- 允许 fallback 逃生口:False
- 实际使用逃生口:False
- 迁移产物自身 build_inputs 已执行:True

## 执行来源门禁
- Audit 接口:NpuBackend.prepare_profile_guard
- Audit ID:unaudited-asnumpy-version
- 结论:严格 NPU ✓
- 迁移后算子:std
- strict forward 静态数据流:通过 ✓
- 静态 ap.* callsite:std×1
- 运行时 callsite 门禁:通过 ✓
  - kernel: std×1
  - artifact: std×1
- 模块路径:`std/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
- CANN:9.2.0-beta.2
- SoC:Ascend910B3

## 数值门禁
- 输入生成一致性:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=1.000e-05, rtol=1.000e-04
- 同一规范输入上的 forward:通过 ✓
  - max_abs=2.384e-07, max_norm=2.363e-07, atol=1.000e-05, rtol=1.000e-04
- 迁移产物 build_inputs + forward:通过 ✓
  - max_abs=2.384e-07, max_norm=2.363e-07, atol=1.000e-05, rtol=1.000e-04

## 线程与计时证据
- CPU 基线 BLAS 线程:1 (来源=threadpoolctl)
- NPU 评测进程 BLAS 线程:64 (来源=threadpoolctl)
- CPU 时延:15.956 ms
- NPU 时延:0.372 ms
- CPU 原始样本:warmup=3, repeats=10, samples_ms=[16.03491324931383, 16.483795829117298, 15.861362218856812, 15.955823473632336, 15.938342548906803, 15.679972246289253, 16.05675369501114, 16.45220536738634, 15.819131396710873, 15.570230782032013]
- NPU 原始样本:warmup=3, repeats=10, samples_ms=[0.3818906843662262, 0.37373136729002, 0.37585198879241943, 0.36697089672088623, 0.383191742002964, 0.3712419420480728, 0.37179235368967056, 0.370771624147892, 0.36794133484363556, 0.3695022314786911]
- NPU 计时口径:forward 输入预先驻留设备；每轮计时包含递归 D2H 回传同步
- 口径分类:kernel-only=False, end-to-end=False, 含 H2D=False, 每轮含 D2H 同步=True

## 性能结论
- **合格口径的 NPU/CPU 比值:42.9160×（NPU 快 42.92×）**
- 数值、provenance 与 CPU 基线一致性门禁均已通过。
