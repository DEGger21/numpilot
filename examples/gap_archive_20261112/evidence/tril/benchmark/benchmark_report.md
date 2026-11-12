# NumPilot 性能评测报告 · tril_contract

- 源程序:`tril/source/tril_contract.py`
- 源码 SHA256:`d916e98aa99223f185b6c6f6a4c5f24a47cc9cd50d774904e08f9b5fe55a1636`
- 迁移产物 SHA256:`c6ad3ecfd37635a8da8a1b60b93da4244d6cf3c4a9660f7cf1b93ce6838677eb`
- 状态:**pass**
- AsNumpy 版本:`0.3.1`
- AsNumpy 路径:`tril/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
- 允许 fallback 逃生口:False
- 实际使用逃生口:False
- 迁移产物自身 build_inputs 已执行:True

## 执行来源门禁
- Audit 接口:NpuBackend.prepare_profile_guard
- Audit ID:unaudited-asnumpy-version
- 结论:严格 NPU ✓
- 迁移后算子:tril
- strict forward 静态数据流:通过 ✓
- 静态 ap.* callsite:tril×1
- 运行时 callsite 门禁:通过 ✓
  - kernel: tril×1
  - artifact: tril×1
- 模块路径:`tril/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
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
- CPU 时延:14.157 ms
- NPU 时延:1.649 ms
- CPU 原始样本:warmup=3, repeats=10, samples_ms=[14.684637077152729, 14.240795746445656, 13.865862973034382, 14.110574498772621, 14.180496335029602, 14.242324978113174, 13.85310385376215, 14.156665652990341, 13.832923956215382, 14.064704068005085]
- NPU 原始样本:warmup=3, repeats=10, samples_ms=[1.6698483377695084, 1.6835182905197144, 1.6621770337224007, 1.6315784305334091, 1.669498160481453, 1.6292771324515343, 1.6274480149149895, 1.6269972547888756, 1.6487082466483116, 1.6114171594381332]
- NPU 计时口径:forward 输入预先驻留设备；每轮计时包含递归 D2H 回传同步
- 口径分类:kernel-only=False, end-to-end=False, 含 H2D=False, 每轮含 D2H 同步=True

## 性能结论
- **合格口径的 NPU/CPU 比值:8.5865×（NPU 快 8.59×）**
- 数值、provenance 与 CPU 基线一致性门禁均已通过。
