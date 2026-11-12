# NumPilot 性能评测报告 · ediff1d_contract

- 源程序:`ediff1d/source/ediff1d_contract.py`
- 源码 SHA256:`9681613c066b42bb13ca58d0173d1731d32e55617161e999468c0170f088f4d8`
- 迁移产物 SHA256:`57994ac02a16216a243ee765666e803e986d0520d528037187e13ee6c001cd9b`
- 状态:**pass**
- AsNumpy 版本:`0.3.1`
- AsNumpy 路径:`ediff1d/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
- 允许 fallback 逃生口:False
- 实际使用逃生口:False
- 迁移产物自身 build_inputs 已执行:True

## 执行来源门禁
- Audit 接口:NpuBackend.prepare_profile_guard
- Audit ID:unaudited-asnumpy-version
- 结论:严格 NPU ✓
- 迁移后算子:ediff1d
- strict forward 静态数据流:通过 ✓
- 静态 ap.* callsite:ediff1d×1
- 运行时 callsite 门禁:通过 ✓
  - kernel: ediff1d×1
  - artifact: ediff1d×1
- 模块路径:`ediff1d/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
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
- CPU 时延:4.480 ms
- NPU 时延:1.664 ms
- CPU 原始样本:warmup=3, repeats=10, samples_ms=[6.074297241866589, 4.557250998914242, 4.454520530998707, 4.485759884119034, 4.439780488610268, 4.480339586734772, 4.446149803698063, 4.480821080505848, 4.445699974894524, 4.464970901608467]
- NPU 原始样本:warmup=3, repeats=10, samples_ms=[1.667606644332409, 1.6644187271595001, 1.6612280160188675, 1.6529271379113197, 1.6597583889961243, 1.660437323153019, 1.6777673736214638, 1.6658985987305641, 1.6598077490925789, 1.6664378345012665]
- NPU 计时口径:forward 输入预先驻留设备；每轮计时包含递归 D2H 回传同步
- 口径分类:kernel-only=False, end-to-end=False, 含 H2D=False, 每轮含 D2H 同步=True

## 性能结论
- **合格口径的 NPU/CPU 比值:2.6918×（NPU 快 2.69×）**
- 数值、provenance 与 CPU 基线一致性门禁均已通过。
