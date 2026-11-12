# NumPilot 性能评测报告 · argpartition_contract

- 源程序:`argpartition/source/argpartition_contract.py`
- 源码 SHA256:`0ddb6a97b1dc177f4e1c36d445e0dfbde2311e23639acc347fa75204abc5400d`
- 迁移产物 SHA256:`7f185e9d91bea87f9e5d60135d8ffa2e2ca92faf657772610fc7831c7d8f7438`
- 状态:**pass**
- AsNumpy 版本:`0.3.1`
- AsNumpy 路径:`argpartition/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
- 允许 fallback 逃生口:False
- 实际使用逃生口:False
- 迁移产物自身 build_inputs 已执行:True

## 执行来源门禁
- Audit 接口:NpuBackend.prepare_profile_guard
- Audit ID:unaudited-asnumpy-version
- 结论:严格 NPU ✓
- 迁移后算子:argpartition
- strict forward 静态数据流:通过 ✓
- 静态 ap.* callsite:argpartition×1
- 运行时 callsite 门禁:通过 ✓
  - kernel: argpartition×1
  - artifact: argpartition×1
- 模块路径:`argpartition/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
- CANN:9.2.0-beta.2
- SoC:Ascend910B3

## 数值门禁
- 输入生成一致性:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=1.000e-05, rtol=1.000e-04
- 同一规范输入上的 forward:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=0.000e+00, rtol=0.000e+00
- 迁移产物 build_inputs + forward:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=0.000e+00, rtol=0.000e+00

## 线程与计时证据
- CPU 基线 BLAS 线程:1 (来源=threadpoolctl)
- NPU 评测进程 BLAS 线程:64 (来源=threadpoolctl)
- CPU 时延:0.010 ms
- NPU 时延:0.349 ms
- CPU 原始样本:warmup=3, repeats=10, samples_ms=[0.013910233974456787, 0.012380070984363556, 0.011489726603031158, 0.010759569704532623, 0.01049041748046875, 0.010379590094089508, 0.009969808161258698, 0.010279938578605652, 0.009990297257900238, 0.010089948773384094]
- NPU 原始样本:warmup=3, repeats=10, samples_ms=[0.3577824681997299, 0.35042036324739456, 0.3476319834589958, 0.34890230745077133, 0.34617073833942413, 0.34573208540678024, 0.35554077476263046, 0.34721195697784424, 0.36523211747407913, 0.34622102975845337]
- NPU 计时口径:forward 输入预先驻留设备；每轮计时包含递归 D2H 回传同步
- 口径分类:kernel-only=False, end-to-end=False, 含 H2D=False, 每轮含 D2H 同步=True

## 性能结论
- **合格口径的 NPU/CPU 比值:0.0301×（CPU 快 33.26×）**
- 数值、provenance 与 CPU 基线一致性门禁均已通过。
