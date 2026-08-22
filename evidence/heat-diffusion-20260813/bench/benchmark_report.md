# NumPilot 性能评测报告 · heat_diffusion_forward

- 源程序:`examples/heat_diffusion_forward.py`
- 源码 SHA256:`6f83b11ae500b2201b6bd1e416f1b67e935bf8d147d80a385f8c2eb605e2bc9f`
- 迁移产物 SHA256:`f1713c39aa9865b781dd0cb53dcab3c83e3278b9cb833936e491a36571f1b2bf`
- 状态:**pass**
- AsNumpy 版本:`0.3.1.dev4+g1929a3ba4`
- AsNumpy 路径:`/home/ma-user/work/numpilot-runs/heat-diffusion-20260813/audited-site/asnumpy/__init__.py`
- 允许 fallback 逃生口:False
- 实际使用逃生口:False
- 迁移产物自身 build_inputs 已执行:True

## 执行来源门禁
- Audit 接口:NpuBackend.prepare_profile_guard
- Audit ID:asnumpy-0.3.1.dev4+g1929a3ba4-source-1929a3ba4-v1
- 结论:严格 NPU ✓
- 迁移后算子:add, multiply, subtract
- strict forward 静态数据流:通过 ✓
- 静态 ap.* callsite:add×4, multiply×2, subtract×1
- 运行时 callsite 门禁:通过 ✓
  - kernel: add×4, multiply×2, subtract×1
  - artifact: add×4, multiply×2, subtract×1
- 模块路径:`/home/ma-user/work/numpilot-runs/heat-diffusion-20260813/audited-site/asnumpy/__init__.py`
- CANN:9.0.0
- SoC:unknown

## 数值门禁
- 输入生成一致性:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=1.000e-05, rtol=1.000e-04
- 同一规范输入上的 forward:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=1.000e-05, rtol=1.000e-04
- 迁移产物 build_inputs + forward:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=1.000e-05, rtol=1.000e-04

## 线程与计时证据
- CPU 基线 BLAS 线程:1 (来源=environment)
- NPU 评测进程 BLAS 线程:None (来源=unavailable)
- CPU 时延:15.552 ms
- NPU 时延:14.424 ms
- NPU 计时口径:forward 输入预先驻留设备；每轮计时包含递归 D2H 回传同步
- 口径分类:kernel-only=False, end-to-end=False, 含 H2D=False, 每轮含 D2H 同步=True

## 性能结论
- **合格口径的 NPU/CPU 比值:1.0782×（NPU 快 1.08×）**
- 数值、provenance 与 CPU 基线一致性门禁均已通过。
