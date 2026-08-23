# NumPilot 迁移报告

- 源文件:`examples/poisson_ritz_iteration_broadcast.py`
- 状态:**pass**
- 数值一致:True
- 差值:max_abs=9.54e-07, max_rel=1.43e-07, atol=1e-05, rtol=1e-04
- 迁移产物自身 build_inputs 已执行:是
- 三层门禁:
  - 输入生成:True
  - 同一规范输入 forward:True
  - 产物 build_inputs + forward:True

## 替换算子(np.* → ap.*, 引擎=ast)
- 共替换 **101** 处调用
- `ap.multiply` ×40
- `ap.divide` ×14
- `ap.add` ×11
- `ap.matmul` ×10
- `ap.subtract` ×7
- `ap.negative` ×5
- `ap.apply_gradient_descent` ×4
- `ap.clip` ×3
- `ap.exp` ×2
- `ap.outer` ×2
- `ap.smooth_l1_loss` ×2
- `ap.sqrt` ×1

## 执行证据
- 后端:**npu**
- asnumpy 模块:`/home/ma-user/work/numpilot-gap-runs/poisson-ritz-p0-20260821/direct-local-v5/asnumpy-p0-site-v10-move/asnumpy/__init__.py`
- asnumpy 版本:`0.1.dev106+g21368abf9.d20260822`(真机 wheel)
- capability audit:`asnumpy-0.1.dev106-g21368abf9-poisson-move-init4f4e8083-math62e0ba09-direct4479c588-utils1b298145-core4465ce6e-v1+qualification-1c5aa2ae5c4ebf2d`
- strict-NPU 门禁:True

## 性能对比
性能对比拆成两个独立进程，记录实际 BLAS 线程并避免设置互相污染。只有 strict-NPU provenance 通过时才发布加速比:
1. `python -m numpilot bench-cpu examples/poisson_ritz_iteration_broadcast.py --out /home/ma-user/.local/state/numpilot/runs/benchmark`
2. `python -m numpilot bench-npu examples/poisson_ritz_iteration_broadcast.py --out /home/ma-user/.local/state/numpilot/runs/benchmark`

## 卸载方案
- 策略:`whole_forward_npu`
- `add` → NPU
- `apply_gradient_descent` → NPU
- `clip` → NPU
- `divide` → NPU
- `exp` → NPU
- `matmul` → NPU
- `multiply` → NPU
- `negative` → NPU
- `outer` → NPU
- `smooth_l1_loss` → NPU
- `sqrt` → NPU
- `subtract` → NPU
