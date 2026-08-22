# NumPilot 迁移报告

- 源文件:`examples/heat_diffusion_forward.py`
- 状态:**pass**
- 数值一致:True
- 差值:max_abs=0.00e+00, max_rel=0.00e+00, atol=1e-05, rtol=1e-04
- 迁移产物自身 build_inputs 已执行:是
- 三层门禁:
  - 输入生成:True
  - 同一规范输入 forward:True
  - 产物 build_inputs + forward:True

## 替换算子(np.* → ap.*, 引擎=ast)
- 共替换 **7** 处调用
- `ap.add` ×4
- `ap.multiply` ×2
- `ap.subtract` ×1

## 执行证据
- 后端:**shim**
- 仿真后端(shim):离线用 numpy 承载 asnumpy 语义,数值恒等,非真机
- capability audit:`runtime-shim-v1`
- strict-NPU 门禁:未执行（shim/非 strict）
- 非严格 NPU:add, multiply, subtract

## 性能对比
性能对比拆成两个独立进程，记录实际 BLAS 线程并避免设置互相污染。只有 strict-NPU provenance 通过时才发布加速比:
1. `python -m numpilot bench-cpu examples/heat_diffusion_forward.py --out out_demo`
2. `python -m numpilot bench-npu examples/heat_diffusion_forward.py --out out_demo`

## 卸载方案
- 策略:`whole_forward_npu`
- `add` → NPU
- `multiply` → NPU
- `subtract` → NPU
