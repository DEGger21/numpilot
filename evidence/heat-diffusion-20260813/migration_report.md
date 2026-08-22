# NumPilot 迁移报告 · heat_diffusion_forward

- 生成时间:2026-08-13 23:22:10
- 源程序:`examples/heat_diffusion_forward.py`
- 迁移引擎:ast
- 执行后端:shim
- 状态:**pass**

## 多智能体阵容

| 智能体 | 职责 | 模型 / 引擎 |
|---|---|---|
| 编排智能体 Orchestrator | 统筹流水线 · 传递上下文 · 汇总报告 | 规则编排(无模型) |
| 分析智能体 Analysis | AST 静态解析算子 · 比对覆盖集 | Python AST(规则,无模型) |
| 决策智能体 Decision | 把整个 forward 划为单一 NPU 岛 | 确定性边界规则(无模型) |
| 迁移智能体 Migration | np.* → ap.* 代码改写 | AST 确定性改写(规则,无模型) |
| 验证智能体 Verification | 双端执行 · 数值对比 · 测速 | 数值对比 + 真机执行(无模型) |

## 算子分析
- numpy 调用点:7 处
- 去重算子数:3 个
- 静态门禁:API 名称与已建模签名均通过（运行时仍须 capability/provenance 审计）

## 算子替换(np.* → ap.*,共 7 处)
- `ap.add` ×4
- `ap.multiply` ×2
- `ap.subtract` ×1

## 命中的已有 AsNumpy 扩展 API
- 无；本次新生成自定义算子数量为 **0**。

## 卸载方案
- 策略:whole_forward_npu
- NPU:add, multiply, subtract
- CPU:—

## 执行证据
- shim 仿真(numpy 承载 asnumpy 语义,非真机)
- strict-NPU capability 门禁:未通过/未执行 ✗
- Audit ID:`runtime-shim-v1`
- 环境:CANN=9.0.0, SoC=unknown
- 非严格 NPU API:`add`
- 非严格 NPU API:`multiply`
- 非严格 NPU API:`subtract`

## 三层语义校验
- 输入生成一致性:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=1.000e-05, rtol=1.000e-04
- 同一规范输入上的 forward:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=1.000e-05, rtol=1.000e-04
- 迁移产物 build_inputs + forward:通过 ✓
  - max_abs=0.000e+00, max_norm=0.000e+00, atol=1.000e-05, rtol=1.000e-04
- 迁移产物自身 build_inputs 已执行:是

## 整网性能对比
- 未做真机性能对比。可跑:`numpilot bench-cpu <file> --out D` + `numpilot bench-npu <file> --out D`

- 迁移代码:`/home/ma-user/work/numpilot-runs/heat-diffusion-20260813/shim/migrated.py`
