# NumPilot 迁移报告

- 源文件:`flip/source/flip_contract.py`
- 状态:**pass**
- 路线:`direct_migration`
- 迁移任务书:`translation-taskbook-1abf90e5e62052a3cf4834a0789fb9065ed249d0c6241e9bab8361cffb8305e2` (`flip/custom-v1/migrate-npu-ast-v1/TRANSLATION_TASKBOOK.json`)
- 数值一致:True
- 差值:max_abs=0.00e+00, max_rel=0.00e+00, atol=1e-05, rtol=1e-04
- 迁移产物自身 build_inputs 已执行:是
- 三层门禁:
  - 输入生成:True
  - 同一规范输入 forward:True
  - 产物 build_inputs + forward:True

## 六角色执行记录
- ① 分析智能体 Analyzer（understand，ok）：IR 2 节点/1 边；1 个调用点，缺口 0 个 · 模型=规则(无模型)
- ② 规划智能体 Planner（understand，ok）：直迁（Direct Migration） · 模型=规则(无模型)
- ③ 缺口智能体 API Gap Filler（migrate，ok）：0 个任务，无阻断 · 模型=规则(无模型)
- ④ 迁移智能体 Code Migrator（migrate，ok）：1 处改写，完整性=通过 · 模型=AST 确定性改写(规则,无模型)
- ⑤ 诊断智能体 Diagnostics Agent（validate，skipped）：验证通过，无需诊断 · 模型=规则(无模型)
- ⑥ 报告智能体 Migration Reporter（validate，ok）：已写出报告、验证证据与迁移产物 · 模型=规则(无模型)

## 替换算子(np.* → ap.*, 引擎=ast)
- 共替换 **1** 处调用
- `ap.flip` ×1

## 执行证据
- 后端:**npu**
- asnumpy 模块:`flip/custom-v1/adapter-build-v1/stage/asnumpy/__init__.py`
- asnumpy 版本:`0.3.1`(真机 wheel)
- capability audit:`unaudited-asnumpy-version`
- strict-NPU 门禁:True

## 性能对比
性能对比拆成两个独立进程，记录实际 BLAS 线程并避免设置互相污染。只有 strict-NPU provenance 通过时才发布加速比:
1. `python -m numpilot bench-cpu flip/source/flip_contract.py --out <server-home>`
2. `python -m numpilot bench-npu flip/source/flip_contract.py --out <server-home>`

## 卸载方案
- 策略:`whole_forward_npu`
- `flip` → NPU
