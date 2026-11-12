# NumPilot 迁移报告

- 源文件:`tril/source/tril_contract.py`
- 状态:**fail**
- 路线:`direct_migration`
- 迁移任务书:`translation-taskbook-2a34a9a52889149770701c77ff05b286517a05c076420e67e4e0ceaa016bb3c6` (`tril/custom-v1/migrate-npu-ast-v1/TRANSLATION_TASKBOOK.json`)
- 数值一致:False
- 执行异常:`migrated.capabilities` / `ValueError: runtime target snapshot is invalid`
- 迁移产物自身 build_inputs 已执行:否

## 六角色执行记录
- ① 分析智能体 Analyzer（understand，ok）：IR 2 节点/1 边；1 个调用点，缺口 0 个 · 模型=规则(无模型)
- ② 规划智能体 Planner（understand，ok）：直迁（Direct Migration） · 模型=规则(无模型)
- ③ 缺口智能体 API Gap Filler（migrate，ok）：0 个任务，无阻断 · 模型=规则(无模型)
- ④ 迁移智能体 Code Migrator（migrate，ok）：1 处改写，完整性=通过 · 模型=AST 确定性改写(规则,无模型)
- ⑤ 诊断智能体 Diagnostics Agent（validate，ok）：算子未通过能力/来源门禁（migrated.capabilities） · 模型=规则(无模型)
- ⑥ 报告智能体 Migration Reporter（validate，ok）：已写出报告、验证证据与迁移产物 · 模型=规则(无模型)

## 诊断
- 分类:`capability_gate` · 回退角色:`gap_filler`
- 结论:算子未通过能力/来源门禁（migrated.capabilities）

## 替换算子(np.* → ap.*, 引擎=ast)
- 共替换 **1** 处调用
- `ap.tril` ×1

## 执行证据
- 后端:**npu**
- asnumpy 模块:`<server-home>`
- asnumpy 版本:`0.3.1.dev70+g60d139c54.d20261108`(真机 wheel)
- capability audit:`?`
- strict-NPU 门禁:未执行（shim/非 strict）

## 性能对比
性能对比拆成两个独立进程，记录实际 BLAS 线程并避免设置互相污染。只有 strict-NPU provenance 通过时才发布加速比:
1. `python -m numpilot bench-cpu tril/source/tril_contract.py --out <server-home>`
2. `python -m numpilot bench-npu tril/source/tril_contract.py --out <server-home>`

## 卸载方案
- 策略:`whole_forward_npu`
- `tril` → NPU
