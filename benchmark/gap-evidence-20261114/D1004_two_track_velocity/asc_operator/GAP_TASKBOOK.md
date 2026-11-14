# 算子缺口生成任务书 / Operator gap taskbook

- Target: `numpy.diff`
- Status: `requirements_prepared_unverified`
- Taskbook ID: `gap-taskbook-f5a450716c1d019ef15e383dc223aeee054bedbf1decb25abbfd501a3976a3c5`
- Source: `benchmark/cases/D1_partial_low_kernel/D1004_two_track_velocity/numpilot-work/asc_operator/numpy_source.py`
- Source SHA-256: `8288101fd207e63795f16500acf6525cebd6118c07045e687170ee1911f78dc5`

本任务书在模型调用前留存，可交给外部 Codex 或开发者继续。JSON 文件保存机器可读的同一份需求。
Planner 整理需求；Gap Filler 生成候选；确定性验证执行验收。此任务书不是能力或性能证书。

## 1. 缺口来源、源码位置与原因

- 接入目标：`numpy.diff` → `asnumpy.diff`。
- Gap ID：`diff-5c181afd5736f1ee`。
- 既有路线：`ascendc_registry_candidate`；route_decisions 为已有决策，不能被任务书重新解释为已验证。

- 行 15，列 20：`numpy.diff(<arg0>, axis=<redacted>)`；原因：API 不在静态覆盖集。
- 行 16，列 21：`numpy.diff(<arg0>, axis=<redacted>)`；原因：API 不在静态覆盖集。

## 2. 已知参数与未确认事实

静态字面量只说明源码写了什么，不代表已经运行或取得张量 profile。

- 行 15，列 20：
  - 参数 0：`left`（unknown）。
  - `axis=1`（static_literal）。
- 行 16，列 21：
  - 参数 0：`right`（unknown）。
  - `axis=1`（static_literal）。

- 未确认：axis, oracle_inputs, numerical_tolerances, candidate_support_and_rejection_profiles。
- 已观测运行时 profiles：`observed`。观测方式是在 CPU 上执行原 NumPy 程序并记录调用点实参，源码 SHA-256 与任务书绑定。

| 调用点 | 参数名 | dtype | shape | rank | C 连续 |
| --- | --- | --- | --- | --- | --- |
| 行 15 `numpy.diff` | `arg[0]` | `float32` | `[1024, 1024]` | 2 | True |
| 行 15 `numpy.diff` | 输出 | `float32` | `[1024, 1023]` | 2 | True |
| 行 16 `numpy.diff` | `arg[0]` | `float32` | `[1024, 1024]` | 2 | True |
| 行 16 `numpy.diff` | 输出 | `float32` | `[1024, 1023]` | 2 | True |

观测只覆盖 CPU 上实际执行到的调用点；设备区域在迁移验收前未被 AsNumpy 执行。

## 3. NumPy 语义

语义来源：`numpy.diff`。

- input and output dtype, dtype promotion, scalar materialization
- shape, axis normalization, broadcasting, layout and return structure
- NaN, positive/negative infinity, positive/negative zero and signbit
- out/where/casting and dtype keywords: preserve semantics or reject explicitly
- empty input, invalid axes/shapes, unsupported dtypes and exceptions

## 4. 本批最小支持范围与拒绝范围

- 输入 dtype：**float32**（已观测，见第 2 节）。
- 输出 dtype/返回结构服从 NumPy oracle；比较接口可以返回 bool，不能强转为 float32。
- 其他输入 dtype 显式拒绝或留待后续，不能静默 cast，也不能静默回退 CPU。
- 调用签名和 shape/dtype 声明只引用原 GapSpec；未知 rank/shape/layout 不等于任意范围已支持。
- 生成者须明确本候选支持哪些 profiles、拒绝哪些输入，并分别写正反测试。最终资格只覆盖实测范围。
- 本候选只需支持第 2 节已观测的输入 dtype/shape/rank/连续性组合；这是针对目标输入的定向特化算子，范围外输入必须显式拒绝，不做泛化承诺。

## 5. 官方优先与源码接入边界

- 目标 SoC：`Ascend910B3`；CANN：`9.2.0-beta.2`。未提供时先补齐，不能自行猜测。
- inspect pinned official CANN/ACLNN support first; custom Ascend C is a candidate when official reuse is not qualified
- reviewed isolated copy of bundled AsNumpy source; Python public API, pybind, native implementation and tests must agree
- .asc or op_host/op_kernel is acceptable when the required two-stage ACLNN interface is exposed
- 必须记录公开 API → Python wrapper → pybind → native → ACLNN 两段式符号映射。
- 接入依据引用：gap-spec.json; verified AsNumpy source manifest and adapter context plan; verified embedded CANNBot skill bundle; official operator resolution evidence。
- 禁止隐式改系统 CANN/共享 AsNumpy、CPU fallback 或激活未验证包。

## 6. 可执行 Oracle 与验收

Oracle 状态：`inputs_observed_cpu`；输入已在 CPU 上由原程序实际执行并记录（见第 2 节），设备区域仍须由确定性验证对拍。

```python
import numpy as np

def numpy_oracle(*args, **kwargs):
    return np.diff(*args, **kwargs)
```

误差要求：record dtype-specific rtol/atol before numerical qualification; compare boolean outputs exactly and preserve oracle special values/signbit

| 门禁 | 必需证据 |
| --- | --- |
| build | isolated build/import and public API binding checks |
| unit_and_boundary | NumPy oracle, admitted profiles, unsupported dtype/shape/kwargs, repeated input lifecycle |
| numerical | saved original inputs, expected/actual outputs, explicit tolerances, NaN/Inf/signed-zero checks |
| native_execution | same source/wheel/runtime identity, native call evidence and real NPU profiler; no host fallback |
| whole_program | original program versus migrated program, actual callsite profiles and all runtime guards |
| performance | separate performance receipt: one CPU core vs one NPU, timing scope, warmup, synchronization and raw samples; correctness alone is no speedup claim |
| activation | all required qualification gates and capability rescan; generated code remains unverified until then |

所有门禁当前均 pending，由确定性验证执行；生成成功不等于测试通过。性能单独验收，不能从正确性推断加速比。

## 交接与生成准入

- 下一责任角色：`gap_filler`；生成角色：`gap_filler`（配置模型、外部 Codex 或手工开发均可）。
- 允许生成未验证候选：`True`。
- 生成阻塞项：无；所有未知项仍阻塞资格发布。
- 失败时保留任务书和错误证据，修改候选后重新验证；不得用 Planner 自认证替代测试。

## 唯一规范快照 / Authoritative GapSpec

JSON 中 `#/gap_spec/...` 均指向以下快照，避免维护第二份 shape/dtype/route 规则。

```json
{
  "api_name": "diff",
  "asnumpy_api": "asnumpy.diff",
  "callsites": [
    {
      "call_expr": "numpy.diff(<arg0>, axis=<redacted>)",
      "col_offset": 20,
      "lineno": 15,
      "qualified_api": "numpy.diff",
      "reason": "API 不在静态覆盖集"
    },
    {
      "call_expr": "numpy.diff(<arg0>, axis=<redacted>)",
      "col_offset": 21,
      "lineno": 16,
      "qualified_api": "numpy.diff",
      "reason": "API 不在静态覆盖集"
    }
  ],
  "cann_version": "9.2.0-beta.2",
  "dtype_rules": [
    "runtime_profile_required"
  ],
  "gap_id": "diff-5c181afd5736f1ee",
  "qualified_api": "numpy.diff",
  "reference_cases": [],
  "route": "ascendc_registry_candidate",
  "route_decisions": [
    {
      "reason": "requires a semantic composition resolver",
      "route": "asnumpy_composite",
      "status": "not_evaluated"
    },
    {
      "reason": "check a pinned CANN operator catalog before new kernel work",
      "route": "cann_existing",
      "status": "review_required"
    },
    {
      "reason": "explicit draft generation is allowed; execution remains gated",
      "route": "ascendc_registry",
      "status": "candidate_for_draft"
    }
  ],
  "schema_version": 1,
  "semantic_source": "numpy.diff",
  "shape_rules": [
    "runtime_profile_required"
  ],
  "signature_variants": [
    {
      "keyword_args": [
        "axis"
      ],
      "positional_args": 1
    }
  ],
  "source_path": "benchmark/cases/D1_partial_low_kernel/D1004_two_track_velocity/numpilot-work/asc_operator/numpy_source.py",
  "source_sha256": "8288101fd207e63795f16500acf6525cebd6118c07045e687170ee1911f78dc5",
  "target_soc": "Ascend910B3"
}
```
