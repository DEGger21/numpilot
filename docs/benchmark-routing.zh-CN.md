# Benchmark 路由、验证策略与结果回执

本页说明 `benchmark/` 150 个用例如何绑定预期路由、如何验证、如何生成逐用例回执，
以及当前全量结果。目录名称只表示设计意图，**不单独决定 DIRECT**。

## 1. 运行入口

```bash
cd <repo>
python -m numpilot.benchmark_suite \
  --root benchmark/cases \
  --out <run-dir> \
  --backend shim
```

- `--backend shim` 用 NumPy 承载 AsNumpy 语义，只证明数值与迁移路径，不证明真机执行。
- `--backend npu` 走 Ascend 后端；只有 `strict_npu` provenance 通过时才把
  `strict_npu_qualified` 记为 `true`，否则失败关闭。
- `--engine` 固定为 `ast`；批量 runner 不调用任何模型。
- `--expectations` 可覆盖默认的 `benchmark/expectations.json`。

每个用例必须有 `build_inputs() -> dict[str, numpy.ndarray]` 和
`forward(**inputs) -> dict[str, numpy.ndarray]`。`J_out_of_subset` 只做静态拒绝，
从不执行。

## 2. 四类预期路由

| 路由 | 含义 | 判定为 honored 的观测 |
| --- | --- | --- |
| `direct` | 调用契约在已审计 AsNumpy 表面内（允许文档化等价改写） | 数值验证通过的迁移 |
| `gap` | 程序触发已审计表面之外的契约 | 干净的 gap stop；若缺口被等价改写关闭，记为 `improved` |
| `audit` | 语义审计未解决前不得宣称 DIRECT | 数值失败或 gap stop；任何 `migrated` 都算 unexpected |
| `reject` | 超出支持子集，必须静态拒绝 | `refused`，且不执行源码 |

`benchmark/expectations.json` 保存组默认路由和逐用例覆盖。覆盖只用于契约有明确差异的
用例，例如：

- `B001`/`B002`/`B003`：`out=`、`where=`、`casting=` 被固定快照拒绝，归入
  `gap/contract_consent`，不做静默改写；
- `B014`：重叠别名的原地修改无法由“改写成新数组”保持，归入
  `audit/alias_semantics`；
- `D1001`、`G004`、`I001`、`I003`：缺口按文档化组合关闭，路由仍是 `gap`，
  结果记为 `improved`。

运行时会解析“用例覆盖优先、组默认其次”，并在每个用例的 `receipt.json` 里记录
`expected.source`（`case_override` / `group_default`）。

## 3. 验证策略

### 3.1 输入构造

- `build_inputs()` 保留在 CPU；迁移侧会执行自己的 `build_inputs()` 并单独比较输入。
- 双端 forward 使用同一份规范输入的独立克隆。
- 克隆按真实 `shares_memory` 分组，在同一字节缓冲区上按原偏移和 strides 重建视图，
  因此重叠别名、Fortran 顺序、负 stride 和只读标记都不会被 `.copy()` 破坏。
- 每个输出检查名称集合、shape、dtype；浮点按 `atol + rtol * abs(reference)` 比较，
  整数/布尔精确比较，NaN 只与 NaN 匹配，Inf 要求同号。

### 3.2 结构控制流

`numpilot/benchmark_lowering.py` 在分析前做整程序降级：

- 内联从 `forward` 可达的 helper，支持 tuple 返回和解构；
- 展开静态 `for range(...)`、`while` 和可判定的 `if`；
- NumPy 调用提升为临时变量；
- 无法证明的控制流（数组条件、非 range 迭代、动态分支）**失败关闭**，保留原源码并
  记录 `lowering_notes`。

原始程序始终作为 NumPy oracle；降级只用于分析/迁移源码。

### 3.3 文档化等价改写

`numpilot/equivalents.py` 只做可证明的等价：

- 别名：`abs/round/acos/...`；
- 组合：`var/std/ptp/vecdot/isnan/cbrt`、单轴 `all/any` 的 tuple 形式；
- 标量广播：`ufunc(array, scalar)` 物化为同 shape/dtype 数组；
- 已证明数组上的 `asarray` 恒等。

其中 `cbrt(x)` 下降为 `copysign(power(|x|, full_like(x, 1/3, x.dtype)), x)`，
保留负数和负零符号；`isnan(x)` 下降为 `not_equal(x, x)`，对实数浮点是精确等价。

## 4. 逐用例回执

每次运行在 `<out>/cases/<group>/<case>/receipt.json` 写入：

```json
{
  "schema": "numpilot.benchmark-case-receipt",
  "expected": {"route": "direct", "gap_kind": "", "source": "group_default",
               "verification": ["named_outputs", "shape_dtype", "..."]},
  "observed": {
    "status": "migrated",
    "outcome": "migrated",
    "analysis_admitted": true,
    "migration_completed": true,
    "numerical_verified": true,
    "strict_npu_qualified": null,
    "performance_qualified": null
  },
  "artifacts": {"source_sha256": "...", "analysis_source_sha256": "...",
                "out_dir": "cases/<group>/<case>"},
  "verdict": "honored"
}
```

这五个维度分别回答：

| 字段 | 含义 |
| --- | --- |
| `analysis_admitted` | 静态分析是否接纳该程序 |
| `migration_completed` | 是否产出迁移候选 |
| `numerical_verified` | 迁移结果是否通过双端数值验证 |
| `strict_npu_qualified` | 是否有完整的 strict-NPU provenance；shim 为 `null` |
| `performance_qualified` | 性能口径是否合格；Phase 9 前保持 `null` |

`verdict` 取 `honored`、`improved`、`unexpected`。汇总 `summary.json` 额外给出
`route_counts`、`group_counts`、`status_counts`；`summary.tsv` 和 `SUMMARY.md`
包含同样的路由与判定列。

## 5. 当前全量结果

运行目录：仓库外 `runs/phase8-benchmark-20261104/full-suite-03`（由 `--out` 指定；
backend `shim`，engine `ast`，150 例）。

| 指标 | 数值 |
| --- | ---: |
| total | 150 |
| migrated | 68 |
| refused | 15 |
| gap stop | 66 |
| audit numerical failure | 1 |
| route honored | 133 |
| gap closed (`improved`) | 17 |
| unexpected | 0 |

分组：

| group | total | migrated | unsupported | failed | refused |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_direct_simple | 15 | 15 | 0 | 0 | 0 |
| B_direct_semantics | 20 | 16 | 3 | 1 | 0 |
| C_direct_structure | 15 | 15 | 0 | 0 | 0 |
| D1_partial_low_kernel | 15 | 1 | 14 | 0 | 0 |
| D2_partial_low_official | 5 | 5 | 0 | 0 | 0 |
| E_partial_high | 15 | 0 | 15 | 0 | 0 |
| F_partial_consent | 10 | 0 | 10 | 0 | 0 |
| G_partial_dedup | 5 | 1 | 4 | 0 | 0 |
| H_gap_official | 15 | 13 | 2 | 0 | 0 |
| I_gap_model | 15 | 2 | 13 | 0 | 0 |
| J_out_of_subset | 15 | 0 | 0 | 0 | 15 |
| K_gap_all | 5 | 0 | 5 | 0 | 0 |

证据哈希：

```text
summary.json  1e17394addaab8028071ac593ba44d53d3c14a3574405237cd8ba73daf5a740f
SUMMARY.md    8032caa4a6deae3de46895b3d6319f902629426e510e524ce85264f9fde6c07b
summary.tsv   c77376239fe4bfe2ceee53071f82a3d7eec8280cc2933ac6194f712edd31b7e8
run log       acb80a686558eed540357ddbf19b9e2d0a218157bbc17d38fe304b048bfc8ebf
```

### 5.1 别名审计样例 B014

`B014_overlapping_alias` 的原始程序对 `base[1:]` 做原地 `+=`，同时返回
`base[:-1]` 的别名输出。只改写成新数组会丢失对别名输出的可观察修改。修正后的
输入克隆保留共享存储后，该用例稳定报告数值失败，路由判定为 `audit/honored`，
不计入迁移成功。

### 5.2 strict-NPU 维度样例

用资格运行时对 `A001_add_sqrt` 做一次 NPU 后端探针：

```text
backend npu@0.4.1.post0+numpilot.bee27ba2 cannot satisfy required APIs
(non_strict_npu=add,sqrt,square)
```

该运行记录 `strict_npu_qualified=false`，并保留 provenance 的
`non_strict_npu` 清单。这说明“数值迁移成功”与“strict-NPU 合格”是两个独立维度；
没有版本级审计的 API 不会被标成 strict-NPU。

## 6. 已知缺口

当前 66 个 gap stop 主要来自：

- 缺失 API：`spacing`、`nextafter`、`diff`、`pad`、`repeat`、`tile`、`roll`、
  `median`、`quantile`、`argmax`、`take`、`convolve`、`searchsorted`、
  `packbits`/`unpackbits`、`bincount`、`take_along_axis`、`stack`、
  `histogram2d`、FFT 与大部分 `linalg`；
- 契约差异：`logspace`/`geomspace` 的 `linspace` 参数要求静态标量，而 H012/H013
  的 `start`/`stop` 是 0 维数组；
- 官方 `concatenate` overlay 已发布，但尚未接入 benchmark 分析路由
  （D1005/G001/I006）；
- F 组 10 例与 B001–B003 需要显式同意分支，不能静默改写。

## 7. 复现与边界

- 全量 sweep 不调用模型、不访问网络。
- J 组只执行静态分析，`receipt.json` 中 `execution_performed` 固定为 `false`。
- shim 结果不能对外宣称 NPU 执行；性能数据必须走 Phase 9 的独立 bench 口径。
- 公开回执不包含私有绝对路径、token、endpoint、原始模型响应或 prompt 明文。
