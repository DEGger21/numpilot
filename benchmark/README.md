## Scope manifests

The original 150 programs under `benchmark/cases/` are immutable benchmark
inputs. Use `benchmark/scopes/numpy_compute_only.json` to select the 135-case
NumPy compute-only view without deleting or renumbering the corpus:

```bash
python -m numpilot.benchmark_suite \
  --root benchmark/cases \
  --scope benchmark/scopes/numpy_compute_only.json \
  --out runs/numpy-compute-only \
  --backend shim
```

The manifest assigns every discovered case to one auditable category:
`covered_compute`, `contract_limited`, `single_gap`, `official_gap`,
`multi_gap`, or `out_of_scope`. The default include list excludes only
`out_of_scope` (the 15 `J_out_of_subset` cases). Case-level overrides keep
`B001`/`B002`/`B003` contract consent and `B014` overlapping alias semantics
visible instead of treating them as ordinary direct cases.

The runner records the scope name, category, source, rationale, and category
counts in `summary.json`, `SUMMARY.md`, and each case `receipt.json`. A scope
run changes selection only; it never changes the source programs or the
underlying route expectations.

# NumPilot NumPy 程序迁移测试集

本目录为 `numpilot` 仓库的 `benchmark/`，包含 **150 个独立 NumPy 程序**，用于开发和评估将 NumPy 程序自动迁移为 AsNumpy 程序的框架。设计参考《NumPilot 测试集设计规范》。

**当前定位：v0 开发测试集。** 源码已完成 NumPy 端检查，但尚未完成全部迁移预期的独立审定，也没有完成真实 AsNumpy/NPU 环境下的全量迁移验证。目录名称代表设计意图，不能直接作为已验证的迁移分类标签。

本目录包含程序源码和使用说明。`scopes/numpy_compute_only.json` 提供不修改原始样例的可审计筛选视图；持久化 oracle、预注册记录、迁移报告或 L 组验证器变异用例仍不属于本目录。

## 1. 可以测试什么

- **数值正确性**：迁移前后每个命名输出的 shape、dtype、数值是否一致。
- **语义保持**：广播、原地修改、别名、非连续输入、空数组、NaN/Inf、轴和维度语义是否保留。
- **程序分析与改写**：跨函数调用、共享中间结果、循环、分支是否被正确处理。
- **缺口处理**：是否识别缺口，是否能通过等价改写或新算子补齐，是否正确合并相同契约的多个调用点。
- **拒绝行为**：对明确超出产品支持子集的程序，是否在执行前给出可理解的拒绝原因。

如果框架目标包含 NPU 执行，还需要单独验证设备执行位置和 CPU 回退情况。仅检查源码出现 `asnumpy` 或输出数值相同，不足以证明计算在 NPU 上完成。

## 2. 目录与分组

每个用例均位于 `cases/<分组>/<用例编号及名称>/program.py`，不依赖其他用例文件。

| 设计层级 | 目录 | 数量 | 主要测试意图 |
| --- | --- | ---: | --- |
| 纯覆盖 | `A_direct_simple` | 15 | 简单逐元素运算、规约、矩阵乘法 |
| 纯覆盖 | `B_direct_semantics` | 20 | 输出缓冲区、掩码、类型提升、布局和别名等语义陷阱 |
| 纯覆盖 | `C_direct_structure` | 15 | 多函数、循环、分支和共享中间结果 |
| 半覆盖 | `D1_partial_low_kernel` | 15 | 在较多已知 API 中插入少量缺口 API |
| 半覆盖 | `D2_partial_low_official` | 5 | 混合程序中的别名或组合等价改写 |
| 半覆盖 | `E_partial_high` | 15 | 多缺口程序的发现、补齐与失败定位 |
| 半覆盖 | `F_partial_consent` | 10 | NumPy 调用与指定 AsNumpy 快照的签名差异 |
| 半覆盖 | `G_partial_dedup` | 5 | 同一缺口契约的三个静态调用点 |
| 完全不覆盖 | `H_gap_official` | 15 | 8 个别名用例和 7 个组合用例 |
| 完全不覆盖 | `I_gap_model` | 15 | 一元、多输入/轴语义、类型和边界条件三档缺口用例 |
| 完全不覆盖 | `J_out_of_subset` | 15 | 动态执行、I/O、并发、回调、特殊 dtype 等拒绝场景 |
| 完全不覆盖 | `K_gap_all` | 5 | 多阶段、多调用点的缺口压力程序 |
| **合计** | | **150** | **设计配额为 50 / 50 / 50** |

例如：

```text
benchmark/
├── README.md
└── cases/
    ├── A_direct_simple/
    │   └── A001_add_sqrt/
    │       └── program.py
    ├── B_direct_semantics/
    ├── C_direct_structure/
    ├── D1_partial_low_kernel/
    │   └── D1001_signed_compression/
    │       └── program.py
    ├── ...
    ├── J_out_of_subset/
    └── K_gap_all/
```

`D1001` 表示 D1 组第 001 例，`D2001` 表示 D2 组第 001 例。编号在各组内部连续。

**135 个非 J 组程序用于数值执行；15 个 J 组程序用于静态分析和拒绝测试。** 这并不表示前 135 个程序均应成功迁移。

## 3. 程序接口与迁移边界

所有程序提供两个入口：

```python
inputs = build_inputs()       # dict[str, np.ndarray]
outputs = forward(**inputs)  # dict[str, np.ndarray]
```

- `build_inputs()` 在 CPU 上准备固定输入。非 J 组的随机输入使用固定种子，或使用确定性数组构造。
- `forward()` 返回命名数组；标量结果包装为 shape 为 `()` 的 NumPy 数组。
- helper 是该用例的一部分。迁移范围应包含从 `forward()` 可达的计算 helper。
- 默认迁移边界是计算入口及其可达计算代码，`build_inputs()` 保留在 CPU。
- 文件没有统一的命令行入口；直接执行 `python program.py` 通常只定义函数，不会执行用例。应显式调用上述两个入口。

若要评估“任意完整 Python 脚本的自动迁移”，需要另行覆盖顶层执行、导入组织和不同输入输出方式。本集合统一采用函数契约，不能单独代表任意脚本迁移能力。

覆盖统计也必须明确边界：输入构造、输出包装、helper 内调用、`@` 和其他数组运算符是否计入，需要采用一致口径。只统计 `np.<名称>` 是否出现在导出表中，不能证明调用契约受支持。

## 4. NumPy 端运行

已验证的环境为 Python **3.14.7**、NumPy **2.5.4**。部分用例使用 `np.acos`、`np.bitwise_count`、`np.linalg.vecdot` 等较新的 API；不要默认它们兼容 NumPy 1.x。测试其他环境时应记录实际版本。

以下路径均相对于仓库根目录。先进入自己的 `numpilot` 仓库检出目录，并激活已安装 NumPy 的 Python 环境；命令中的 `python` 指当前环境的解释器，不依赖固定盘符、检出目录名称或虚拟环境位置。

### 单个用例

在仓库根目录通过 PowerShell 执行：

```powershell
@'
from pathlib import Path
import runpy

path = Path("benchmark/cases/A_direct_simple/A001_add_sqrt/program.py")
case = runpy.run_path(str(path))
inputs = case["build_inputs"]()
outputs = case["forward"](**inputs)
for name, value in outputs.items():
    print(name, value.shape, value.dtype)
'@ | python -B -
```

### 全量源码与 NumPy 复现检查

下面的检查只运行原始 NumPy 程序，不运行迁移器。它会编译全部 150 个文件，跳过 J 组执行，对其余 135 个用例重新构建输入并运行两次，比较输出名称、类型、形状和数据哈希。

```powershell
@'
from pathlib import Path
import hashlib
import runpy
import numpy as np

root = Path("benchmark/cases")
paths = sorted(root.glob("*/*/program.py"))
if len(paths) != 150:
    raise RuntimeError(f"Expected 150 programs, found {len(paths)}")

def fingerprint(outputs):
    if not isinstance(outputs, dict) or not outputs:
        raise TypeError("forward must return a nonempty dict")
    result = {}
    for name, array in outputs.items():
        if not isinstance(name, str) or not isinstance(array, np.ndarray):
            raise TypeError("Outputs must be dict[str, np.ndarray]")
        header = repr((array.dtype.str, array.shape)).encode("utf-8")
        result[name] = hashlib.sha256(header + array.tobytes()).hexdigest()
    return result

executed = 0
skipped = 0
for path in paths:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    if path.parent.parent.name == "J_out_of_subset":
        skipped += 1
        continue
    case = runpy.run_path(str(path))
    fingerprints = []
    for _ in range(2):
        inputs = case["build_inputs"]()
        if not isinstance(inputs, dict) or not all(
            isinstance(k, str) and isinstance(v, np.ndarray)
            for k, v in inputs.items()
        ):
            raise TypeError(f"Invalid inputs: {path}")
        outputs = case["forward"](**inputs)
        fingerprints.append(fingerprint(outputs))
    if fingerprints[0] != fingerprints[1]:
        raise RuntimeError(f"Non-reproducible output: {path}")
    executed += 1

print(f"Compiled {len(paths)}; NumPy reproducibility passed {executed}; J skipped {skipped}")
'@ | python -B -
```

预期汇总为 `Compiled 150; NumPy reproducibility passed 135; J skipped 15`。此处哈希用于同环境下的重复运行检查，不作为不同设备、不同浮点实现之间的逐位相等要求。

J 组有意包含网络访问、文件操作、未播种随机数等行为。普通批量检查必须在加载执行前将其排除；正式测试应将源码交给静态分析器，验证它是否依据产品声明的支持子集拒绝执行。编译通过本身不证明拒绝测试通过。

## 5. 目前的验证范围

源码交付时完成了以下检查：

- 150 个文件的分组配额、连续编号、入口和规范行数范围。
- 135 个非 J 组程序的 NumPy 执行、命名数组输出和两次运行哈希一致性。
- 15 个 J 组程序的语法及拒绝场景静态检查，未执行其 `forward()`。
- B 组的 12 类语义陷阱属性检查，包括实际内存共享、布局和掩码保留值。
- D2/H 共 20 个用例的 NumPy 等价表达式比较，包括全部输出的 shape、dtype 和容差内数值。
- F 组 10 个用例与指定快照 Python 包装器签名的静态对照。
- G 组 5 个用例的三个缺口调用点，输入 shape、dtype 和显式参数一致性。
- 部分应用不变量检查，例如概率质量、下降步骤、矩阵重建、求解和位打包往返。

上述检查不等同于 AsNumpy 真机验证。专项检查脚本未包含在本目录中；本 README 提供的全量命令可以复查基础 NumPy 运行和复现性，不能替代全部专项检查或正式评测记录。

## 6. 已知限制与接入注意事项

### 输入约定与现有验证器不一致

测试集按设计规范使用 `dict` 输入和 `forward(**inputs)`。当前项目的 [验证入口](../numpilot/agents/verification.py) 中，`_normalise_inputs()` 只接受 `tuple/list`。接入前应统一接口，不能直接把这些字典传给旧入口，或依赖字典迭代顺序进行位置参数调用。

### 输入复制不能破坏布局和别名

B 组包含共享底层内存、非连续视图和负步长输入。当前验证器的 `_clone_tree()` 逐数组调用 `.copy()`，可能破坏共享关系和布局。

双端比较需要彼此独立但语义等价的输入：保持每一端内部的 shape、dtype、strides 和共享关系。不要先运行会修改输入的 NumPy 程序，再把同一批已修改数组交给 AsNumpy。对本集合，可分别调用两次确定性的 `build_inputs()` 准备两端输入；设备上传是否支持这些布局与共享语义仍需独立判断。

### B 组并非全部已证明可直接迁移

例如 [B001](cases/B_direct_semantics/B001_out_buffer/program.py) 的 `out=`、B002 的 `where=`、B003 的 `casting=`，在指定快照的对应包装器中不被接受。这些是有价值的语义测试，但不能仅凭所在目录将它们标成 `DIRECT`。相关冲突已在用例注释中说明。

### C 组包含超出现有直线迁移路径的结构

C 组使用多个 helper、循环和分支。现有 [严格 forward 审计](../numpilot/engines/ast_rewrite.py) 的部分路径面向直线计算，并拒绝未审计 helper 或控制流。应依据框架承诺支持的语言子集审定预期，区分当前能力与后续扩展目标。

### 缺失名称不代表必须生成新 kernel

D1/I 等组中的 API 不在快照导出表中，并不证明不存在等价组合。例如 `isnan(x)` 存在 `not_equal(x, x)` 这一组合候选，仍需验证目标端 dtype、NaN 和执行位置契约。

D2/H 的公式已做 NumPy 端输入实例比较，但“数学公式或 NumPy 表达式等价”不代表 AsNumpy 已支持其中每个精确调用契约，也不代表框架已经能发现并应用该改写。正式标签需要核对目标端实现、参数、类型、布局和边界行为。

### API 存在不代表 NPU 原生执行

设计规范指定的能力快照为：

```text
1929a3ba44eb532a9f544f774a249cdf1c4119be
```

该版本的 `matmul` 包装器内部调用 NumPy 计算并重新包装结果。因此，A 组中矩阵乘法的“API 存在”不能被解释为“原生 NPU 执行已通过”。是否允许 CPU 回退必须是框架明确的产品契约，并在报告中区分。

F 组当前主要覆盖关键字不被接受造成的签名差异。它们没有完整覆盖默认 axis、dtype 提升、NaN 传播等所有潜在目标端语义差异。

### 真实程序代表性有限

15 个用例注明了从本仓库 `examples/` 改编的来源，其余主要为人工构造。来源是注释中的相对路径，不是运行依赖。这些样例可用于开发回归，但不构成独立的外部真实项目测试集。

多数用例只有一套固定输入。正式正确性评测还应增加不同 shape、dtype、边界值、布局和随机种子的输入实例。单个输入通过不能证明改写对整个调用契约成立。

## 7. 正式迁移评测的使用原则

1. **先固定产品契约和版本。** 明确迁移边界、支持的 Python 子集、NumPy/AsNumpy 版本，以及是否允许 CPU 回退。
2. **独立审定预期。** 审核每例的分类、精确缺口和等价路径。目录配额不是正确答案；不要用迁移器输出倒填预期标签。
3. **准备可追溯参考结果。** 在迁移前运行原始 NumPy 程序，保存输入、命名输出及版本信息；若执行正式规范，则补齐 manifest 和预注册流程。
4. **验证每个输出及可观察副作用。** 检查名称集合、shape、dtype、数值、NaN/Inf、必要时的符号零，以及输入修改和别名语义。整数和布尔输出采用精确比较，浮点容差按计算类型设定。
5. **单独验证执行位置。** 若目标是 NPU，检查设备执行证据和主机回退；禁止仅用 NumPy 兼容替身验证后宣称真机迁移通过。
6. **分组报告结果。** 区分直接迁移、等价改写、缺口补齐、需同意回退和拒绝。J 组考察正确拒绝，不能混入数值迁移成功率。
7. **扩充泛化与验证器测试。** 增加独立真实项目、多输入实例；若要验证验证器自身可靠性，再补充规范中的 L 组变异测试。

这 150 份程序可以作为上述流程的源程序集合。正式对外报告迁移成功率之前，需要完成预期审核和真实目标环境验证。
