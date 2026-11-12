"""Canonical AsNumpy code-format guide for configured models.

The framework must not assume that an external model already knows the shape of
an AsNumpy program.  This module is the single, deterministic description of
what a migrated ``forward`` may look like; the Planner's translation taskbook
supplies the per-program facts, and this guide supplies the language contract.

It is intentionally plain text: it is embedded in model prompts, rendered in
reports, and never executes code.
"""

from __future__ import annotations


ASNUMPY_IMPORTS = "import numpy as np\nimport asnumpy as ap"

ASNUMPY_FORMAT_GUIDE = """\
# AsNumpy 代码格式（必须遵守）

AsNumpy 是 NumPy 在昇腾 NPU 上的运行时替身。一个可被 NumPilot 验收的迁移产物
必须同时满足“数值等价”和“可审计的表面”两个条件。

## 1. 导入与边界

```python
import numpy as np
import asnumpy as ap
```

- `build_inputs()` 保持使用 NumPy 在 CPU 上构造输入，不要改成 AsNumpy。
- 只在 `forward(...)` 的入口把 host ndarray 上传到设备：

```python
if isinstance(x, np.ndarray):
    x = ap.ndarray.from_numpy(x)
```

- 已经是设备数组（NPUArray）的入参保持原样；标量入参保持原样。
- 每个数组入参都要单独做一次这种幂等边界判断。

## 2. forward 的合法形状

`forward` 必须是**直线代码**（straight-line），并且只能包含：

1. 上述 `isinstance(..., np.ndarray)` + `ap.ndarray.from_numpy(...)` 上传语句；
2. `结果名 = ap.<op>(...)` 形式的单名称赋值；
3. 恰好一条 `return`，返回一个名字，或由名字/直接 `ap.*` 调用组成的
   `dict` / `tuple` / `list`；
4. 字符串注释可以保留，其他语句一律不允许。

禁止出现在 `forward` 内：

- Python 二元/一元/比较运算（`+ - * / ** < ==` 等）；
- `for` / `while` / `if`（除上述上传判断）/ `try` / `with`；
- 调用任何 helper、`getattr`、`eval`、`exec`、`__import__`；
- 残留任何 `np.*` 计算调用；
- `.to_numpy()`、`.tolist()`、`.item()`、`.__array__()` 等 host escape；
- `import` 语句（导入只能放在模块顶层）；
- 把 `ap.*` 的结果丢弃不用；
- 多目标赋值、元组解包、链式赋值。

如果输入程序本身含有 helper、静态循环、`if`、解包或别名拼写，Planner 交给你的
源码已经由确定性准备阶段处理过；请直接迁移你看到的源码，不要重新引入这些结构。

## 3. 算子映射

- 直接调用：`np.<op>(...)` → `ap.<op>(...)`，参数顺序和关键字保持不变。
- 别名使用规范名：`np.abs` → `ap.absolute`，`np.acos` → `ap.arccos`，
  `np.asin` → `ap.arcsin`，`np.atan` → `ap.arctan`，`np.atan2` → `ap.arctan2`，
  `np.pow` → `ap.power`，`np.round` → `ap.around`。
- 只有当调用契约明确允许时，才保留关键字参数。任务书里标为
  `consent_required` / `gap_required` 的调用点，不能靠删掉关键字来“绕过”：
  删掉 `out=`、`where=`、`initial=`、`kind=` 等会改变语义。
- 二元算子（`add`/`subtract`/`multiply`/`divide`/`matmul`/`maximum`/
  `minimum`/`power` 等）**两侧都必须是设备数组**，不能直接传 Python 标量。
  遇到标量参与二元运算时，使用明确的设备数组物化，例如：

```python
zero = ap.full(x.shape, 0, dtype=x.dtype)
y = ap.maximum(x, zero)
```

  ReLU 形式的 `np.maximum(x, 0)` 优先写成 `ap.relu(x)`（仅当任务书确认
  `relu` 可用且原调用就是该语义）。
- `np.matmul` → `ap.matmul`；不要自行替换成 `dot` 或其他未在任务书中出现的
  API。
- 归约算子的 `axis`、`keepdims`、`dtype` 必须与原调用一致；任务书未确认的
  关键字不得新增。
- 排序：`np.sort(x, kind="stable")` 可写成 `ap.sort(x, stable=True)`；
  `kind="quicksort"` 可省略 `kind`。其他 `kind` 值不得静默丢弃。
- 如果任务书把某个调用点标为 `gap_required`，不要伪造替代实现，也不要换成
  另一个语义不同的 AsNumpy API；该程序应走补缺口路线。

## 4. 返回与输出

- 返回的每个数据叶子必须来自直接的 `ap.*` 计算结果，或来自已上传设备的
  `forward` 入参透传。
- 返回值可以是 NPUArray；验证器会用 `.to_numpy()` 把它取回 host 比较。
- 不要为了“看起来像 NumPy”在 `forward` 里把结果转回 host；输出边界由验证器
  负责。
- 不要新增、删除或重命名输出键；保持原程序的输出名称集合完全一致。

## 5. 验收门禁（模型输出不是成功证明）

迁移代码会经过独立门禁，任何一条不通过都视为失败：

1. **表面审计**：上面的合法形状与禁用项逐条静态检查；
2. **输入门禁**：迁移产物不能静默替换或修改 `build_inputs()` 的输入语义；
3. **数值门禁**：同一组输入分别执行原始 NumPy 程序与迁移程序，比较输出名称、
   shape、dtype、数值、NaN/Inf 语义；
4. **能力门禁**：实际调用的每个 `ap.*` 必须由任务书或运行时能力清单覆盖。

只输出完整、可解析的 Python 源码，用 ```python 代码块包裹；不要输出解释文字。
"""


def render_asnumpy_guide() -> str:
    """Return the canonical guide text embedded in model prompts."""

    return ASNUMPY_FORMAT_GUIDE
