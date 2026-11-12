# 科学计算示例：维生素 C 电子密度网格完整链

这个示例把 GPU4PySCF 的真实 Vitamin C 场景，从“只迁移一个矩阵乘”扩展为完整的
DFT 网格密度计算链：

```text
projected      = D.T @ AO
rho            = sum(AO * projected, axis=0)
weighted       = rho * W
electron_count = sum(weighted)
```

其中 `D` 是 SCF 收敛后的密度矩阵，`AO` 是真实 DFT 网格上的原子轨道值，`W` 是网格积分权重。
科学公式没有改写；迁移改变的只是执行后端：`matmul`、`multiply`、`sum` 从 NumPy 单核
CPU 切换到 AsNumpy / Ascend NPU。

## 真实来源

- 上游仓库：[`pyscf/gpu4pyscf`](https://github.com/pyscf/gpu4pyscf)
- 上游提交：`5fde981316127cd3506d9d3c7d5633647fcc4508`
- 几何文件：[`source/Vitamin_C.xyz`](source/Vitamin_C.xyz)，20 个原子
- 科学内核：`gpu4pyscf/dft/numint.py:_dot_ao_dm` 中的 `cupy.dot(dm.T, ao)`，
  原始片段保存在 [`source/_dot_ao_dm.original.py`](source/_dot_ao_dm.original.py)
- 来源与哈希：[`source/origin.json`](source/origin.json)
- 许可证：[`source/LICENSE`](source/LICENSE)（Apache-2.0；源文件未单独声明版权人）

真实流程为：`Vitamin_C.xyz` → RKS/PBE/cc-pVDZ SCF（网格级别 1）→ 收敛密度矩阵 `dm`
和 81936 点 DFT 网格 AO 值 → 网格密度链 → 电子数。SCF 收敛总能量为
`-684.0044734045816` Hartree，电子数为 92。

## 文件

- [`numpy_fullchain.py`](numpy_fullchain.py)：单逻辑 CPU 参考实现；从已保存的
  SCF/网格输入重放完整链，不重新跑 SCF。
- [`asnumpy_fullchain.py`](asnumpy_fullchain.py)：AsNumpy 迁移实现；`matmul`、
  `multiply`、`sum` 全部通过审计过的运行时执行，每个算子调用先经过
  `Numpilot RuntimeProfileGuard` 校验。
- [`aclnn_evidence.py`](aclnn_evidence.py)：解析 AsNumpy 的 ACLNN 日志行，
  与回执中的逻辑调度计数逐项核对。
- [`run_fullchain_evidence.sh`](run_fullchain_evidence.sh)：在已准备好的服务器上
  生成一对 write-once CPU/NPU 证据目录。

输入张量 `dm_t=[208,208]`、`ao_t=[208,81936]`、`weights=[81936]` 以 float32、
C-contiguous 形式保存在服务器的 NumPy 基线目录中，不提交进仓库。

## 迁移代码

NumPy 参考：

```python
projected = np.matmul(dm_t, ao_t)
rho = np.sum(np.multiply(ao_t, projected), axis=0, dtype=np.float32)
weighted = np.multiply(rho, weights)
electron_count = np.sum(weighted, dtype=np.float32)
```

两种实现都先把保存的输入固定为连续 float32：

```python
dm_t = np.ascontiguousarray(inputs['dm_t'])
ao_t = np.ascontiguousarray(inputs['ao_t'])
weights = np.ascontiguousarray(inputs['weights'])
assert dm_t.dtype == ao_t.dtype == weights.dtype == np.float32
assert dm_t.flags.c_contiguous and ao_t.flags.c_contiguous and weights.flags.c_contiguous
```

AsNumpy（实际代码通过带守卫和调度计数的 helper 调用）：

```python
def matmul(lhs, rhs, scope):
    guard.validate_call('matmul', (lhs, rhs), {})
    counts(scope)['matmul'] += 1
    return ap.matmul(lhs, rhs)

# 完整链（驻留态与暖态复用同一组 helper）
projected = matmul(dm_device, ao_device, scope)
product = multiply(ao_device, projected, scope)
rho = sum_axis0(product, scope)
weighted = multiply(rho, weights_device, scope)
electron_count = sum_scalar(weighted, scope)
```

`multiply`、`sum_axis0`、`sum_scalar` 与 `matmul` 一样先调用 `guard.validate_call`，
再进入 `ap.multiply` / `ap.sum`；`sum_scalar` 处理 `axis=None` 时库内部的主机标量回传。
电子数没有使用 `ap.dot`：能力目录把 `dot` 标记为条件 CPU fallback，因此这里用
等价的 `multiply + sum` 设备归约，避免把静默回退算作 NPU 加速。

## 实测结果（Ascend 910B3，单卡）

环境：CANN `9.2.0-beta.2`、AsNumpy `0.3.1.dev70+g60d139c54.d20261108`、
Python 3.12.14；单逻辑 CPU（CPU0、BLAS=1）；`ASCEND_VISIBLE_DEVICES=6`
选择可见设备，回执记录 `visible_device_environment="6"`、`device_count=1`。

| 计时范围 | 单核 CPU 中位数 | NPU 驻留态中位数 | 驻留加速 | NPU 暖态端到端中位数 | 暖态加速 |
|---|---:|---:|---:|---:|---:|
| 只算 `matmul` | 273.922 ms | 12.182 ms | 22.49× | — | — |
| `matmul + rho` | 340.530 ms | 12.914 ms | 26.37× | — | — |
| 完整网格密度链 | 341.750 ms | 13.157 ms | **25.98×** | 17.419 ms | **19.62×** |

计时口径：

- **驻留态**：三个输入提前上传，计时包含设备端完整链、`projected`/`rho` 回传和设备同步，
  不含每轮 H2D。
- **暖态端到端**：每轮从主机输入开始，包含三次 H2D、设备端完整链、两次显式 D2H、标量
  归约的内部 D2H 和设备同步；排除 SCF、网格/AO 构造和进程冷启动。
- 均为 3 次预热 + 10 次采样，使用 `statistics.median`。

数值验收：

| 输出 | 最大绝对误差 | 相对 Frobenius 误差 |
|---|---:|---:|
| `projected`（208×81936） | `9.536743164e-06` | `1.335199896e-07` |
| `rho`（81936） | `1.831054688e-04` | `1.617231004e-07` |

电子数：NPU（驻留态与暖态）均为 `91.99974060058594`；CPU 用相同
`multiply + sum` 形式的参考值也是 `91.99974060058594`，差值为 `0.0`。
早期基线使用的 `np.dot(rho, weights)` CPU 值为 `91.99925231933594`，与
`sum` 形式相差 `4.8828125e-04`；这是 CPU 自身两种归约顺序的差异，不是 NPU 误差。

设备执行证据：

- `RuntimeProfileGuard` 的整体状态为 `profile_status=matched`、`strict_npu=true`，
  `required_ops=[matmul, multiply, sum]`。其中 `profile_scoped=["matmul"]`：运行时
  对 `matmul` 的每次调用做 dtype/shape/layout 校验（52/52 次 matched）；
  `multiply` / `sum` 每次调用做能力准入和运行时目标身份校验，不做逐输入 profile
  匹配；H2D/D2H 只由脚本的 dispatch 计数记录，不经过 `guard.validate_call`。
- AsNumpy 日志逐次记录 ACLNN 执行：全部计时范围合计 `aclnnMatmul` 52 次、
  `aclnnMul` 65 次、`aclnnReduceSum` 65 次、`aclnnFlatten` 26 次，与回执中的逻辑
  调度计数完全一致。

## 边界

- 这是**完整跑通真实 Vitamin C 电子密度网格计算流程**后，对其中矩阵与归约热点的
  NPU 迁移；SCF（约 294.7 秒）、网格/AO 构造仍在 CPU，不能把 19.62× 或 25.98×
  宣称为完整 SCF/分子计算加速。
- 输入在进入内核前显式转成 float32；这是当前已审计 profile 的要求，不声称
  float64 化学精度。
- 本示例是为 PPT 展示手写的迁移代码，使用与 NumPilot 相同的审计运行时和守卫；
  不声称由模型生成算子，也不替代 NumPilot 的自动迁移主链。
- 结果适用于本输入、dtype、shape 和本机运行时；扩大到其他分子/网格时需要重新
  做 dtype、layout、strict-NPU 和数值验收。

## 复现

在已准备好运行时和证据目录的服务器上：

```bash
set -euo pipefail
REPO=${NUMPILOT_REPO:-$HOME/work/numpilot}
BASE=${VC_EVIDENCE_DIR:-$HOME/work/documents/gpu4pyscf-vitamin-c-rdm-grid}
RUNTIME=$HOME/work/documents/gpu4pyscf-sandwich-demo/runtime-venv/bin/python

export PYTHONPATH="$REPO"
export NUMPILOT_BOOTSTRAP_RECEIPT=$HOME/work/documents/gpu4pyscf-sandwich-demo/bootstrap-receipt.json

# 已存在的真实 SCF/网格输入与旧回执
test -f "$BASE/real-numpy-baseline-20261111-v2/scf_grid_inputs.npz"

# CPU 参考（单逻辑核）
OPENBLAS_NUM_THREADS=1 taskset -c 0 \
  "$BASE/pyscf-venv/bin/python" "$REPO/examples/scientific/gpu4pyscf_vitamin_c/numpy_fullchain.py" \
  --baseline "$BASE/real-numpy-baseline-20261111-v2" \
  --out "$BASE/real-numpy-fullchain-20261111-v1"

# NPU 迁移（需要 CANN 环境、物理 NPU 6）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ASCEND_VISIBLE_DEVICES=6 OPENBLAS_NUM_THREADS=1 taskset -c 0 \
  "$RUNTIME" "$REPO/examples/scientific/gpu4pyscf_vitamin_c/asnumpy_fullchain.py" \
  --baseline "$BASE/real-numpy-baseline-20261111-v2" \
  --cpu-fullchain "$BASE/real-numpy-fullchain-20261111-v1" \
  --out "$BASE/real-asnumpy-fullchain-20261111-v1"

python3 "$REPO/examples/scientific/gpu4pyscf_vitamin_c/aclnn_evidence.py" \
  --log "$BASE/real-asnumpy-fullchain-20261111-v1.stdout" \
  --receipt "$BASE/real-asnumpy-fullchain-20261111-v1/asnumpy_fullchain_receipt.json" \
  --out "$BASE/real-asnumpy-fullchain-20261111-v1/aclnn_evidence.json"
```

`run_fullchain_evidence.sh` 把上述步骤封装成一次 write-once 运行；如果目标目录已存在，
脚本会拒绝覆盖。更早的 matmul-only 回执
（`real-numpy-baseline-20261111-v2`、`real-asnumpy-e2e-20261111-v1`）保留原样，用于历史对比。

## 与旧版 matmul-only 结果的关系

旧回执的 `27.869×` 驻留态与 `20.825×` 暖态，是“NPU 只做 matmul”对“CPU 做
matmul + rho + 电子数”的混合范围比较。本示例按相同范围重新测量后：

- 只做 matmul 的匹配范围加速为 `22.49×`（驻留态）；
- 完整链的驻留态为 `25.98×`、暖态端到端为 `19.62×`。

旧回执不被改写；PPT 建议统一使用本页的匹配范围数字。
