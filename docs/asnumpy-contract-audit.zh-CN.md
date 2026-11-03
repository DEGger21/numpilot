# AsNumpy 运行时契约审计（2026-11-02，2026-11-03 复现）

本文记录当前 NumPilot 固定 AsNumpy 组件在目标机器上的最小契约探针结果。探针只执行本地 NumPy 与 AsNumpy 调用，不读取 llm.json、不调用模型、不联网、不编译源码。

## 固定身份

- AsNumpy version：0.4.1.dev12+gc9115e8ce.d20261031
- source commit：1e7177f12039b2d87e3cdff6693459d895744f68
- source tree：a5fb1f3a70be9fc17ecf0dd928776924d4c78e1e
- wheel SHA-256：a8c47cfe71994f3bfd4180d7354df3904d2bc96c476c3119083fefac747a2138
- manifest SHA-256：9b4e95fea4db4ce897c229f0f8b92e04c0cfc9dad9863275730a3f6edaf39db5
- 运行时：CPython 3.12.14、NumPy 2.5.4、CANN 9.2.0-beta.2、Ascend910B3

## 复现命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/asnumpy_contract_probe.py --out /absolute/private/asnumpy-contract.json
```

探针默认把语义差异作为报告结果而不是进程错误；CI 或发布资格门可以增加 `--require-match`，将任一差异转换为非零退出。

运行时身份采集不依赖调用者手工导出：未设置 `CANN_VERSION` 时从 `ASCEND_TOOLKIT_HOME` 等路径解析 CANN 版本，未设置 `ASCEND_SOC_VERSION` 时通过 pyACL `acl.get_soc_name()` 读取 SoC。两者都失败时回执明确记录 `unknown`，不会猜测。

## 结果

| API | 输入边界 | NumPy 结果 | 当前 AsNumpy 结果 | 结论 |
|---|---|---|---|---|
| fmax | float32，shape=(1)，NaN 与 2.0 | 2.0 | NaN | 语义不等价；不能仅按名称直迁 |
| copysign | float32，1.0 与 -0.0 | -1.0 | +0.0 | 有符号零丢失；需要 contract gap |
| add | float32，两个 0 维标量 | shape=() | shape=() | 当前边界匹配 |
| matmul | float32，2D contiguous，(2,3)×(3,2) | 正确矩阵结果 | 正确矩阵结果 | 当前 strict MatMul profile 匹配 |

## 对 NumPilot 的约束

1. `hasattr(asnumpy, api)`、导出列表或函数名相同都不能证明 API 可迁移。
2. capability catalog 必须继续绑定 dtype、shape、关键字、边界语义和运行时身份。
3. `fmax` 与 `copysign` 不能进入 direct strict profile；应保留为 API gap 或显式 fallback consent。
4. 当前 fp32 2D MatMul 可以继续作为 strict profile 的正例，但结论只覆盖 manifest 绑定的 wheel、CANN、SoC、Python ABI 和输入边界。
5. 后续 gap qualification 应复用同一份 probe/evidence 结构，不能把名称存在误写成已资格化。

## 2026-11-03 复现与 MatMul 可用性验证

在目标机同一固定组件上重新执行探针，生成新的回执：

- `documents/AsNumpy运行时契约审计-20261103.json`

结果与 2026-11-02 完全一致：4 个 case，2 个 `verified_match`（0 维 add、fp32 2D matmul），2 个 `semantic_mismatch`（fmax NaN、copysign 负零），0 个错误；运行时身份记录为 CPython 3.12.14、NumPy 2.5.4、CANN 9.2.0-beta.2、Ascend910B3，manifest 哈希不变。

另对补充的 MatMul API 执行真实 NPU 可用性验证，回执：

- `documents/NumPilot-matmul可用性验证-20261103.json`

9/9 项通过：

- 公共 `asnumpy.matmul` 存在；2D fp32、C contiguous、内维匹配时，dispatch 观测确认进入原生 `_core.matmul`。
- 原生 `aclnnMatmul` 路径执行 5 次，日志事件数与调用点一致；结果与 NumPy 对照一致（256×128 @ 128×64，`max_abs_diff=1.9e-05`）。
- 重复运行结果逐位一致。
- 3D batch、float64、1D 向量、内维不匹配均不进入原生路径；内维不匹配由 NumPy 抛出 `ValueError`。

结论：当前锁定 AsNumpy 组件中的 MatMul 原生 ACLNN 路径真实可用，但 NumPilot 资格范围仍只覆盖已审计的 strict profile（2D、fp32、C 连续、内维匹配）；其余输入只能按兼容回退处理，不能据此宣称 NPU 加速。
