# AsNumpy 固定版本 API 清单

本文档绑定 NumPilot 当前静态 AsNumpy 组件，不代表所有 API 在所有输入上都一定调用 NPU。

## 组件身份

| 字段 | 值 |
|---|---|
| AsNumpy repository | https://gitcode.com/cann/asnumpy.git |
| source commit | 60d139c54d9e181b8ae101dcbd1479d6b2a86439 |
| source tree | f1c17d947a165c587a1b1d618d11d03fc5f5b183 |
| version | 0.3.1.dev70+g60d139c54.d20261108 |
| wheel | asnumpy-0.3.1.dev70+g60d139c54.d20261108-cp312-cp312-linux_aarch64.whl |
| wheel SHA-256 | d3ea03b2fd8c6eace56f532e2b30a917d63a138abe654c0020c14499898ed682 |
| build target | Linux AArch64 / CPython 3.12 / CANN 9.0+ (strict qualification receipt uses 9.2.0-beta.2) |
| license | Apache-2.0 |

AsNumpy 和 CANNBot 都通过 NumPilot 的静态锁定附件提供。运行时不从 Git 拉取、不解析 latest，也不自动升级；新版本由作者更新锁文件、附件、哈希和本文档后再发布。

安装兼容性由 AsNumpy 组件定义：当前冻结 wheel 面向 CANN 9.0 或更高版本，安装器不会把审计机的 CANN 精确版本当作通用安装锁；不同于当前 strict 资格画像的运行时仍需单独审计。

## API 表面

### 数组与数据交换

zeros, zeros_like, full, full_like, empty, empty_like, eye, ones, ones_like, identity, linspace, save, savez, savez_compressed, load.

### 直接线性代数

dot, inner, outer, vdot, matmul, einsum.

### linalg

matrix_power, qr, norm, det, slogdet, inv.

### 逻辑

all, any, equal, isfinite, isinf, isneginf, isposinf, logical_and, logical_or, logical_not, logical_xor, greater, greater_equal, less, less_equal, not_equal.

### 数学

add, negative, sin, cos, tan, arcsin, arccos, arctan, arctan2, hypot, radians, deg2rad, degrees, rad2deg, absolute, fabs, sign, heaviside, clip, nan_to_num, sqrt, square, relu, gelu, reciprocal, positive, multiply, divide, true_divide, subtract, floor_divide, float_power, fmod, mod, modf, remainder, divmod, power, prod, sum, nanprod, nansum, cumprod, cumsum, nancumprod, nancumsum, cross, exp, expm1, exp2, log, log10, log2, log1p, logaddexp, logaddexp2, real, signbit, ldexp, copysign, sinh, cosh, tanh, arcsinh, arccosh, arctanh, sinc, gcd, lcm, around, round_, rint, fix, floor, ceil, trunc, maximum, minimum, fmax, fmin, max, amax, nanmax, nanmin, min, amin.

### 其他公开模块与类型

- softmax
- sort
- mean
- random
- ndarray
- broadcast_shape
- can_cast, dtype, finfo, iinfo, isdtype, issubdtype, promote_types, result_type
- 类型别名：ArrayLike, DTypeLike, ShapeLike, AxisLike, AxisOptional, ScalarLike

## 当前已验证的 NPU 路径

| API | 已验证 admission | 其它输入 |
|---|---|---|
| matmul | 两个位置参数；AsNumpy ndarray；二维、非空、C 连续；两端 float32；内维匹配；ACLNN aclnnMatmul，cubeMathType=0 (KEEP_DTYPE) | 保留原 NumPy 兼容回退 |
| sin + matmul | 上述 FP32 MatMul profile；真实 NPU 输出已与 NumPy 做 shape、dtype、allclose 校验 | 小规模 workload 仍可能由 CPU 更快 |

inner、outer 以及 matmul 的非 admitted 情形当前仍走 Python host compatibility path。其它 API 的公开存在不等于已经完成 strict-NPU qualification；正式声明需要对应输入合同、运行时身份、无回退证据和数值测试。

## 2026-11-04 official ACLNN qualification overlay

本节记录一个**独立于静态组件**的 run-scoped overlay。静态组件仍是上表中的
`0.3.1.dev70+g60d139c54.d20261108`；overlay 不修改该 wheel、manifest 或 API catalog，
而是把一次隔离集成和六门 qualification 的证据绑定到一个显式证据根目录。只有调用方同时
提供 `--capability-overlay` 与 `--qualification-evidence-root` 时，`migrate` 才能使用它；
NumPilot 不自动发现、不联网拉取、不把 overlay 写成全局能力。

### 固定身份

| 字段 | 值 |
|---|---|
| qualification ID | `official-aclnn-35ce345fb531110e76a8` |
| overlay schema | `numpilot.official-aclnn-capability-overlay` |
| overlay revision | 1（parent 为零哈希） |
| overlay SHA-256 | `811be0c277a076a866d43de773ff1b9940b660ed4d21ed8f0319d28fa835238f` |
| producer | `numpilot-official-aclnn-qualification-v1` |
| runtime identity SHA-256 | `50ff6221b6be7bdf160b555b30ffcc15da5a3048fe5f0507670d506c6025c665` |
| runtime version | `0.4.1.post0+numpilot.bee27ba2` |
| CANN / SoC / Python | `9.2.0-beta.2` / `Ascend910B3` / `3.12.14` |
| base manifest SHA-256 | `048af828b641100264f5c8e14b0acfd45aa7e067888b78080c1d186948f13ac1` |
| admission audit ID | `unaudited-asnumpy-version+official-qualification-35ce345fb531110e76a8` |

六门 gate 均为 `passed`：`artifact_integrity` `9d8d7b88…`、`build` `fac7a9e6…`、
`unit_test` `04124cbd…`、`numerical_validation` `cb158984…`、`public_asnumpy_api`
`7d00f647…`、`npu_execution` `f42be44c…`。producer receipt SHA-256 为
`06f9ef899a7f8824691e68a205fed6ab60c765c8873c60e11d9a3d67dd41d07c`。

### 已签订 API profile

| API | 签名变体 | 准入范围 | 非准入输入 |
|---|---|---|---|
| `concatenate` | 1 个位置参数 + `axis` 关键字；2 个位置参数；1 个位置参数 | 输入序列元素为 float32、rank 2、C-contiguous；axis 为显式静态整数；输出按预计算 shape 分配 | audit 或 blocked，不进入 native 路径 |
| `ones` | 1 个位置参数 + `dtype` 关键字；2 个位置参数 | float32、rank 1..3、C-contiguous | 保留原公开行为，不声明 native |
| `zeros` | 1 个位置参数 + `dtype` 关键字；2 个位置参数 | float32、rank 1..3、C-contiguous | 保留原公开行为，不声明 native |

`numerical_validation` 覆盖 10 cases、30 次 NumPy oracle、73 个元素、0 mismatch；
`public_asnumpy_api` 覆盖 3 个 API、7 个签名变体和 8 个拒绝用例。已用该 overlay 完成
`np.ones → np.zeros → np.concatenate` 的整程序迁移，验证回执为
`417b870a10281c3256f8b72c934d86e0bb72b8c393d20a3c4f35c8243a0c82af`，18 个元素、
`max_abs_diff=0.0`。

### 证据边界

- 该 overlay 只对绑定 runtime identity 生效；identity、base manifest、audit ID 或任一 gate
  哈希漂移时 admission 失败关闭。
- `strict_npu_verified=false`、`performance_valid=false`；overlay 不是 `ACTIVATED` 状态迁移，
  也不改写 recovery state 的 `INTEGRATED_CANDIDATE / READY`。
- 静态组件默认安装仍只能声明 MatMul 的既有 profile；没有显式 overlay 时，上述三个 API 的
  native 路径不可用。
- 复验应只读加载现有 overlay 和 receipt，不要为了“再确认”重跑 producer；重跑会发布
  revision 2 并产生新的证据身份。

## 关联固定 Skill

CANNBot static bundle：

- repository commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
- tree: 3a95d925b085dfb401233f26f44fa9fab3d31999
- bundle SHA-256: da0ee3d6a12561da0d630125d7f4ae8602848dbd4c6f1b3f7150b26a57ead79f
- license: CANN Open Software License Agreement Version 2.0
- bundle: components/cannbot-skills/39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c/bundle.json (canonical static payload)


