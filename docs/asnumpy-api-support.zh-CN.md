# AsNumpy 固定版本 API 清单

本文档绑定 NumPilot 当前静态 AsNumpy 组件，不代表所有 API 在所有输入上都一定调用 NPU。

## 组件身份

| 字段 | 值 |
|---|---|
| AsNumpy repository | https://gitcode.com/cann/asnumpy.git |
| source commit | 1e7177f12039b2d87e3cdff6693459d895744f68 |
| source tree | a5fb1f3a70be9fc17ecf0dd928776924d4c78e1e |
| version | 0.4.1.dev12+gc9115e8ce.d20261031 |
| wheel | asnumpy-0.4.1.dev12+gc9115e8ce.d20261031-cp312-cp312-linux_aarch64.whl |
| wheel SHA-256 | a8c47cfe71994f3bfd4180d7354df3904d2bc96c476c3119083fefac747a2138 |
| build target | Linux AArch64 / CPython 3.12 / CANN 9.2.0-beta.2 |
| license | Apache-2.0 |

AsNumpy 和 CANNBot 都通过 NumPilot 的静态锁定附件提供。运行时不从 Git 拉取、不解析 latest，也不自动升级；新版本由作者更新锁文件、附件、哈希和本文档后再发布。

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

## 关联固定 Skill

CANNBot static bundle：

- repository commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
- tree: 3a95d925b085dfb401233f26f44fa9fab3d31999
- bundle SHA-256: da0ee3d6a12561da0d630125d7f4ae8602848dbd4c6f1b3f7150b26a57ead79f
- license: CANN Open Software License Agreement Version 2.0
- bundle: components/cannbot-skills/39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c/bundle.json (canonical static payload)


