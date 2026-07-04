"""AsNumpy 覆盖集快照。

来源:真实 asnumpy/src/asnumpy/__init__.py 的 _LAZY_MAPPING 键。
再生成:python -c "import asnumpy,inspect; ..." 或手工比对上游。
本 demo 用静态快照,避免运行时依赖 asnumpy。
"""

# 从 _LAZY_MAPPING 派生的算子名(去掉 dtype 别名与模块名如 linalg/random/custom_ops)
SUPPORTED_OPS = frozenset({
    # array 创建
    "empty", "empty_like", "eye", "full", "full_like", "identity",
    "linspace", "ones", "ones_like", "zeros", "zeros_like",
    # linalg.direct
    "dot", "einsum", "inner", "matmul", "outer", "vdot",
    # logic
    "all", "any", "equal", "greater", "greater_equal", "isfinite",
    "isinf", "isneginf", "isposinf", "less", "less_equal",
    "logical_and", "logical_not", "logical_or", "logical_xor", "not_equal",
    # math
    "absolute", "add", "amax", "amin", "around", "arccos", "arccosh",
    "arcsin", "arcsinh", "arctan", "arctan2", "arctanh", "ceil", "clip",
    "copysign", "cos", "cosh", "cross", "cumprod", "cumsum", "deg2rad",
    "degrees", "divide", "divmod", "exp", "exp2", "expm1", "fabs", "fix",
    "floor", "floor_divide", "fmax", "fmin", "fmod", "float_power", "gelu",
    "gcd", "heaviside", "hypot", "lcm", "ldexp", "log", "log10", "log1p",
    "log2", "logaddexp", "logaddexp2", "max", "maximum", "min", "minimum",
    "mod", "modf", "multiply", "nan_to_num", "nancumprod", "nancumsum",
    "nanmax", "nanprod", "nansum", "negative", "power", "positive", "prod",
    "rad2deg", "radians", "real", "reciprocal", "relu", "remainder", "rint",
    "round_", "sign", "signbit", "sin", "sinc", "sinh", "sqrt", "square",
    "subtract", "sum", "tan", "tanh", "true_divide", "trunc",
    # sorting / statistics / nn
    "sort", "mean", "softmax",
    # custom_ops
    "angle",
})


# asnumpy 提供的"自定义 / 非 numpy 原生"算子:relu/gelu/softmax 是 numpy 没有的 nn 算子,
# angle 在 asnumpy 里走 custom_ops。迁移时命中这些即"接入了自定义 NPU 算子"。
CUSTOM_OPS = frozenset({"relu", "gelu", "softmax", "angle"})


def is_supported(name: str) -> bool:
    """判断某 numpy 函数名是否被 AsNumpy 覆盖。"""
    return name in SUPPORTED_OPS


def is_custom(name: str) -> bool:
    """判断某算子是否为 asnumpy 自定义 / 非 numpy 原生算子(relu/gelu/softmax/angle)。"""
    return name in CUSTOM_OPS
