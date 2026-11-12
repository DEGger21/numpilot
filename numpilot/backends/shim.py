"""numpy 支撑的假 asnumpy 后端。让迁移后的 `import asnumpy as ap` 代码
在任意机器离线跑通, 数值与 numpy 恒等, 便于复现与 CI。"""
import types

import numpy as np

from numpilot.backends.base import ExecutionBackend


class _NDArray:
    """对标 asnumpy.NPUArray 的最小包装, 内部用 numpy 承载。"""
    def __init__(self, data):
        self._data = np.asarray(data)

    @staticmethod
    def from_numpy(arr):
        return _NDArray(arr)

    def to_numpy(self):
        return np.asarray(self._data)

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def itemsize(self):
        return self._data.itemsize

    @property
    def nbytes(self):
        return self._data.nbytes

    @property
    def strides(self):
        return self._data.strides


def _unwrap(x):
    return x._data if isinstance(x, _NDArray) else x


def _wrap(x):
    return _NDArray(x)


def _wrap_result(value):
    if isinstance(value, tuple):
        return tuple(_wrap_result(item) for item in value)
    return _wrap(value)


def _cast_result(value, dtype):
    if dtype is None:
        return value
    if isinstance(value, tuple):
        return tuple(np.asarray(item, dtype=dtype) for item in value)
    return np.asarray(value, dtype=dtype)


def _require_array(value, operation, position):
    if not isinstance(value, _NDArray):
        raise TypeError(
            "%s operand %d must be an asnumpy ndarray" % (operation, position)
        )


def _require_creation_dtype(dtype, operation):
    if dtype is None:
        raise TypeError("%s dtype must be explicit and non-None" % operation)
    return dtype


def _require_integer_axis(axis, operation):
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise TypeError("%s axis must be an integer" % operation)
    return int(axis)


def _concatenate(arrays, axis=0):
    """Offline stand-in for the qualified ``asnumpy.concatenate`` surface."""

    if not isinstance(arrays, (tuple, list)):
        raise TypeError("concatenate expects a tuple or list of arrays")
    return _wrap(np.concatenate([_unwrap(item) for item in arrays], axis=axis))


# Overlay operators are installed only when a validated official overlay asks
# for them.  The default shim surface stays exactly as before so an
# unqualified program cannot execute an op that the pinned surface rejects.
_OVERLAY_STANDINS = {
    "concatenate": _concatenate,
}


def _numpy_fallback_standin(name):
    """NumPy-equivalent stand-in for an explicitly admitted operator.

    Only names carried by a validated custom-operator admission file
    reach this factory.  The shim is the offline numerical oracle, so
    the stand-in delegates to the NumPy function of the same name; an
    unknown name is still rejected instead of being fabricated.
    """

    npfunc = getattr(np, name, None)
    if not callable(npfunc):
        raise ValueError("shim backend has no NumPy equivalent for %s" % name)

    def standin(*args, **kwargs):
        return _wrap_result(
            npfunc(
                *[_unwrap(item) for item in args],
                **{key: _unwrap(value) for key, value in kwargs.items()},
            )
        )

    standin.__name__ = name
    return standin


def install_overlay_ops(module, names, numpy_fallbacks=()):
    fallbacks = frozenset(str(name) for name in numpy_fallbacks)
    for name in names:
        if hasattr(module, name):
            continue
        factory = _OVERLAY_STANDINS.get(str(name))
        if factory is None:
            if str(name) not in fallbacks:
                raise ValueError(
                    "shim backend has no overlay stand-in for %s" % name
                )
            factory = _numpy_fallback_standin(str(name))
        setattr(module, name, factory)
    return module


def _make_module():
    m = types.ModuleType("asnumpy")

    # Match the public metadata aliases exposed by real AsNumpy.  Ordinary
    # whole-file migration changes ``np.float32`` to ``ap.float32`` as part of
    # the same namespace-only rewrite.
    for name in (
        "float16", "float32", "float64",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "complex64", "complex128", "bool_",
    ):
        setattr(m, name, getattr(np, name))

    def _u(name, npfunc):
        setattr(m, name, lambda a, *args, **kw: _wrap(npfunc(_unwrap(a), *args, **kw)))

    def _b(name, npfunc):
        setattr(m, name, lambda a, b, *args, **kw: _wrap(npfunc(_unwrap(a), _unwrap(b), *args, **kw)))

    # 二元 / 通用
    for name in ["add", "subtract", "multiply", "divide", "matmul", "dot",
                 "maximum", "minimum", "power"]:
        _b(name, getattr(np, name))
    m.outer = lambda a, b: _wrap(np.outer(_unwrap(a), _unwrap(b)))
    # 一元
    for name in [
        "exp", "sqrt", "square", "tanh", "sin", "cos", "log", "abs",
        "absolute", "negative",
    ]:
        _u(name, getattr(np, name))
    # 归约(带 axis 透传)
    for name in ["sum", "mean", "max", "min", "amax", "amin", "nanmin"]:
        _u(name, getattr(np, name))

    # 非 numpy 原生算子
    m.relu = lambda a: _wrap(np.maximum(_unwrap(a), 0))

    def _softmax(a, axis=-1):
        x = _unwrap(a)
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return _wrap(e / np.sum(e, axis=axis, keepdims=True))
    m.softmax = _softmax

    m.clip = lambda a, lower, upper, **kw: _wrap(
        np.clip(_unwrap(a), _unwrap(lower), _unwrap(upper), **kw)
    )

    def _smooth_l1_loss(predict, target, reduction="mean", beta=1.0):
        error = np.subtract(_unwrap(predict), _unwrap(target))
        absolute = np.abs(error)
        beta_value = np.float32(beta)
        bounded = np.minimum(absolute, beta_value)
        loss = (
            bounded * bounded / (np.float32(2.0) * beta_value)
            + (absolute - bounded)
        )
        if reduction == "none":
            return _wrap(loss)
        if reduction == "sum":
            return _wrap(np.sum(loss))
        if reduction == "mean":
            return _wrap(np.mean(loss))
        raise ValueError("reduction must be none, mean, or sum")

    def _apply_gradient_descent(var, alpha, delta):
        return _wrap(
            np.subtract(
                _unwrap(var),
                np.multiply(_unwrap(alpha), _unwrap(delta)),
            )
        )

    m.smooth_l1_loss = _smooth_l1_loss
    m.apply_gradient_descent = _apply_gradient_descent

    # 创建
    m.zeros = lambda shape, dtype=None: _wrap(np.zeros(
        shape, dtype=_require_creation_dtype(dtype, "zeros")
    ))
    m.ones = lambda shape, dtype=None: _wrap(np.ones(
        shape, dtype=_require_creation_dtype(dtype, "ones")
    ))

    def _linspace(start, end, steps=50, dtype=None):
        # 与真实 AsNumpy 公开签名保持一致: 只接受 start/end/steps/dtype,
        # 拒绝 numpy 独有的关键字(如 num/endpoint), 避免迁移代码在真实
        # NPU 上因 TypeError 而失败。
        if (
            isinstance(steps, bool)
            or not isinstance(steps, (int, np.integer))
            or int(steps) <= 0
        ):
            raise ValueError("linspace steps must be a positive integer")
        try:
            real_start = float(start)
            real_end = float(end)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("linspace start/end must be real scalars") from exc
        return _wrap(np.linspace(
            real_start, real_end, num=int(steps), dtype=dtype
        ))

    m.linspace = _linspace

    # Keep the offline semantic adapter complete with the frozen AsNumpy
    # coverage surface.  These wrappers deliberately mirror AsNumpy's public
    # argument order; they are CPU simulation only and never claim NPU
    # provenance.
    def _unary(name, *, accepts_dtype=False):
        npfunc = getattr(np, name)
        if accepts_dtype:
            def function(x, dtype=None):
                return _wrap_result(
                    _cast_result(npfunc(_unwrap(x)), dtype)
                )
        else:
            def function(x):
                return _wrap_result(npfunc(_unwrap(x)))
        setattr(m, name, function)

    def _binary(name, *, accepts_dtype=False, operand_policy="both"):
        npfunc = getattr(np, name)

        def check_operands(x1, x2):
            if operand_policy == "both":
                _require_array(x1, name, 1)
                _require_array(x2, name, 2)
            elif operand_policy == "first":
                _require_array(x1, name, 1)
            elif operand_policy == "either":
                if not isinstance(x1, _NDArray) and not isinstance(x2, _NDArray):
                    raise TypeError(
                        "%s requires at least one asnumpy ndarray" % name
                    )
            else:
                raise RuntimeError("unknown shim operand policy")

        if accepts_dtype:
            def function(x1, x2, dtype=None):
                check_operands(x1, x2)
                return _wrap_result(
                    _cast_result(npfunc(_unwrap(x1), _unwrap(x2)), dtype)
                )
        else:
            def function(x1, x2):
                check_operands(x1, x2)
                return _wrap_result(npfunc(_unwrap(x1), _unwrap(x2)))
        setattr(m, name, function)

    for name in (
        "absolute arccos arcsin arctan cos deg2rad degrees exp exp2 expm1 "
        "fabs isfinite isinf isneginf isposinf log log10 log1p log2 "
        "logical_not modf real rad2deg radians sign signbit sin sqrt square tan"
    ).split():
        _unary(name)
    for name in (
        "arccosh arcsinh arctanh ceil cosh floor negative positive reciprocal "
        "rint sinh tanh trunc"
    ).split():
        _unary(name, accepts_dtype=True)

    def _fix_or_sinc(name):
        npfunc = getattr(np, name)

        def function(x, dtype=None):
            return _wrap_result(
                _cast_result(npfunc(_unwrap(x)), dtype)
            )

        setattr(m, name, function)

    for name in ("fix", "sinc"):
        _fix_or_sinc(name)

    for name in (
        "arctan2 copysign heaviside hypot ldexp logaddexp logaddexp2 "
        "logical_and logical_or logical_xor matmul"
    ).split():
        _binary(name, operand_policy="both")
    for name in (
        "add divide divmod float_power floor_divide fmax fmin fmod gcd lcm "
        "maximum minimum mod multiply remainder subtract true_divide"
    ).split():
        _binary(name, accepts_dtype=True, operand_policy="both")
    for name in (
        "equal greater greater_equal less less_equal not_equal"
    ).split():
        _binary(name, accepts_dtype=True, operand_policy="first")
    _binary("power", accepts_dtype=True, operand_policy="either")

    def _direct_product(name):
        npfunc = getattr(np, name)

        def function(a, b):
            _require_array(a, name, 1)
            _require_array(b, name, 2)
            return _wrap_result(npfunc(_unwrap(a), _unwrap(b)))

        setattr(m, name, function)

    for name in ("dot", "inner", "outer", "vdot"):
        _direct_product(name)

    def _einsum(subscripts, *operands):
        if not isinstance(subscripts, str):
            raise TypeError("einsum subscripts must be a string")
        for index, operand in enumerate(operands, start=1):
            _require_array(operand, "einsum", index)
        return _wrap_result(
            np.einsum(subscripts, *(_unwrap(item) for item in operands))
        )

    m.einsum = _einsum

    m.empty = lambda shape, dtype=None: _wrap(np.empty(
        shape, dtype=_require_creation_dtype(dtype, "empty")
    ))
    m.empty_like = lambda prototype, dtype=None: _wrap(
        np.empty_like(
            _unwrap(prototype),
            dtype=_require_creation_dtype(dtype, "empty_like"),
        )
    )
    m.eye = lambda n, dtype=None: _wrap(np.eye(
        n, dtype=_require_creation_dtype(dtype, "eye")
    ))
    m.full = lambda shape, value, dtype=None: _wrap(
        np.full(
            shape,
            float(value),
            dtype=_require_creation_dtype(dtype, "full"),
        )
    )
    m.full_like = lambda other, value, dtype=None: _wrap(
        np.full_like(
            _unwrap(other),
            float(value),
            dtype=_require_creation_dtype(dtype, "full_like"),
        )
    )
    m.identity = lambda n, dtype=None: _wrap(np.identity(
        n, dtype=_require_creation_dtype(dtype, "identity")
    ))
    m.ones_like = lambda other, dtype=None: _wrap(
        np.ones_like(
            _unwrap(other),
            dtype=_require_creation_dtype(dtype, "ones_like"),
        )
    )
    m.zeros_like = lambda other, dtype=None: _wrap(
        np.zeros_like(
            _unwrap(other),
            dtype=_require_creation_dtype(dtype, "zeros_like"),
        )
    )

    def _truth_reduction(name):
        npfunc = getattr(np, name)

        def function(x, axis=None, keepdims=False):
            if axis is not None:
                if not (
                    isinstance(axis, tuple)
                    and len(axis) == 1
                    and not isinstance(axis[0], bool)
                    and isinstance(axis[0], (int, np.integer))
                ):
                    raise TypeError(
                        "%s axis must be a one-element integer tuple" % name
                    )
            return _wrap_result(
                npfunc(_unwrap(x), axis=axis, keepdims=keepdims)
            )

        setattr(m, name, function)

    for name in ("all", "any"):
        _truth_reduction(name)

    def _typed_reduction(name):
        npfunc = getattr(np, name)

        def function(a, axis=None, keepdims=False, dtype=None):
            if axis is None:
                kwargs = {"dtype": dtype} if name == "mean" else {}
                return np.asarray(npfunc(_unwrap(a), **kwargs)).item()
            axis_value = _require_integer_axis(axis, name)
            return _wrap_result(
                npfunc(
                    _unwrap(a), axis=axis_value, keepdims=keepdims, dtype=dtype
                )
            )

        setattr(m, name, function)

    for name in ("prod", "sum", "nanprod", "nansum", "mean"):
        _typed_reduction(name)

    def _extreme_reduction(name):
        npfunc = getattr(np, name)

        def function(a, axis=None, keepdims=False):
            if axis is None:
                return np.asarray(npfunc(_unwrap(a))).item()
            axis_value = _require_integer_axis(axis, name)
            return _wrap_result(
                npfunc(_unwrap(a), axis=axis_value, keepdims=keepdims)
            )

        setattr(m, name, function)

    for name in ("max", "amax", "amin", "nanmax", "min"):
        _extreme_reduction(name)

    def _cumulative(name):
        npfunc = getattr(np, name)

        def function(a, axis=None, dtype=None):
            axis_value = _require_integer_axis(axis, name)
            return _wrap_result(
                npfunc(_unwrap(a), axis=axis_value, dtype=dtype)
            )

        setattr(m, name, function)

    for name in ("cumprod", "cumsum", "nancumprod", "nancumsum"):
        _cumulative(name)

    def _around(x, decimals=0, dtype=None):
        return _wrap_result(
            _cast_result(np.around(_unwrap(x), decimals=decimals), dtype)
        )

    m.around = _around
    m.round_ = _around
    m.clip = lambda a, a_min, a_max: _wrap_result(
        np.clip(_unwrap(a), _unwrap(a_min), _unwrap(a_max))
    )
    def _cross(a, b, axis=None):
        _require_array(a, "cross", 1)
        _require_array(b, "cross", 2)
        return _wrap_result(np.cross(
            _unwrap(a),
            _unwrap(b),
            axis=_require_integer_axis(axis, "cross"),
        ))

    m.cross = _cross
    m.nan_to_num = lambda x, nan=0.0, posinf=None, neginf=None: _wrap_result(
        np.nan_to_num(
            _unwrap(x), copy=True, nan=nan, posinf=posinf, neginf=neginf
        )
    )

    def _sort(a, axis=-1, stable=True):
        axis = _require_integer_axis(axis, "sort")
        kind = "stable" if stable else "quicksort"
        return _wrap_result(np.sort(_unwrap(a), axis=axis, kind=kind))

    m.sort = _sort
    m.angle = lambda x: _wrap_result(np.angle(_unwrap(x)))

    def _relu(x, dtype=None):
        return _wrap_result(
            _cast_result(np.maximum(_unwrap(x), 0), dtype)
        )

    def _gelu(x, dtype=None):
        value = _unwrap(x)
        scaled = np.sqrt(2.0 / np.pi) * (
            value + np.float32(0.044715) * value * value * value
        )
        result = np.float32(0.5) * value * (1.0 + np.tanh(scaled))
        return _wrap_result(_cast_result(result, dtype))

    def _softmax_complete(x, axis=-1, dtype=None):
        value = np.asarray(_unwrap(x), dtype=dtype)
        exponent = np.exp(value - np.max(value, axis=axis, keepdims=True))
        return _wrap_result(exponent / np.sum(exponent, axis=axis, keepdims=True))

    m.relu = _relu
    m.gelu = _gelu
    m.softmax = _softmax_complete

    m.ndarray = _NDArray
    return m


class ShimBackend(ExecutionBackend):
    name = "shim"

    def __init__(
        self,
        extra_ops=(),
        analysis_profile_resolver=None,
        numpy_fallback_ops=(),
    ):
        self._module = install_overlay_ops(
            _make_module(),
            tuple(extra_ops or ()),
            tuple(numpy_fallback_ops or ()),
        )
        self._analysis_profile_resolver = analysis_profile_resolver

    def get_module(self):
        return self._module

    def prepare_analysis_profile_resolver(self, qualified_names):
        resolver = self._analysis_profile_resolver
        if resolver is None:
            raise RuntimeError("no analysis profile resolver is configured")
        return resolver.prepare_analysis_profile_resolver(qualified_names)

    def resolve_profile_capability(self, qualified_name, profile):
        resolver = self._analysis_profile_resolver
        if resolver is None:
            raise RuntimeError("no analysis profile resolver is configured")
        return resolver.resolve_profile_capability(qualified_name, profile)

    def to_device(self, np_array):
        return _NDArray.from_numpy(np_array)

    def to_host(self, device_array):
        return device_array.to_numpy()
