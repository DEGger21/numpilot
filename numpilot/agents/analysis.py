"""分析智能体：提取 ``forward`` NPU 岛中的 NumPy 算子并比对覆盖集。"""
import ast
import hashlib
from math import prod

import numpy as _np

from numpilot.capabilities import AnalysisProfileSet, CapabilityMatch
from numpilot.contracts import AnalysisResult, OpUse
from numpilot.coverage import SUPPORTED_OPS, is_supported
from numpilot.operator_specs import collect_static_operator_specs


_ARRAY_ONLY_BINARY_OPS = frozenset({
    "add", "arctan2", "copysign", "cross", "divide", "divmod", "dot",
    "float_power", "floor_divide", "fmax", "fmin", "fmod", "gcd",
    "heaviside", "hypot", "inner", "lcm", "ldexp", "logaddexp",
    "logaddexp2", "logical_and", "logical_or", "logical_xor", "matmul",
    "maximum", "minimum", "mod", "multiply", "outer", "remainder",
    "subtract", "true_divide", "vdot",
})
_SCALAR_SECOND_BINARY_OPS = frozenset({
    "equal", "greater", "greater_equal", "less", "less_equal", "not_equal",
})
_EXPLICIT_DTYPE_CREATION_OPS = frozenset({
    "empty", "empty_like", "eye", "full", "full_like", "identity", "ones",
    "ones_like", "zeros", "zeros_like",
})
_EXPLICIT_DTYPE_POSITION = {
    "empty": 1,
    "empty_like": 1,
    "eye": 1,
    "full": 2,
    "full_like": 2,
    "identity": 1,
    "ones": 1,
    "ones_like": 1,
    "zeros": 1,
    "zeros_like": 1,
}
_VECTOR_AXIS_REDUCTIONS = frozenset({"all", "any"})
_SCALAR_AXIS_REDUCTIONS = frozenset({
    "amax", "amin", "max", "mean", "min", "nanmax", "nanprod", "nansum",
    "prod", "sum",
})
_REQUIRED_SCALAR_AXIS_OPS = frozenset({
    "cross", "cumprod", "cumsum", "nancumprod", "nancumsum",
})
_SAFE_ARRAY_METADATA = frozenset({
    "dtype", "itemsize", "nbytes", "ndim", "shape", "strides",
})
_STATIC_DTYPE_NAMES = frozenset({
    "bool", "bool_", "float16", "float32", "float64", "int8", "int16",
    "int32", "int64", "uint8", "uint16", "uint32", "uint64",
})
_STATIC_NUMPY_REAL_CONSTANTS = frozenset({
    "e", "euler_gamma", "inf", "nan", "pi",
})
_ARRAY_INPUT_CONSTRUCTORS = frozenset({
    "arange", "array", "asarray", "empty", "empty_like", "eye", "full",
    "full_like", "geomspace", "identity", "linspace", "logspace", "ones",
    "ones_like", "zeros", "zeros_like",
})


def _build_asnumpy_call_contracts():
    """Freeze the source-audited call surface used by module-only rewrites."""

    contracts = {}

    def add(
        names,
        positional_limit,
        keywords=(),
        required=(None,),
        positional_names=None,
    ):
        if positional_names is None:
            positional_names = (
                () if positional_limit is None
                else (None,) * positional_limit
            )
        if (
            positional_limit is not None
            and len(positional_names) != positional_limit
        ):
            raise RuntimeError("invalid AsNumpy positional contract")
        for name in names.split():
            if name in contracts:
                raise RuntimeError("duplicate AsNumpy call contract: " + name)
            contracts[name] = (
                tuple(required),
                positional_limit,
                frozenset(keywords),
                tuple(positional_names),
            )

    # Array creation.  Only keyword spellings shared by NumPy and AsNumpy are
    # admitted; for example NumPy ``fill_value`` becomes AsNumpy ``value`` and
    # therefore must remain positional.
    add(
        "empty", 2, ("shape", "dtype"), ("shape", "dtype"),
        ("shape", "dtype")
    )
    add(
        "empty_like",
        2,
        ("prototype", "dtype"),
        ("prototype", "dtype"),
        ("prototype", "dtype"),
    )
    # NumPy's second positional eye argument is M, while AsNumpy's is dtype.
    add("eye", 1, ("dtype",), (None, "dtype"), (None,))
    add(
        "full",
        3,
        ("shape", "dtype"),
        ("shape", None, "dtype"),
        ("shape", None, "dtype"),
    )
    add(
        "full_like", 3, ("dtype",), (None, None, "dtype"),
        (None, None, "dtype")
    )
    add(
        "identity", 2, ("n", "dtype"), ("n", "dtype"),
        ("n", "dtype")
    )
    add(
        "linspace",
        3,
        ("start", "dtype"),
        ("start", None),
        ("start", None, None),
    )
    add(
        "ones zeros",
        2,
        ("shape", "dtype"),
        ("shape", "dtype"),
        ("shape", "dtype"),
    )
    add(
        "ones_like zeros_like",
        2,
        ("dtype",),
        required=(None, "dtype"),
        positional_names=(None, "dtype"),
    )

    # Direct linear algebra.  ``einsum`` accepts a variable number of
    # positional operands but no NumPy-only options such as ``optimize``.
    add(
        "dot inner outer vdot",
        2,
        ("a", "b"),
        ("a", "b"),
        ("a", "b"),
    )
    add("matmul", 2, required=(None, None))
    add("einsum", None, required=(None, None))

    add(
        "all any",
        2,
        ("axis", "keepdims"),
        positional_names=(None, "axis"),
    )
    add("isfinite isinf logical_not", 1)
    add("logical_and logical_or logical_xor", 2, required=(None, None))
    add(
        "equal greater greater_equal less less_equal not_equal",
        2,
        required=(None, None),
    )

    add(
        "absolute arccos arcsin arctan cos deg2rad degrees exp exp2 expm1 "
        "fabs log log10 log1p log2 modf real rad2deg radians sign signbit "
        "sin sqrt square tan",
        1,
    )
    add(
        "arccosh arcsinh arctanh ceil cosh floor negative positive reciprocal "
        "rint sinh tanh trunc",
        1,
    )
    add(
        "fix isneginf isposinf sinc", 1, ("x",), ("x",), ("x",)
    )
    add(
        "arctan2 copysign heaviside hypot ldexp logaddexp logaddexp2",
        2,
        required=(None, None),
    )
    add(
        "add divide divmod float_power floor_divide fmax fmin fmod gcd lcm "
        "maximum minimum mod multiply power remainder subtract true_divide",
        2,
        required=(None, None),
    )

    add(
        "around round_",
        2,
        ("decimals",),
        positional_names=(None, "decimals"),
    )
    add(
        "clip",
        3,
        ("a", "a_min", "a_max"),
        ("a", "a_min", "a_max"),
        ("a", "a_min", "a_max"),
    )
    add(
        "cross",
        2,
        ("a", "b", "axis"),
        ("a", "b"),
        ("a", "b"),
    )
    add(
        "cumprod cumsum nancumprod nancumsum",
        3,
        ("a", "axis", "dtype"),
        ("a",),
        ("a", "axis", "dtype"),
    )
    add(
        "max amax amin min nanmax",
        2,
        ("a", "axis", "keepdims"),
        ("a",),
        ("a", "axis"),
    )
    add(
        "mean",
        2,
        ("a", "axis", "keepdims", "dtype"),
        ("a",),
        ("a", "axis"),
    )
    add(
        "nanprod nansum prod sum",
        2,
        ("a", "axis", "keepdims", "dtype"),
        ("a",),
        ("a", "axis"),
    )
    add(
        "nan_to_num",
        1,
        ("x", "nan", "posinf", "neginf"),
        ("x",),
        ("x",),
    )
    add("sort", 2, ("a", "axis"), ("a",), ("a", "axis"))

    # These are real AsNumpy extensions in the pinned public mapping. Runtime
    # provenance still decides whether an exact wheel may execute them.
    add("gelu relu", 1)
    add("softmax", 2, ("axis", "dtype"))

    missing = SUPPORTED_OPS.difference(contracts)
    extra = set(contracts).difference(SUPPORTED_OPS)
    if missing or extra:
        raise RuntimeError(
            "AsNumpy call contracts do not match coverage snapshot: "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    return contracts


_ASNUMPY_CALL_CONTRACTS = _build_asnumpy_call_contracts()
_AXIS_NONE_IGNORED_KEYWORD_DEFAULTS = {
    "all": {"keepdims": False},
    "amax": {"keepdims": False},
    "amin": {"keepdims": False},
    "any": {"keepdims": False},
    "max": {"keepdims": False},
    "mean": {"keepdims": False},
    "min": {"keepdims": False},
    "nanmax": {"keepdims": False},
    "nanprod": {"keepdims": False, "dtype": None},
    "nansum": {"keepdims": False, "dtype": None},
    "prod": {"keepdims": False, "dtype": None},
    "sum": {"keepdims": False, "dtype": None},
}
_MISSING_API_REASON = "API 不在静态覆盖集"


def resolve_numpy_function(func, numpy_bindings):
    """Resolve a call target through one conservative lexical binding map."""
    if isinstance(func, ast.Name):
        qualified = numpy_bindings.get(func.id)
        if qualified and qualified.startswith("numpy."):
            return qualified.rsplit(".", 1)[-1], qualified
    if isinstance(func, ast.Attribute):
        attributes = []
        root = func
        while isinstance(root, ast.Attribute):
            attributes.append(root.attr)
            root = root.value
        if isinstance(root, ast.Name) and root.id in numpy_bindings:
            path = ".".join(reversed(attributes))
            qualified = numpy_bindings[root.id] + "." + path
            return qualified.rsplit(".", 1)[-1], qualified
    return None


class _OpVisitor(ast.NodeVisitor):
    def __init__(
        self,
        numpy_bindings,
        domain_specs=None,
        domain_errors=None,
        known_array_names=(),
        known_scalar_names=(),
        known_non_array_names=(),
        strict_operand_types=False,
    ):
        # Local name -> canonical NumPy object, e.g. {"np": "numpy",
        # "fft": "numpy.fft.fft"}. Bindings come only from real imports.
        self.numpy_bindings = numpy_bindings
        self.domain_specs = dict(domain_specs or {})
        self.domain_errors = dict(domain_errors or {})
        self.known_array_names = set(known_array_names)
        self.known_scalar_names = set(known_scalar_names)
        self.known_non_array_names = set(known_non_array_names)
        self.strict_operand_types = bool(strict_operand_types)
        self._call_argument_depth = 0
        self.op_uses = []

    def visit_Call(self, node):
        function = resolve_numpy_function(node.func, self.numpy_bindings)
        if function is not None:
            name, qualified_name = function
            supported, reason = self._call_supported(
                name, qualified_name, node
            )
            self.op_uses.append(
                OpUse(
                    name=name,
                    lineno=getattr(node, "lineno", 0),
                    supported=supported,
                    est_elements=self._est_elements(name, node),
                    reason=reason,
                    col_offset=getattr(node, "col_offset", 0),
                    qualified_name=qualified_name,
                    call_expr=self._redacted_call_expr(
                        qualified_name, node
                    ),
                    positional_args=len(node.args),
                    keyword_args=tuple(sorted(
                        keyword.arg or "**" for keyword in node.keywords
                    )),
                )
            )
        elif isinstance(node.func, ast.Name) and node.func.id in self.domain_specs:
            spec = self.domain_specs[node.func.id]
            keywords = tuple(sorted(
                keyword.arg or "**" for keyword in node.keywords
            ))
            supported = spec.admits_call(len(node.args), keywords)
            self.op_uses.append(OpUse(
                name=spec.name,
                lineno=getattr(node, "lineno", 0),
                supported=supported,
                reason=(
                    None
                    if supported
                    else "调用签名不在 OperatorSpec 声明的 variants 中"
                ),
                col_offset=getattr(node, "col_offset", 0),
                qualified_name=spec.qualified_api,
                call_expr=self._redacted_call_expr(spec.qualified_api, node),
                positional_args=len(node.args),
                keyword_args=keywords,
                kind="domain_operator",
                target_api=spec.target_api,
                requires_qualification=True,
                operator_spec=spec.to_dict(),
            ))
        elif isinstance(node.func, ast.Name) and node.func.id in self.domain_errors:
            self.op_uses.append(OpUse(
                name="operator_spec." + node.func.id,
                lineno=getattr(node, "lineno", 0),
                supported=False,
                reason=self.domain_errors[node.func.id],
                col_offset=getattr(node, "col_offset", 0),
                kind="domain_operator",
            ))
        # Do not visit the function attribute as a value (``np.add`` is
        # already resolved above).  Module constants and dtype markers are
        # allowed only while visiting actual call arguments.
        self._call_argument_depth += 1
        try:
            for argument in node.args:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)
        finally:
            self._call_argument_depth -= 1

    def visit_Import(self, node):
        self._unsupported_python_structure(node, "Import")

    def visit_ImportFrom(self, node):
        self._unsupported_python_structure(node, "ImportFrom")

    def visit_Attribute(self, node):
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        # NumPy module attributes are handled by call resolution (or are
        # static constants/dtype markers).  Every value attribute surviving
        # migration must exist on both ndarray implementations.
        numpy_root = (
            isinstance(root, ast.Name)
            and root.id in self.numpy_bindings
        )
        allowed_numpy_value = (
            self._call_argument_depth > 0
            and node.attr in (
                _STATIC_DTYPE_NAMES
                | _STATIC_NUMPY_REAL_CONSTANTS
                | {"ndarray"}
            )
        )
        if (
            (numpy_root and not allowed_numpy_value)
            or (not numpy_root and node.attr not in _SAFE_ARRAY_METADATA)
        ):
            self.op_uses.append(OpUse(
                name="python_attribute." + node.attr,
                lineno=getattr(node, "lineno", 0),
                supported=False,
                reason=(
                    "属性不在 NumPy ndarray 与 AsNumpy ndarray 的共同只读表面"
                ),
                col_offset=getattr(node, "col_offset", 0),
            ))
        self.generic_visit(node)

    def visit_Assign(self, node):
        self.visit(node.value)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self._unsupported_python_structure(node, "DestructuringAssign")
            names = [
                item.id for target in node.targets for item in ast.walk(target)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
            ]
            self._record_name_kinds(names, "nonarray")
            return
        if not self.strict_operand_types:
            return
        kind = self._expression_kind(node.value)
        for target in node.targets:
            names = [
                item.id for item in ast.walk(target)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
            ]
            target_kind = (
                "array"
                if kind == "array_tuple" and isinstance(target, (ast.Tuple, ast.List))
                else "nonarray" if kind == "array_tuple" else kind
            )
            self._record_name_kinds(names, target_kind)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit(node.value)
            kind = self._expression_kind(node.value)
        else:
            kind = "nonarray"
        if not isinstance(node.target, ast.Name):
            self._unsupported_python_structure(node, "DestructuringAssign")
            kind = "nonarray"
        elif not self.strict_operand_types:
            return
        names = [
            item.id for item in ast.walk(node.target)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
        ]
        self._record_name_kinds(names, kind)

    # Defining a nested scope runs in forward, but its body does not. Skipping
    # the body also prevents shadowed inner aliases from leaking into results.
    def visit_FunctionDef(self, node):
        return None

    def visit_AsyncFunctionDef(self, node):
        return None

    def visit_ClassDef(self, node):
        return None

    def visit_Lambda(self, node):
        return None

    def visit_BinOp(self, node):
        self._unsupported_python_operator(node, type(node.op).__name__)
        self.generic_visit(node)

    def visit_Compare(self, node):
        names = ",".join(type(op).__name__ for op in node.ops)
        self._unsupported_python_operator(node, names)
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        # 字面量 -1 是函数参数语法的一部分，不代表 ndarray dunder。
        if not isinstance(node.operand, ast.Constant):
            self._unsupported_python_operator(node, type(node.op).__name__)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        # ``x += y`` needs device write-back and aliasing semantics that the
        # strict forward surface does not model; report it explicitly instead
        # of leaving migratable=False without a reason.
        self._unsupported_python_structure(
            node,
            "AugAssign",
            reason="原地运算需要 NPU 写回与别名语义，当前不在 strict 执行表面内",
        )
        self.generic_visit(node)

    def _unsupported_python_operator(self, node, operator):
        self.op_uses.append(OpUse(
            name="python_operator." + operator,
            lineno=getattr(node, "lineno", 0),
            supported=False,
            reason="当前 NPUArray 未验证对应 Python dunder，需显式 np.* 调用",
        ))

    def _unsupported_python_structure(self, node, structure, reason=None):
        self.op_uses.append(OpUse(
            name="python_structure." + structure,
            lineno=getattr(node, "lineno", 0),
            supported=False,
            reason=reason or "forward 内部 import 不在 strict NPU 可信执行表面",
        ))

    def _call_supported(self, name, qualified_name, node):
        # The frozen coverage snapshot describes the top-level AsNumpy
        # namespace. A matching leaf is not evidence that a nested API such
        # as numpy.linalg.matmul is exposed at the same path.
        if qualified_name != "numpy." + name or not is_supported(name):
            return False, _MISSING_API_REASON
        required, positional_limit, allowed_keywords, positional_names = (
            _ASNUMPY_CALL_CONTRACTS[name]
        )
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            return False, "AsNumpy 未验证动态位置参数展开: *args"
        risky_keywords = sorted(
            keyword.arg or "**kwargs"
            for keyword in node.keywords
            if keyword.arg is None or keyword.arg not in allowed_keywords
        )
        if risky_keywords:
            return False, "AsNumpy 未验证 NumPy 关键字: " + ", ".join(risky_keywords)
        keyword_names = {keyword.arg for keyword in node.keywords}
        duplicate_keywords = sorted({
            positional_name
            for index, positional_name in enumerate(positional_names)
            if positional_name is not None
            and index < len(node.args)
            and positional_name in keyword_names
        })
        if duplicate_keywords:
            return False, (
                "NumPy/AsNumpy 调用重复绑定位置与关键字参数: "
                + ", ".join(duplicate_keywords)
            )
        missing_required = [
            index + 1
            for index, keyword_name in enumerate(required)
            if index >= len(node.args)
            and (
                keyword_name is None
                or keyword_name not in keyword_names
            )
        ]
        if missing_required:
            return False, (
                "NumPy/AsNumpy 调用缺少共同可验证的必需参数位置: "
                + ", ".join(str(index) for index in missing_required)
            )
        if positional_limit is not None and len(node.args) > positional_limit:
            return False, (
                "AsNumpy 与 NumPy 的位置参数语义不兼容: "
                f"{name} 仅验证前 {positional_limit} 个位置参数"
            )
        if name in _EXPLICIT_DTYPE_CREATION_OPS:
            dtype = self._bound_argument(
                node, _EXPLICIT_DTYPE_POSITION[name], ("dtype",)
            )
            if not self._is_static_dtype(dtype):
                return False, (
                    "当前 AsNumpy 创建器要求显式且静态可验证的非 None dtype"
                )
        if name in {"full", "full_like"}:
            value = self._bound_argument(node, 1, ())
            if not self._is_static_real_scalar(value):
                return False, (
                    "当前 AsNumpy full value 仅验证静态实数标量"
                )
        if name == "linspace":
            start = self._bound_argument(node, 0, ("start",))
            end = self._bound_argument(node, 1, ())
            if not all(
                self._is_static_real_scalar(value)
                for value in (start, end)
            ):
                return False, (
                    "当前 AsNumpy linspace start/end 仅验证静态实数标量"
                )
            if len(node.args) > 2 and not self._is_static_positive_int(
                node.args[2]
            ):
                return False, (
                    "当前 AsNumpy linspace steps 必须是静态正整数"
                )
        if name == "einsum":
            subscripts = self._bound_argument(node, 0, ())
            if not (
                isinstance(subscripts, ast.Constant)
                and isinstance(subscripts.value, str)
            ):
                return False, (
                    "AsNumpy einsum 仅验证首参数为静态字符串的 subscript 形式"
                )

        axis_position = 2 if name == "cross" else 1
        axis = self._bound_argument(node, axis_position, ("axis",))
        if name in _VECTOR_AXIS_REDUCTIONS:
            if axis is not None and not self._is_literal_value(axis, None):
                if not self._is_static_single_axis_vector(axis):
                    return False, (
                        "当前 AsNumpy all/any axis 仅验证单元素静态整数 tuple"
                    )
        elif name in _SCALAR_AXIS_REDUCTIONS:
            if not self._is_static_integer_axis(axis):
                return False, (
                    "当前 AsNumpy reduction 要求静态整数 axis；axis=None "
                    "会返回与 NumPy dtype 不兼容的 Python scalar"
                )
        elif name in _REQUIRED_SCALAR_AXIS_OPS:
            if not self._is_static_integer_axis(axis):
                return False, (
                    "当前 AsNumpy 算子要求显式静态整数 axis"
                )
        elif name == "sort" and axis is not None:
            if not self._is_static_integer_axis(axis):
                return False, (
                    "当前 AsNumpy sort 仅验证缺省或静态整数 axis"
                )
        ignored_defaults = _AXIS_NONE_IGNORED_KEYWORD_DEFAULTS.get(name)
        if ignored_defaults:
            keywords = {
                keyword.arg: keyword.value
                for keyword in node.keywords
                if keyword.arg is not None
            }
            risky_axis_keywords = [
                keyword_name
                for keyword_name, default in ignored_defaults.items()
                if keyword_name in keywords
                and not self._is_literal_value(keywords[keyword_name], default)
            ]
            if risky_axis_keywords:
                axis = (
                    node.args[1]
                    if len(node.args) > 1
                    else keywords.get("axis")
                )
                if not self._is_static_non_none_axis(axis):
                    return False, (
                        "AsNumpy 在 axis=None 时忽略关键字: "
                        + ", ".join(sorted(risky_axis_keywords))
                    )
        first = self._bound_argument(node, 0, ("x1", "a"))
        second = self._bound_argument(node, 1, ("x2", "b"))
        if name in _ARRAY_ONLY_BINARY_OPS:
            operands_are_arrays = all(
                self._is_proven_array(value)
                for value in (first, second)
            )
            has_known_non_array = any(
                self._is_known_non_array(value)
                for value in (first, second)
            )
            if (
                (self.strict_operand_types and not operands_are_arrays)
                or has_known_non_array
            ):
                if name == "maximum" and self._is_relu_pattern(node):
                    return True, None
                return False, (
                    "当前 AsNumpy 二元 binding 要求两个操作数均可静态证明为数组，"
                    "不接受标量或其他非数组值"
                )
        elif name in _SCALAR_SECOND_BINARY_OPS:
            if (
                (self.strict_operand_types and not self._is_proven_array(first))
                or self._is_known_non_array(first)
            ):
                return False, (
                    "当前 AsNumpy comparison 仅验证数组作为第一个操作数"
                )
            if (
                self.strict_operand_types
                and not self._is_proven_array(second)
                and not self._is_known_scalar(second)
            ):
                return False, (
                    "当前 AsNumpy comparison 第二个操作数仅验证数组或数值标量"
                )
        elif name == "power":
            if not any(
                self._is_proven_array(value) for value in (first, second)
            ) and (
                self.strict_operand_types
                or all(
                    self._is_known_scalar(value) for value in (first, second)
                )
            ):
                return False, "当前 AsNumpy power 要求至少一个数组操作数"
            if self.strict_operand_types and any(
                not self._is_proven_array(value)
                and not self._is_known_scalar(value)
                for value in (first, second)
            ):
                return False, (
                    "当前 AsNumpy power 仅验证数组与数值标量操作数"
                )
        return True, None

    @staticmethod
    def _bound_argument(node, position, keyword_names):
        if position < len(node.args):
            return node.args[position]
        for keyword in node.keywords:
            if keyword.arg in keyword_names:
                return keyword.value
        return None

    @staticmethod
    def _is_literal_value(node, expected):
        return isinstance(node, ast.Constant) and node.value is expected

    @classmethod
    def _is_static_non_none_axis(cls, node):
        return (
            cls._is_static_integer_axis(node)
            or cls._is_static_single_axis_vector(node)
        )

    @staticmethod
    def _is_static_integer_axis(node):
        if isinstance(node, ast.Constant):
            return isinstance(node.value, int) and not isinstance(node.value, bool)
        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
            and not isinstance(node.operand.value, bool)
        )

    @classmethod
    def _is_static_single_axis_vector(cls, node):
        return (
            isinstance(node, ast.Tuple)
            and len(node.elts) == 1
            and cls._is_static_integer_axis(node.elts[0])
        )

    def _is_static_dtype(self, node):
        if isinstance(node, ast.Constant):
            return isinstance(node.value, str) and node.value in _STATIC_DTYPE_NAMES
        if isinstance(node, ast.Name):
            qualified = self.numpy_bindings.get(node.id, "")
            return (
                qualified.startswith("numpy.")
                and qualified.rsplit(".", 1)[-1] in _STATIC_DTYPE_NAMES
            )
        if isinstance(node, ast.Attribute):
            resolved = resolve_numpy_function(node, self.numpy_bindings)
            return (
                resolved is not None
                and resolved[1].count(".") == 1
                and resolved[0] in _STATIC_DTYPE_NAMES
            )
        return False

    def _is_static_real_scalar(self, node):
        if isinstance(node, ast.Constant):
            return (
                isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
            )
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
        ):
            return self._is_static_real_scalar(node.operand)
        if isinstance(node, ast.Attribute):
            resolved = resolve_numpy_function(node, self.numpy_bindings)
            return (
                resolved is not None
                and resolved[1].count(".") == 1
                and resolved[0] in _STATIC_NUMPY_REAL_CONSTANTS
            )
        return False

    @staticmethod
    def _is_static_positive_int(node):
        if isinstance(node, ast.Constant):
            return (
                isinstance(node.value, int)
                and not isinstance(node.value, bool)
                and node.value > 0
            )
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.UAdd)
            and isinstance(node.operand, ast.Constant)
        ):
            return (
                isinstance(node.operand.value, int)
                and not isinstance(node.operand.value, bool)
                and node.operand.value > 0
            )
        return False

    @staticmethod
    def _is_literal_scalar(node):
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float, complex, bool))
        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float, complex))
        )

    def _is_known_scalar(self, node):
        return self._is_literal_scalar(node) or (
            isinstance(node, ast.Name)
            and node.id in self.known_scalar_names
        )

    def _is_known_non_array(self, node):
        return self._is_known_scalar(node) or (
            isinstance(node, ast.Name)
            and node.id in self.known_non_array_names
        )

    def _is_proven_array(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.known_array_names:
                return True
            return (
                not self.strict_operand_types
                and node.id not in self.known_scalar_names
                and node.id not in self.known_non_array_names
            )
        if isinstance(node, ast.Call):
            resolved = resolve_numpy_function(node.func, self.numpy_bindings)
            if resolved is not None and resolved[0] in {"divmod", "modf"}:
                return False
            if (
                resolved is not None
                and resolved[1] == "numpy." + resolved[0]
                and resolved[0] in SUPPORTED_OPS
            ):
                return True
            if resolved is not None and resolved[1] == "numpy." + resolved[0]:
                candidate = getattr(_np, resolved[0], None)
                if isinstance(candidate, _np.ufunc):
                    return any(
                        self._is_proven_array(argument)
                        for argument in node.args[:candidate.nin]
                    )
            return False
        return False

    def _expression_kind(self, node):
        if isinstance(node, ast.Call):
            resolved = resolve_numpy_function(node.func, self.numpy_bindings)
            if resolved is not None and resolved[0] in {"divmod", "modf"}:
                return "array_tuple"
        if self._is_proven_array(node):
            return "array"
        if self._is_known_scalar(node):
            return "scalar"
        if isinstance(node, ast.Name) and node.id in self.known_non_array_names:
            return "nonarray"
        return "nonarray"

    def _record_name_kinds(self, names, kind):
        for name in names:
            self.known_array_names.discard(name)
            self.known_scalar_names.discard(name)
            self.known_non_array_names.discard(name)
            if kind == "array":
                self.known_array_names.add(name)
            elif kind == "scalar":
                self.known_scalar_names.add(name)
            else:
                self.known_non_array_names.add(name)

    @classmethod
    def _is_relu_pattern(cls, node):
        return (
            len(node.args) == 2
            and not node.keywords
            and (
                cls._is_numeric_zero(node.args[0])
                or cls._is_numeric_zero(node.args[1])
            )
        )

    @staticmethod
    def _is_numeric_zero(node):
        return (
            isinstance(node, ast.Constant)
            and not isinstance(node.value, bool)
            and node.value == 0
        )

    @staticmethod
    def _redacted_call_expr(qualified_name, node):
        """Build a structural summary without source values or identifiers."""
        parts = []
        for index, argument in enumerate(node.args):
            marker = "<arg%d>" % index
            if isinstance(argument, ast.Starred):
                marker = "*" + marker
            parts.append(marker)
        for keyword in node.keywords:
            if keyword.arg is None:
                parts.append("**<kwargs>")
            else:
                parts.append(keyword.arg + "=<redacted>")
        return qualified_name + "(" + ", ".join(parts) + ")"

    def _est_elements(self, name, node):
        if name not in ("zeros", "ones", "full", "empty"):
            return None
        if not node.args:
            return None
        shape_node = node.args[0]
        dims = self._const_shape(shape_node)
        return prod(dims) if dims else None

    @staticmethod
    def _const_shape(shape_node):
        if isinstance(shape_node, ast.Constant) and isinstance(shape_node.value, int):
            return [shape_node.value]
        if isinstance(shape_node, (ast.Tuple, ast.List)):
            dims = []
            for el in shape_node.elts:
                if isinstance(el, ast.Constant) and isinstance(el.value, int):
                    dims.append(el.value)
                else:
                    return None
            return dims
        return None


class _BindingCollector(ast.NodeVisitor):
    """Collect conservative bindings for one Python lexical scope."""

    def __init__(self, arguments=None):
        self.numpy_imports = {}
        self.other_bindings = set()
        if arguments is not None:
            positional = list(arguments.posonlyargs) + list(arguments.args)
            positional += list(arguments.kwonlyargs)
            self.other_bindings.update(arg.arg for arg in positional)
            if arguments.vararg is not None:
                self.other_bindings.add(arguments.vararg.arg)
            if arguments.kwarg is not None:
                self.other_bindings.add(arguments.kwarg.arg)

    @property
    def all_bound_names(self):
        return set(self.numpy_imports) | self.other_bindings

    def visit_Import(self, node):
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".", 1)[0]
            if alias.name == "numpy" or alias.name.startswith("numpy."):
                canonical = alias.name if alias.asname else "numpy"
                self.numpy_imports[local_name] = canonical
            else:
                self.other_bindings.add(local_name)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        is_numpy = (
            node.level == 0
            and (module == "numpy" or module.startswith("numpy."))
        )
        for alias in node.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            if is_numpy:
                self.numpy_imports[local_name] = module + "." + alias.name
            else:
                self.other_bindings.add(local_name)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.other_bindings.add(node.id)

    def visit_FunctionDef(self, node):
        self.other_bindings.add(node.name)

    def visit_AsyncFunctionDef(self, node):
        self.other_bindings.add(node.name)

    def visit_ClassDef(self, node):
        self.other_bindings.add(node.name)

    def visit_Lambda(self, node):
        return None

    def visit_ExceptHandler(self, node):
        if node.name:
            self.other_bindings.add(node.name)
        for statement in node.body:
            self.visit(statement)

    if hasattr(ast, "MatchAs"):
        def visit_MatchAs(self, node):
            if node.name:
                self.other_bindings.add(node.name)
            if node.pattern is not None:
                self.visit(node.pattern)

        def visit_MatchStar(self, node):
            if node.name:
                self.other_bindings.add(node.name)

        def visit_MatchMapping(self, node):
            if node.rest:
                self.other_bindings.add(node.rest)
            self.generic_visit(node)


def numpy_bindings_for_forward(tree, forward):
    """Return local names that conservatively still denote NumPy objects."""
    module_scope = _BindingCollector()
    for statement in tree.body:
        module_scope.visit(statement)

    local_scope = _BindingCollector(arguments=forward.args)
    for statement in forward.body:
        local_scope.visit(statement)

    bindings = {
        name: qualified
        for name, qualified in module_scope.numpy_imports.items()
        if name not in module_scope.other_bindings
        and name not in local_scope.all_bound_names
    }
    bindings.update({
        name: qualified
        for name, qualified in local_scope.numpy_imports.items()
        if name not in local_scope.other_bindings
    })
    return bindings


def _static_forward_parameter_kinds(tree, forward):
    """Classify explicit build_inputs leaves without executing user code."""

    builds = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_inputs"
    ]
    if len(builds) != 1:
        return None
    build = builds[0]
    bindings = numpy_bindings_for_forward(tree, build)
    kinds = {}

    def bound_argument(node, position, keyword):
        if position < len(node.args):
            return node.args[position]
        for item in node.keywords:
            if item.arg == keyword:
                return item.value
        return None

    def is_none(node):
        return isinstance(node, ast.Constant) and node.value is None

    def classify(node):
        if isinstance(node, ast.Name):
            return kinds.get(node.id, "unknown")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex, bool)):
                return "scalar"
            return "nonarray"
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
        ):
            return classify(node.operand)
        if isinstance(node, ast.BinOp):
            sides = (classify(node.left), classify(node.right))
            if "array" in sides:
                return "array"
            return "scalar" if sides == ("scalar", "scalar") else "nonarray"
        if isinstance(node, (ast.Tuple, ast.List, ast.Dict, ast.Set)):
            return "nonarray"
        if isinstance(node, ast.Attribute):
            resolved = resolve_numpy_function(node, bindings)
            if (
                resolved is not None
                and resolved[1].count(".") == 1
                and resolved[0] in _STATIC_NUMPY_REAL_CONSTANTS
            ):
                return "scalar"
            return "nonarray"
        if isinstance(node, ast.Call):
            resolved = resolve_numpy_function(node.func, bindings)
            if resolved is None:
                return "nonarray"
            name, qualified = resolved
            if qualified.startswith("numpy.random."):
                if name in {"rand", "randn"}:
                    return "array" if node.args else "scalar"
                if name in {"random", "random_sample", "standard_normal"}:
                    size = bound_argument(node, 0, "size")
                    return (
                        "scalar" if size is None or is_none(size) else "array"
                    )
                if name in {"normal", "uniform"}:
                    size = bound_argument(node, 2, "size")
                    if size is not None and not is_none(size):
                        return "array"
                    data_arguments = [
                        item for item in node.args[:2]
                    ] + [
                        item.value for item in node.keywords
                        if item.arg in {"loc", "scale", "low", "high"}
                    ]
                    return (
                        "array"
                        if any(classify(item) == "array" for item in data_arguments)
                        else "scalar"
                    )
                return "nonarray"
            if qualified == "numpy." + name and name in _ARRAY_INPUT_CONSTRUCTORS:
                return "array"
            if qualified == "numpy." + name and name in _STATIC_DTYPE_NAMES:
                return "scalar"
        return "nonarray"

    returned = None
    for statement in build.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            kinds[statement.targets[0].id] = classify(statement.value)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
        ):
            kinds[statement.target.id] = classify(statement.value)
        elif isinstance(statement, ast.Return):
            returned = statement.value
    if not isinstance(returned, ast.Tuple):
        return None
    arguments = list(forward.args.posonlyargs) + list(forward.args.args)
    if len(arguments) != len(returned.elts):
        return None
    return {
        argument.arg: classify(value)
        for argument, value in zip(arguments, returned.elts)
    }


class AnalysisAgent:
    def __init__(self, profile_resolver=None):
        """Create an analyzer with an optional, explicitly injected resolver.

        The resolver must expose ``resolve_profile_capability(qualified_name,
        profile)``.  No backend or overlay is discovered implicitly.
        """

        if profile_resolver is not None and not callable(
            getattr(profile_resolver, "resolve_profile_capability", None)
        ):
            raise TypeError(
                "profile_resolver must expose resolve_profile_capability"
            )
        self.profile_resolver = profile_resolver

    def analyze(
        self,
        source: str,
        source_path: str = "<mem>",
        *,
        profiles: AnalysisProfileSet = None,
        strict_input_types: bool = False,
    ) -> AnalysisResult:
        if profiles is not None and not isinstance(profiles, AnalysisProfileSet):
            raise TypeError("profiles must be AnalysisProfileSet")
        tree = ast.parse(source)
        static_specs = collect_static_operator_specs(tree)
        # build_inputs 和顶层展示逻辑按 MVP 约定保留在 CPU；只有 forward 属于
        # placement/capability 范围，避免报告出“构造输入也卸载到 NPU”的假证据。
        forwards = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "forward"
        ]
        op_uses = []
        for forward in forwards:
            numpy_bindings = self._numpy_bindings_for_forward(tree, forward)
            local_scope = _BindingCollector(arguments=forward.args)
            for statement in forward.body:
                local_scope.visit(statement)
            domain_specs = {
                name: spec
                for name, spec in static_specs.specs.items()
                if name not in local_scope.all_bound_names
            }
            domain_errors = {
                name: error
                for name, error in static_specs.errors.items()
                if name not in local_scope.all_bound_names
            }
            parameter_kinds = (
                _static_forward_parameter_kinds(tree, forward)
                if strict_input_types
                else None
            )
            visitor = _OpVisitor(
                numpy_bindings,
                domain_specs=domain_specs,
                domain_errors=domain_errors,
                known_array_names=(
                    name for name, kind in parameter_kinds.items()
                    if kind == "array"
                ) if parameter_kinds is not None else (),
                known_scalar_names=(
                    name for name, kind in parameter_kinds.items()
                    if kind == "scalar"
                ) if parameter_kinds is not None else (),
                known_non_array_names=(
                    name for name, kind in parameter_kinds.items()
                    if kind == "nonarray"
                ) if parameter_kinds is not None else (),
                strict_operand_types=parameter_kinds is not None,
            )
            for statement in forward.body:
                visitor.visit(statement)
            op_uses.extend(visitor.op_uses)
        self._apply_profile_capabilities(source, op_uses, profiles)
        unsupported = sorted({op.name for op in op_uses if not op.supported})
        return AnalysisResult(
            source_path=source_path,
            op_uses=op_uses,
            unsupported=unsupported,
            migratable=(len(forwards) == 1 and not unsupported and bool(op_uses)),
        )

    @staticmethod
    def _numpy_bindings_for_forward(tree, forward):
        return numpy_bindings_for_forward(tree, forward)

    def _apply_profile_capabilities(self, source, op_uses, profiles):
        """Promote only exact missing top-level APIs with exact source profiles."""

        if self.profile_resolver is None or profiles is None:
            return
        source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if profiles.source_sha256 != source_sha256:
            return
        candidates = []
        for op in op_uses:
            if (
                not op.supported
                and op.reason == _MISSING_API_REASON
                and op.qualified_name == "numpy." + op.name
            ):
                candidates.append(op)
        if not candidates:
            return
        session_factory = getattr(
            self.profile_resolver,
            "prepare_analysis_profile_resolver",
            None,
        )
        resolver_owner = (
            session_factory(
                tuple(sorted({op.qualified_name for op in candidates}))
            )
            if callable(session_factory)
            else self.profile_resolver
        )
        resolver = getattr(
            resolver_owner,
            "resolve_profile_capability",
            None,
        )
        if not callable(resolver):
            raise TypeError(
                "analysis profile resolver session is invalid"
            )
        for op in candidates:
            if op.supported:
                continue
            profile = profiles.find(
                op.qualified_name,
                op.lineno,
                op.col_offset,
                op.positional_args,
                op.keyword_args,
            )
            if profile is None:
                continue
            match = resolver(op.qualified_name, profile)
            if not isinstance(match, CapabilityMatch):
                raise TypeError(
                    "profile resolver must return CapabilityMatch"
                )
            if match.matched:
                op.supported = True
                op.reason = None
