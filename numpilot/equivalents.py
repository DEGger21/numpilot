"""Deterministic lowering of documented NumPy-equivalent spellings.

The pinned AsNumpy runtime exposes some NumPy APIs under a different name, and
several NumPy APIs are documented one-line compositions of primitives that the
runtime does provide.  This module rewrites those spellings *before* static
analysis and migration, using only equivalences that are exact for real
floating-point inputs:

``np.abs(x)`` → ``np.absolute(x)``
``np.round(x, decimals=n)`` → ``np.around(x, decimals=n)``
``np.acos``/``asin``/``atan``/``acosh``/``asinh``/``atanh`` → arc* spellings
``np.var(x, axis=a, keepdims=k)`` → ``mean(square(x - mean(x, a, True)), a, k)``
``np.std(x, axis=a, keepdims=k)`` → ``sqrt(mean(square(x - mean(x, a, True)), a, k))``
``np.ptp(x, axis=a, keepdims=k)`` → ``max(x, a, k) - min(x, a, k)``
``np.linalg.vecdot(x, y, axis=a)`` → ``sum(multiply(x, y), axis=a)``
``np.isnan(x)`` → ``not_equal(x, x)``
``np.cbrt(x)`` → ``copysign(power(|x|, full_like(x, 1/3, x.dtype)), x)``

It also lowers Python arithmetic dunders on values that are statically proven
to be arrays (``a - b`` → ``np.subtract(a, b)``), because AsNumpy does not
promise ndarray dunder coverage.  Host-scalar arithmetic and control-flow
comparisons are left untouched.

Lowering is analysis-only.  Callers keep the original program as the NumPy
oracle and verify the migrated program against it.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from numpilot.agents.analysis import (
    numpy_bindings_for_forward,
    resolve_numpy_function,
    static_forward_parameter_kinds,
)
from numpilot.coverage import SUPPORTED_OPS

_ALIASES = {
    "abs": "absolute",
    "acos": "arccos",
    "acosh": "arccosh",
    "asin": "arcsin",
    "asinh": "arcsinh",
    "atan": "arctan",
    "atanh": "arctanh",
    "round": "around",
}

_BINOP_CALLS = {
    ast.Add: "add",
    ast.Sub: "subtract",
    ast.Mult: "multiply",
    ast.Div: "divide",
    ast.FloorDiv: "floor_divide",
    ast.Mod: "remainder",
    ast.Pow: "power",
    ast.MatMult: "matmul",
}

# The audited AsNumpy elementwise bindings require two arrays.  A Python
# scalar is therefore lowered to a same-shape, same-dtype array first; this
# preserves NumPy broadcast semantics on the pinned runtime instead of
# depending on an unaudited weak-scalar binding.
_BROADCAST_BINARY_OPS = frozenset({
    "add", "arctan2", "copysign", "divide", "divmod", "float_power",
    "floor_divide", "fmax", "fmin", "fmod", "gcd", "heaviside", "hypot",
    "lcm", "logaddexp", "logaddexp2", "logical_and", "logical_or",
    "logical_xor", "maximum", "minimum", "mod", "multiply", "remainder",
    "subtract", "true_divide",
})
_LOGICAL_BINARY_OPS = frozenset({
    "logical_and", "logical_or", "logical_xor",
})
_NUMPY_REAL_CONSTANTS = frozenset({"e", "euler_gamma", "inf", "nan", "pi"})
_FULL_LIKE_TEMPLATE = "__np__.full(_base.shape, _value, dtype=_base.dtype)"

_SCALAR_REDUCTIONS = frozenset({
    "all", "amax", "amin", "any", "max", "mean", "min", "nanmax", "nanmean",
    "nanmin", "nanprod", "nansum", "prod", "sum", "var", "std", "ptp",
})

_ARRAY_CONSTRUCTORS = frozenset({
    "array", "asarray", "empty", "empty_like", "eye", "full", "full_like",
    "identity", "linspace", "logspace", "geomspace", "ones", "ones_like",
    "zeros", "zeros_like", "arange", "load", "copy", "reshape",
})

_RNG_ARRAY_METHODS = frozenset({
    "integers", "normal", "random", "random_sample", "standard_normal",
    "uniform", "randint",
})

_FORBIDDEN_VAR_KEYWORDS = frozenset({
    "correction", "ddof", "dtype", "mean", "out", "where",
})
_FORBIDDEN_PTP_KEYWORDS = frozenset({"out"})
_FORBIDDEN_VECDOT_KEYWORDS = frozenset()

_VAR_TEMPLATE = (
    "__np__.mean(__np__.square(__np__.subtract("
    "_x, __np__.mean(_x, axis=_axis, keepdims=True))), "
    "axis=_axis, keepdims=_keepdims)"
)
_STD_TEMPLATE = (
    "__np__.sqrt(__np__.mean(__np__.square(__np__.subtract("
    "_x, __np__.mean(_x, axis=_axis, keepdims=True))), "
    "axis=_axis, keepdims=_keepdims))"
)
_PTP_TEMPLATE = (
    "__np__.subtract(__np__.max(_x, axis=_axis, keepdims=_keepdims), "
    "__np__.min(_x, axis=_axis, keepdims=_keepdims))"
)
_VECDOT_TEMPLATE = "__np__.sum(__np__.multiply(_x, _y), axis=_axis)"


@dataclass(frozen=True)
class EquivalentRewrite:
    """One deterministic, source-level equivalence rewrite."""

    kind: str
    line: int
    original: str
    replacement: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "line": self.line,
            "original": self.original,
            "replacement": self.replacement,
        }


@dataclass(frozen=True)
class EquivalentLowering:
    """Result of lowering a program for analysis/migration."""

    source: str
    rewrites: tuple[EquivalentRewrite, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def applied(self) -> bool:
        return bool(self.rewrites)

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "rewrites": [item.to_dict() for item in self.rewrites],
            "notes": list(self.notes),
        }


def lower_equivalents(source: str) -> EquivalentLowering:
    """Return a source with documented equivalent spellings lowered.

    Syntax errors and programs without a top-level ``forward`` are returned
    unchanged; the ordinary analyzer/verifier remains responsible for
    rejecting them.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return EquivalentLowering(source, (), ("syntax_error",))

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    forward = functions.get("forward")
    if forward is None:
        return EquivalentLowering(source, (), ("missing_forward",))

    build_kinds = static_forward_parameter_kinds(tree, forward) or {}
    arguments = _forward_arguments(forward)
    forward_scope = {
        argument: build_kinds.get(argument, "unknown") for argument in arguments
    }

    rewrites: list[EquivalentRewrite] = []
    for function in _reachable_functions(tree, functions, forward):
        bindings = numpy_bindings_for_forward(tree, function)
        if function is forward:
            scope = dict(forward_scope)
        else:
            scope = {argument: "unknown" for argument in _forward_arguments(function)}
        lowerer = _FunctionLowerer(bindings, scope)
        function.body = [lowerer.visit(statement) for statement in function.body]
        rewrites.extend(lowerer.rewrites)

    if not rewrites:
        return EquivalentLowering(source)

    ast.fix_missing_locations(tree)
    return EquivalentLowering(ast.unparse(tree) + "\n", tuple(rewrites))


def _forward_arguments(function: ast.FunctionDef) -> list[str]:
    return [
        argument.arg
        for argument in (
            list(function.args.posonlyargs)
            + list(function.args.args)
            + list(function.args.kwonlyargs)
        )
    ]


def _reachable_functions(tree, functions, forward):
    """Return ``forward`` plus every top-level helper it reaches, in order."""

    ordered = [forward]
    seen = {forward.name}
    index = 0
    while index < len(ordered):
        function = ordered[index]
        index += 1
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            target = functions.get(node.func.id)
            if target is None or target.name in seen or target.name == "build_inputs":
                continue
            seen.add(target.name)
            ordered.append(target)
    return ordered


class _FunctionLowerer(ast.NodeTransformer):
    """Lower one function body while tracking array/scalar names in program order."""

    def __init__(self, bindings, scope):
        self.bindings = dict(bindings)
        self.scope = dict(scope)
        self.np_alias = next(
            (name for name, qualified in self.bindings.items() if qualified == "numpy"),
            None,
        )
        self.rewrites: list[EquivalentRewrite] = []

    # ------------------------------------------------------------- statements
    def visit_FunctionDef(self, node):  # nested scopes are not part of the island
        return node

    def visit_AsyncFunctionDef(self, node):
        return node

    def visit_ClassDef(self, node):
        return node

    def visit_Lambda(self, node):
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        kind = self._kind(node.value)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.scope[node.targets[0].id] = kind
        else:
            for target in node.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                        self.scope[name.id] = "unknown"
        return node

    def visit_AnnAssign(self, node):
        node = self.generic_visit(node)
        if node.value is not None and isinstance(node.target, ast.Name):
            self.scope[node.target.id] = self._kind(node.value)
        return node

    def visit_For(self, node):
        node.iter = self.visit(node.iter)
        if isinstance(node.target, ast.Name):
            node.target = self.visit(node.target)
            self.scope[node.target.id] = (
                "scalar" if _is_range_call(node.iter) else "unknown"
            )
        node.body = [self.visit(statement) for statement in node.body]
        node.orelse = [self.visit(statement) for statement in node.orelse]
        return node

    # ------------------------------------------------------------ expressions
    def visit_Call(self, node):
        original = ast.unparse(node)
        node = self.generic_visit(node)
        resolved = resolve_numpy_function(node.func, self.bindings)
        if resolved is None or self.np_alias is None:
            return node
        name, qualified = resolved
        replacement = None
        kind = None
        if qualified == "numpy." + name and name in _ALIASES:
            replacement = ast.Call(
                func=self._attr(_ALIASES[name]),
                args=node.args,
                keywords=node.keywords,
            )
            kind = "alias"
        elif (
            qualified == "numpy.asarray"
            and len(node.args) == 1
            and not node.keywords
            and self._kind(node.args[0]) == "array"
        ):
            # ``asarray`` is an identity on an already-proven device array.
            replacement = node.args[0]
            kind = "alias"
        elif qualified in {"numpy.all", "numpy.any"}:
            replacement = self._lower_boolean_axis(node)
            kind = "axis"
        elif qualified == "numpy.isnan":
            replacement = self._lower_isnan(node)
            kind = "composition"
        elif qualified == "numpy.cbrt":
            replacement = self._lower_cbrt(node)
            kind = "composition"
        elif qualified == "numpy.var":
            replacement = self._lower_variance(node)
            kind = "composition"
        elif qualified == "numpy.std":
            replacement = self._lower_std(node)
            kind = "composition"
        elif qualified == "numpy.ptp":
            replacement = self._lower_ptp(node)
            kind = "composition"
        elif qualified == "numpy.linalg.vecdot":
            replacement = self._lower_vecdot(node)
            kind = "composition"
        elif qualified == "numpy." + name and name in _BROADCAST_BINARY_OPS:
            replacement = self._lower_scalar_broadcast(node, name)
            kind = "broadcast"
        if replacement is None:
            return node
        ast.copy_location(replacement, node)
        self.rewrites.append(EquivalentRewrite(
            kind=kind,
            line=getattr(node, "lineno", 0),
            original=original,
            replacement=ast.unparse(replacement),
        ))
        return replacement

    def visit_BinOp(self, node):
        original = ast.unparse(node)
        node = self.generic_visit(node)
        call_name = _BINOP_CALLS.get(type(node.op))
        if call_name is None or self.np_alias is None:
            return node
        left_kind = self._kind(node.left)
        right_kind = self._kind(node.right)
        if "array" not in (left_kind, right_kind):
            return node
        if "nonarray" in (left_kind, right_kind):
            return node
        replacement = ast.Call(
            func=self._attr(call_name),
            args=[node.left, node.right],
            keywords=[],
        )
        ast.copy_location(replacement, node)
        self.rewrites.append(EquivalentRewrite(
            kind="dunder",
            line=getattr(node, "lineno", 0),
            original=original,
            replacement=ast.unparse(replacement),
        ))
        return replacement

    def visit_UnaryOp(self, node):
        original = ast.unparse(node)
        node = self.generic_visit(node)
        if self.np_alias is None or isinstance(node.operand, ast.Constant):
            return node
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            return node
        if self._kind(node.operand) != "array":
            return node
        if isinstance(node.op, ast.UAdd):
            return node.operand
        replacement = ast.Call(
            func=self._attr("negative"),
            args=[node.operand],
            keywords=[],
        )
        ast.copy_location(replacement, node)
        self.rewrites.append(EquivalentRewrite(
            kind="unary",
            line=getattr(node, "lineno", 0),
            original=original,
            replacement=ast.unparse(replacement),
        ))
        return replacement

    # ----------------------------------------------------------- compositions
    def _lower_variance(self, node):
        x, axis, keepdims = self._composition_arguments(
            node, _FORBIDDEN_VAR_KEYWORDS
        )
        if x is None or axis is None:
            return None
        return self._render(
            _VAR_TEMPLATE, {"_x": x, "_axis": axis, "_keepdims": keepdims}
        )

    def _lower_std(self, node):
        x, axis, keepdims = self._composition_arguments(
            node, _FORBIDDEN_VAR_KEYWORDS
        )
        if x is None or axis is None:
            return None
        return self._render(
            _STD_TEMPLATE, {"_x": x, "_axis": axis, "_keepdims": keepdims}
        )

    def _lower_ptp(self, node):
        x, axis, keepdims = self._composition_arguments(
            node, _FORBIDDEN_PTP_KEYWORDS
        )
        if x is None or axis is None:
            return None
        return self._render(
            _PTP_TEMPLATE, {"_x": x, "_axis": axis, "_keepdims": keepdims}
        )

    def _lower_boolean_axis(self, node):
        """Express ``axis=i`` in the single-element tuple form AsNumpy audits."""

        if not node.args:
            return None
        if len(node.args) > 2:
            return None
        axis = None
        if len(node.args) > 1:
            axis = node.args[1]
        for keyword in node.keywords:
            if keyword.arg == "axis":
                if axis is not None:
                    return None
                axis = keyword.value
            elif keyword.arg not in {"keepdims", "out", "where"}:
                return None
        if axis is None or isinstance(axis, ast.Tuple):
            return None
        if not _is_static_integer_node(axis):
            return None
        replacement = ast.Call(
            func=node.func,
            args=[
                node.args[0],
                ast.Tuple(elts=[axis], ctx=ast.Load()),
            ],
            keywords=[
                ast.keyword(arg=keyword.arg, value=keyword.value)
                for keyword in node.keywords
                if keyword.arg != "axis"
            ],
        )
        return replacement

    def _lower_isnan(self, node):
        """``isnan(x)`` is exactly ``not_equal(x, x)`` for real floats."""

        if len(node.args) != 1 or node.keywords:
            return None
        if self._kind(node.args[0]) != "array":
            return None
        x = node.args[0]
        return ast.Call(
            func=self._attr("not_equal"),
            args=[x, x],
            keywords=[],
        )

    def _lower_cbrt(self, node):
        """``cbrt(x)`` as ``copysign(|x| ** (1/3), x)``.

        The exponent is materialized as a same-shape array through
        ``full_like`` so both ``power`` operands stay arrays on the pinned
        runtime.  ``copysign`` restores the sign of negative inputs and of
        negative zero, which a plain fractional power cannot express.
        """

        if len(node.args) != 1 or node.keywords:
            return None
        x = node.args[0]
        if self._kind(x) != "array":
            return None
        exponent = ast.Call(
            func=self._attr("full_like"),
            args=[x, ast.Constant(value=1.0 / 3.0)],
            keywords=[
                ast.keyword(
                    arg="dtype",
                    value=ast.Attribute(
                        value=x, attr="dtype", ctx=ast.Load()
                    ),
                )
            ],
        )
        magnitude = ast.Call(
            func=self._attr("power"),
            args=[
                ast.Call(func=self._attr("absolute"), args=[x], keywords=[]),
                exponent,
            ],
            keywords=[],
        )
        return ast.Call(
            func=self._attr("copysign"),
            args=[magnitude, x],
            keywords=[],
        )

    def _lower_scalar_broadcast(self, node, name):
        """Rewrite ``ufunc(array, scalar)`` to two array operands."""

        if len(node.args) != 2 or node.keywords:
            return None
        left, right = node.args
        left_kind = self._kind(left)
        right_kind = self._kind(right)
        if left_kind == "array" and right_kind == "scalar":
            base, scalar = left, right
            scalar_position = 1
        elif left_kind == "scalar" and right_kind == "array":
            base, scalar = right, left
            scalar_position = 0
        else:
            return None
        if not isinstance(base, (ast.Name, ast.Subscript)):
            return None
        if isinstance(scalar, ast.Constant) and isinstance(scalar.value, bool):
            if name not in _LOGICAL_BINARY_OPS:
                return None
            scalar = ast.Constant(1 if scalar.value else 0)
        if not self._is_static_real_scalar(scalar):
            return None
        full = self._render(
            _FULL_LIKE_TEMPLATE, {"_base": base, "_value": scalar}
        )
        args = [left, full] if scalar_position == 1 else [full, right]
        return ast.Call(func=node.func, args=args, keywords=[])

    def _is_static_real_scalar(self, node):
        if isinstance(node, ast.Constant):
            return (
                isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
            )
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            return self._is_static_real_scalar(node.operand)
        if isinstance(node, ast.Attribute):
            resolved = resolve_numpy_function(node, self.bindings)
            return (
                resolved is not None
                and resolved[1].count(".") == 1
                and resolved[0] in _NUMPY_REAL_CONSTANTS
            )
        return False

    def _lower_vecdot(self, node):
        if len(node.args) < 2:
            return None
        if len(node.args) > 2:
            return None
        if any(
            keyword.arg is None or keyword.arg in _FORBIDDEN_VECDOT_KEYWORDS
            for keyword in node.keywords
        ):
            return None
        unexpected = [
            keyword.arg for keyword in node.keywords
            if keyword.arg not in {"axis"}
        ]
        if unexpected:
            return None
        axis = _keyword(node, "axis")
        if axis is None:
            axis = ast.Constant(-1)
        return self._render(
            _VECDOT_TEMPLATE,
            {"_x": node.args[0], "_y": node.args[1], "_axis": axis},
        )

    @staticmethod
    def _composition_arguments(node, forbidden_keywords):
        """Return ``(value, axis, keepdims)`` or ``(None, ...)`` when unsafe."""

        if not node.args:
            return None, None, None
        if len(node.args) > 2:
            return None, None, None
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg in forbidden_keywords:
                return None, None, None
            if keyword.arg not in {"axis", "keepdims"}:
                return None, None, None
        value = node.args[0]
        axis = node.args[1] if len(node.args) > 1 else _keyword(node, "axis")
        if axis is None:
            return None, None, None
        keepdims = _keyword(node, "keepdims")
        if keepdims is None:
            keepdims = ast.Constant(False)
        return value, axis, keepdims

    def _render(self, template, mapping):
        expression = ast.parse(template, mode="eval").body

        class _Substitute(ast.NodeTransformer):
            def visit_Name(self, inner):
                if inner.id == "__np__":
                    return ast.copy_location(
                        ast.Name(id=self_parent.np_alias, ctx=ast.Load()), inner
                    )
                if inner.id in mapping:
                    return ast.copy_location(
                        copy.deepcopy(mapping[inner.id]), inner
                    )
                return inner

        self_parent = self
        return _Substitute().visit(expression)

    def _attr(self, name):
        return ast.Attribute(
            value=ast.Name(id=self.np_alias, ctx=ast.Load()),
            attr=name,
            ctx=ast.Load(),
        )

    # ----------------------------------------------------------------- typing
    def _kind(self, node):
        return _expression_kind(node, self.bindings, self.scope)


def _is_static_integer_node(node):
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    )


def _keyword(node, name):
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _is_range_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
    )


def _expression_kind(node, bindings, scope):
    """Conservatively classify an expression as array/scalar/nonarray/unknown."""

    if isinstance(node, ast.Name):
        return scope.get(node.id, "unknown")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (bool, int, float, complex)):
            return "scalar"
        return "nonarray"
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, (ast.UAdd, ast.USub)):
            kind = _expression_kind(node.operand, bindings, scope)
            return "scalar" if kind == "scalar" else kind
        return "nonarray"
    if isinstance(node, ast.BinOp):
        left = _expression_kind(node.left, bindings, scope)
        right = _expression_kind(node.right, bindings, scope)
        if "array" in (left, right):
            return "array"
        if left == right == "scalar":
            return "scalar"
        return "unknown"
    if isinstance(node, (ast.BoolOp, ast.Compare)):
        return "scalar"
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Attribute) and node.value.attr in {
            "shape", "strides",
        }:
            return "scalar"
        if _expression_kind(node.value, bindings, scope) == "array":
            return "array"
        return "unknown"
    if isinstance(node, ast.Attribute):
        if node.attr in {"shape", "strides"}:
            return "nonarray"
        if node.attr in {"size", "ndim", "itemsize", "nbytes", "dtype"}:
            return "scalar"
        resolved = resolve_numpy_function(node, bindings)
        if (
            resolved is not None
            and resolved[1].count(".") == 1
            and resolved[0] in _NUMPY_REAL_CONSTANTS
        ):
            return "scalar"
        return "unknown"
    if isinstance(node, ast.IfExp):
        body = _expression_kind(node.body, bindings, scope)
        orelse = _expression_kind(node.orelse, bindings, scope)
        if "array" in (body, orelse):
            return "array"
        return body if body == orelse else "unknown"
    if isinstance(node, ast.Call):
        resolved = resolve_numpy_function(node.func, bindings)
        if resolved is not None:
            name, qualified = resolved
            if qualified == "numpy.random." + name or qualified.startswith("numpy.random."):
                return _random_kind(name, node)
            if qualified == "numpy." + name:
                if name in _ALIASES:
                    return "array"
                if name in _ARRAY_CONSTRUCTORS or name in SUPPORTED_OPS:
                    if name in _SCALAR_REDUCTIONS and not _has_axis_or_keepdims(node):
                        return "scalar"
                    return "array"
            return "unknown"
        if isinstance(node.func, ast.Name):
            if node.func.id == "range":
                return "scalar"
            if node.func.id == "len":
                return "scalar"
            if node.func.id in {"list", "tuple", "dict", "set"}:
                return "nonarray"
            return "unknown"
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in _RNG_ARRAY_METHODS:
                return "array" if _has_random_size(node) else "scalar"
            if node.func.attr in {"item", "tolist", "to_numpy", "asnumpy"}:
                return "nonarray"
            return "unknown"
        return "unknown"
    if isinstance(node, (ast.Tuple, ast.List, ast.Dict, ast.Set)):
        return "nonarray"
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return "nonarray"
    return "unknown"


def _random_kind(name, node):
    if name in {"rand", "randn"}:
        return "array" if node.args else "scalar"
    if name in {"random", "random_sample", "standard_normal"}:
        return "array" if _has_random_size(node) else "scalar"
    if name in {"normal", "uniform"}:
        return "array" if _has_random_size(node) else "scalar"
    return "unknown"


def _has_random_size(node):
    for keyword in node.keywords:
        if keyword.arg != "size":
            continue
        return not (
            isinstance(keyword.value, ast.Constant) and keyword.value.value is None
        )
    return bool(node.args)


def _has_axis_or_keepdims(node):
    if len(node.args) > 1:
        first = node.args[1]
        if not (isinstance(first, ast.Constant) and first.value is None):
            return True
    for keyword in node.keywords:
        if keyword.arg == "keepdims":
            if not (
                isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
            ):
                return True
        if keyword.arg == "axis":
            if not (
                isinstance(keyword.value, ast.Constant)
                and keyword.value.value is None
            ):
                return True
    return False
