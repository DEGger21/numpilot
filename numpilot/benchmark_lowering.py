"""Whole-program lowering for the ``build_inputs`` / ``forward`` benchmark contract.

The ordinary AST front end analyses only the body of ``forward`` and rejects
calls to helpers and any control flow.  The benchmark programs deliberately use
both: they factor compute into small helpers and drive fixed-count loops with
host scalars.  This module adapts that documented contract without weakening
the strict-NPU audit:

* every reachable top-level helper is inlined into ``forward``;
* ``for`` over a static ``range`` and ``while`` / ``if`` whose conditions are
  host scalars are partially evaluated into straight-line code;
* tuple returns and destructuring assignments are propagated by value;
* every remaining NumPy call, array dunder and scalar broadcast is left for
  :mod:`numpilot.equivalents`, which owns the audited spelling changes.

The transformation is fail-closed.  If a program uses an array-valued branch,
dynamic indexing, mutation with observable aliasing, or any other construct the
partial evaluator cannot prove, the original source is returned unchanged and
the normal analyzer reports the unsupported construct.  ``build_inputs`` is
never inlined: it stays on the host and remains the input constructor.
"""
from __future__ import annotations

import ast
import copy
import operator
from dataclasses import dataclass
from typing import Any

from numpilot.agents.analysis import (
    numpy_bindings_for_forward,
    resolve_numpy_function,
)
from numpilot.equivalents import EquivalentRewrite, lower_equivalents


_MAX_UNROLL = 1024
_MAX_RECURSION = 32
_NUMPY_REAL_CONSTANTS = frozenset({
    "e", "euler_gamma", "inf", "nan", "pi",
})
_NUMPY_DTYPE_NAMES = frozenset({
    "bool", "bool_", "byte", "cdouble", "complex64", "complex128",
    "csingle", "double", "float16", "float32", "float64", "half", "int8",
    "int16", "int32", "int64", "int_", "intc", "intp", "longlong", "short",
    "single", "ubyte", "uint", "uint8", "uint16", "uint32", "uint64",
    "uintc", "uintp", "ulonglong", "ushort",
})
_ARRAY_METADATA = frozenset({
    "dtype", "itemsize", "nbytes", "ndim", "shape", "size", "strides",
})
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
}
_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}


class _CannotLower(Exception):
    """The partial evaluator cannot prove this program's host control flow."""


@dataclass(frozen=True)
class _Scalar:
    value: Any


@dataclass(frozen=True)
class _Array:
    expr: ast.expr


@dataclass(frozen=True)
class _Sequence:
    items: tuple


@dataclass(frozen=True)
class _Mapping:
    items: tuple


@dataclass(frozen=True)
class _Raw:
    """A non-array expression such as a dtype or an axis tuple."""

    expr: ast.expr


@dataclass(frozen=True)
class BenchmarkLowering:
    """Result of benchmark-contract lowering."""

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


def lower_benchmark_program(source: str) -> BenchmarkLowering:
    """Adapt one benchmark program for analysis and migration.

    The returned source is only an analysis/migration surface.  Callers must
    keep the original source as the NumPy verification oracle.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return BenchmarkLowering(source, (), ("syntax_error",))

    try:
        inlined, rewrites = _PartialEvaluator(tree).run()
    except _CannotLower as exc:
        return BenchmarkLowering(
            source,
            (),
            ("whole_program_lowering_skipped: " + str(exc),),
        )
    except Exception as exc:  # a malformed case must not abort the sweep
        return BenchmarkLowering(
            source,
            (),
            (
                "whole_program_lowering_skipped: %s: %s"
                % (type(exc).__name__, str(exc)),
            ),
        )

    lowered = lower_equivalents(inlined)
    return BenchmarkLowering(
        lowered.source,
        tuple(rewrites) + tuple(lowered.rewrites),
        tuple(lowered.notes),
    )


class _ReturnSignal(Exception):
    def __init__(self, value):
        super().__init__("return")
        self.value = value


class _PartialEvaluator:
    def __init__(self, tree: ast.Module):
        self.tree = tree
        self.functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name != "build_inputs"
        }
        self.forward = self.functions.get("forward")
        self.module_constants = _module_constants(tree)
        self.used_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        self.used_names.update(
            node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
        )
        self.counter = 0
        self.out: list[ast.stmt] = []
        self.env: dict[str, Any] = {}
        self.bindings: dict[str, str] = {}
        self.rewrites: list[EquivalentRewrite] = []
        self.depth = 0

    # ------------------------------------------------------------- entry point
    def run(self):
        if self.forward is None:
            raise _CannotLower("missing forward")
        self.bindings = numpy_bindings_for_forward(self.tree, self.forward)
        self.env = {
            argument.arg: _Array(ast.Name(id=argument.arg, ctx=ast.Load()))
            for argument in _arguments(self.forward)
        }
        try:
            self._exec_block(self.forward.body)
        except _ReturnSignal:
            pass
        if not any(isinstance(statement, ast.Return) for statement in self.out):
            raise _CannotLower("forward has no reachable return")

        new_body = []
        if (
            self.forward.body
            and isinstance(self.forward.body[0], ast.Expr)
            and isinstance(self.forward.body[0].value, ast.Constant)
            and isinstance(self.forward.body[0].value.value, str)
        ):
            new_body.append(self.forward.body[0])
        new_body.extend(self.out)
        self.forward.body = new_body
        ast.fix_missing_locations(self.tree)
        return ast.unparse(self.tree) + "\n", tuple(self.rewrites)

    # -------------------------------------------------------------- statements
    def _exec_block(self, statements):
        for statement in statements:
            self._exec_statement(statement)

    def _exec_statement(self, node):
        if isinstance(node, ast.Assign):
            value = self._eval(node.value)
            for target in node.targets:
                self._bind(target, value)
            return
        if isinstance(node, ast.AnnAssign):
            if node.value is None:
                raise _CannotLower("annotation without value")
            self._bind(node.target, self._eval(node.value))
            return
        if isinstance(node, ast.AugAssign):
            self._exec_augassign(node)
            return
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ):
                return
            if isinstance(node.value, ast.Call):
                self._exec_discarded_call(node.value)
                return
            raise _CannotLower("bare expression")
        if isinstance(node, ast.Return):
            value = self._eval(node.value) if node.value is not None else None
            if value is None:
                raise _CannotLower("empty return")
            if self.depth:
                raise _ReturnSignal(value)
            self.out.append(ast.Return(value=self._to_expr(value)))
            return
        if isinstance(node, ast.If):
            test = self._eval(node.test)
            truth = self._scalar_truth(test)
            branch = node.body if truth else node.orelse
            self.rewrites.append(EquivalentRewrite(
                kind="static_if",
                line=getattr(node, "lineno", 0),
                original=ast.unparse(node.test),
                replacement="then" if truth else "else",
            ))
            self._exec_block(branch)
            return
        if isinstance(node, ast.For):
            self._exec_for(node)
            return
        if isinstance(node, ast.While):
            self._exec_while(node)
            return
        if isinstance(node, ast.Pass):
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise _CannotLower("import inside compute scope")
        raise _CannotLower(
            "unsupported statement %s" % type(node).__name__
        )

    def _exec_discarded_call(self, call):
        if isinstance(call.func, ast.Name) and call.func.id in self.functions:
            self._eval_helper_call(call.func.id, call)
            return
        resolved = self._resolve_numpy(call.func)
        if resolved is not None:
            # Pure NumPy / AsNumpy calls have no observable side effect when
            # their result is discarded.  The strict surface rejects them
            # later, so failing closed here would only hide the real reason.
            raise _CannotLower("discarded NumPy call")
        raise _CannotLower("discarded unknown call")

    def _exec_augassign(self, node):
        if not isinstance(node.target, ast.Name):
            raise _CannotLower("augmented assignment target")
        current = self.env.get(node.target.id)
        if current is None:
            current = self._eval(node.target)
        right = self._eval(node.value)
        left_scalar = isinstance(current, _Scalar)
        right_scalar = isinstance(right, _Scalar)
        if not (left_scalar and right_scalar):
            # Array augmentation is observable through aliasing and write-back
            # semantics.  Rebinding the name to a fresh array would silently
            # drop those semantics, so leave the statement untouched and let
            # the strict analyzer report ``AugAssign`` as unsupported.
            raise _CannotLower("augmented assignment on array")
        function = _BIN_OPS.get(type(node.op))
        if function is None:
            raise _CannotLower("augmented operator")
        self.env[node.target.id] = _Scalar(
            function(current.value, right.value)
        )

    def _exec_for(self, node):
        if not isinstance(node.target, ast.Name):
            raise _CannotLower("for target")
        if node.orelse:
            raise _CannotLower("for-else")
        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            raise _CannotLower("non-range for")
        arguments = [self._eval(argument) for argument in node.iter.args]
        if not all(isinstance(item, _Scalar) for item in arguments):
            raise _CannotLower("non-scalar range argument")
        values = [item.value for item in arguments]
        try:
            indices = list(range(*values))
        except Exception as exc:
            raise _CannotLower("invalid range: %s" % exc) from exc
        if len(indices) > _MAX_UNROLL:
            raise _CannotLower("range exceeds unroll budget")
        self.rewrites.append(EquivalentRewrite(
            kind="unroll_for",
            line=getattr(node, "lineno", 0),
            original=ast.unparse(node.iter),
            replacement="unrolled %d iterations" % len(indices),
        ))
        for index in indices:
            self.env[node.target.id] = _Scalar(index)
            self._exec_block(node.body)

    def _exec_while(self, node):
        if node.orelse:
            raise _CannotLower("while-else")
        self.rewrites.append(EquivalentRewrite(
            kind="unroll_while",
            line=getattr(node, "lineno", 0),
            original=ast.unparse(node.test),
            replacement="static while",
        ))
        for _ in range(_MAX_UNROLL):
            truth = self._scalar_truth(self._eval(node.test))
            if not truth:
                return
            self._exec_block(node.body)
        raise _CannotLower("while exceeds unroll budget")

    # --------------------------------------------------------------- bindings
    def _bind(self, target, value):
        if isinstance(target, ast.Name):
            self.env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, _Sequence):
                raise _CannotLower("destructuring a non-sequence value")
            if len(target.elts) != len(value.items):
                raise _CannotLower("destructuring arity mismatch")
            for item, item_value in zip(target.elts, value.items):
                self._bind(item, item_value)
            return
        raise _CannotLower("assignment target %s" % type(target).__name__)

    # ------------------------------------------------------------ expressions
    def _eval(self, node):
        if isinstance(node, ast.Constant):
            return _Scalar(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            if node.id in self.module_constants:
                return _Scalar(self.module_constants[node.id])
            raise _CannotLower("unknown name %s" % node.id)
        if isinstance(node, ast.Tuple):
            return _Sequence(tuple(self._eval(item) for item in node.elts))
        if isinstance(node, ast.List):
            return _Sequence(tuple(self._eval(item) for item in node.elts))
        if isinstance(node, ast.Dict):
            pairs = []
            for key, value in zip(node.keys, node.values):
                if key is None:
                    raise _CannotLower("dict unpacking")
                evaluated_key = self._eval(key)
                if not (
                    isinstance(evaluated_key, _Scalar)
                    and isinstance(evaluated_key.value, str)
                ):
                    raise _CannotLower("non-string output key")
                pairs.append((evaluated_key.value, self._eval(value)))
            return _Mapping(tuple(pairs))
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._eval_unary(node)
        if isinstance(node, ast.BoolOp):
            return self._eval_boolop(node)
        if isinstance(node, ast.Compare):
            return self._eval_compare(node)
        if isinstance(node, ast.IfExp):
            truth = self._scalar_truth(self._eval(node.test))
            return self._eval(node.body if truth else node.orelse)
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        if isinstance(node, ast.Attribute):
            return self._eval_attribute(node)
        if isinstance(node, ast.NamedExpr):
            value = self._eval(node.value)
            self._bind(node.target, value)
            return value
        raise _CannotLower("expression %s" % type(node).__name__)

    def _eval_binop(self, node):
        left = self._eval(node.left)
        right = self._eval(node.right)
        if isinstance(left, _Scalar) and isinstance(right, _Scalar):
            function = _BIN_OPS.get(type(node.op))
            if function is None:
                raise _CannotLower("binary operator")
            try:
                return _Scalar(function(left.value, right.value))
            except Exception as exc:
                raise _CannotLower("scalar arithmetic: %s" % exc) from exc
        temporary = self._fresh("v")
        self.out.append(ast.Assign(
            targets=[ast.Name(id=temporary, ctx=ast.Store())],
            value=ast.BinOp(
                left=self._to_expr(left),
                op=node.op,
                right=self._to_expr(right),
            ),
        ))
        return _Array(ast.Name(id=temporary, ctx=ast.Load()))

    def _eval_unary(self, node):
        operand = self._eval(node.operand)
        if isinstance(operand, _Scalar):
            function = _UNARY_OPS.get(type(node.op))
            if function is None:
                raise _CannotLower("unary operator")
            return _Scalar(function(operand.value))
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, (ast.USub, ast.Invert)):
            temporary = self._fresh("v")
            self.out.append(ast.Assign(
                targets=[ast.Name(id=temporary, ctx=ast.Store())],
                value=ast.UnaryOp(
                    op=node.op, operand=self._to_expr(operand)
                ),
            ))
            return _Array(ast.Name(id=temporary, ctx=ast.Load()))
        raise _CannotLower("array unary operator")

    def _eval_boolop(self, node):
        if isinstance(node.op, ast.And):
            result = None
            for value_node in node.values:
                result = self._eval(value_node)
                if not self._scalar_truth(result):
                    return _Scalar(False)
            return result
        if isinstance(node.op, ast.Or):
            result = None
            for value_node in node.values:
                result = self._eval(value_node)
                if self._scalar_truth(result):
                    return result
            return result
        raise _CannotLower("boolean operator")

    def _eval_compare(self, node):
        left = self._eval(node.left)
        for operation, comparator in zip(node.ops, node.comparators):
            right = self._eval(comparator)
            if not (
                isinstance(left, _Scalar) and isinstance(right, _Scalar)
            ):
                raise _CannotLower("array-valued comparison")
            function = _COMPARE_OPS.get(type(operation))
            if function is None:
                raise _CannotLower("comparison operator")
            if not function(left.value, right.value):
                return _Scalar(False)
            left = right
        return _Scalar(True)

    def _eval_call(self, node):
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self.functions and name != "forward":
                return self._eval_helper_call(name, node)
            if name == "range":
                raise _CannotLower("range outside for")
            if name in {"len", "abs", "int", "float", "bool", "min", "max"}:
                return self._eval_builtin(name, node)
        resolved = self._resolve_numpy(node.func)
        if resolved is None:
            raise _CannotLower(
                "unknown call %s" % ast.unparse(node.func)
            )
        arguments = []
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                raise _CannotLower("starred call argument")
            arguments.append(self._to_expr(self._eval(argument)))
        keywords = []
        for keyword in node.keywords:
            if keyword.arg is None:
                raise _CannotLower("keyword unpacking")
            keywords.append(ast.keyword(
                arg=keyword.arg,
                value=self._to_expr(self._eval(keyword.value)),
            ))
        temporary = self._fresh("v")
        self.out.append(ast.Assign(
            targets=[ast.Name(id=temporary, ctx=ast.Store())],
            value=ast.Call(
                func=copy.deepcopy(node.func),
                args=arguments,
                keywords=keywords,
            ),
        ))
        return _Array(ast.Name(id=temporary, ctx=ast.Load()))

    def _eval_builtin(self, name, node):
        values = [self._eval(argument) for argument in node.args]
        if any(not isinstance(item, _Scalar) for item in values):
            raise _CannotLower("non-scalar builtin %s" % name)
        raw = [item.value for item in values]
        try:
            if name == "len":
                return _Scalar(len(raw[0]))
            if name == "abs":
                return _Scalar(abs(raw[0]))
            if name == "int":
                return _Scalar(int(raw[0]))
            if name == "float":
                return _Scalar(float(raw[0]))
            if name == "bool":
                return _Scalar(bool(raw[0]))
            if name == "min":
                return _Scalar(min(raw))
            if name == "max":
                return _Scalar(max(raw))
        except Exception as exc:
            raise _CannotLower("builtin %s: %s" % (name, exc)) from exc
        raise _CannotLower("builtin %s" % name)

    def _eval_attribute(self, node):
        resolved = self._resolve_numpy(node)
        if resolved is not None:
            _, qualified = resolved
            name = qualified.rsplit(".", 1)[-1]
            if qualified.count(".") == 1 and name in _NUMPY_REAL_CONSTANTS:
                return _Scalar(getattr(_numpy_module(), name))
            if name in _NUMPY_DTYPE_NAMES:
                return _Raw(copy.deepcopy(node))
            if name == "linalg":
                return _Raw(copy.deepcopy(node))
            raise _CannotLower("NumPy attribute %s" % qualified)
        if isinstance(node.value, _Array) or self._is_array_expr(node.value):
            if node.attr in _ARRAY_METADATA:
                return _Raw(ast.Attribute(
                    value=self._to_expr(self._eval(node.value)),
                    attr=node.attr,
                    ctx=ast.Load(),
                ))
        raise _CannotLower("attribute %s" % ast.unparse(node))

    def _is_array_expr(self, node):
        if isinstance(node, ast.Name):
            return isinstance(self.env.get(node.id), _Array)
        return False

    def _eval_helper_call(self, name, node):
        function = self.functions.get(name)
        if function is None:
            raise _CannotLower("unknown helper %s" % name)
        if self.depth >= _MAX_RECURSION:
            raise _CannotLower("helper recursion budget")
        arguments = [self._eval(argument) for argument in node.args]
        keywords = {
            keyword.arg: self._eval(keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        if len(keywords) != len(node.keywords):
            raise _CannotLower("keyword unpacking")
        parameters = [argument.arg for argument in _arguments(function)]
        if len(arguments) > len(parameters):
            raise _CannotLower("too many helper arguments")
        bound = dict(zip(parameters, arguments))
        for parameter, value in keywords.items():
            if parameter not in parameters:
                raise _CannotLower("unknown helper keyword")
            if parameter in bound:
                raise _CannotLower("duplicate helper argument")
            bound[parameter] = value
        defaults = _default_values(function)
        for parameter in parameters:
            if parameter in bound:
                continue
            if parameter not in defaults:
                raise _CannotLower("missing helper argument")
            bound[parameter] = self._eval(defaults[parameter])
        self.rewrites.append(EquivalentRewrite(
            kind="inline_helper",
            line=getattr(node, "lineno", 0),
            original=ast.unparse(node),
            replacement="inlined %s" % name,
        ))
        previous_env = self.env
        previous_bindings = self.bindings
        self.env = bound
        self.bindings = numpy_bindings_for_forward(self.tree, function)
        self.depth += 1
        try:
            self._exec_block(function.body)
        except _ReturnSignal as signal:
            return signal.value
        finally:
            self.depth -= 1
            self.env = previous_env
            self.bindings = previous_bindings
        raise _CannotLower("helper %s does not return" % name)

    # ----------------------------------------------------------------- helpers
    def _resolve_numpy(self, node):
        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if not isinstance(root, ast.Name):
                return None
            qualified_root = self.bindings.get(root.id)
            if qualified_root != "numpy":
                return None
            parts = []
            cursor = node
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            parts.reverse()
            qualified = "numpy." + ".".join(parts)
            return parts[-1], qualified
        if isinstance(node, ast.Name):
            qualified = self.bindings.get(node.id)
            if qualified is None or not qualified.startswith("numpy."):
                return None
            return qualified.rsplit(".", 1)[-1], qualified
        return None

    def _to_expr(self, value):
        if isinstance(value, _Scalar):
            return ast.Constant(value=value.value)
        if isinstance(value, _Array):
            return copy.deepcopy(value.expr)
        if isinstance(value, _Raw):
            return copy.deepcopy(value.expr)
        if isinstance(value, _Sequence):
            return ast.Tuple(
                elts=[self._to_expr(item) for item in value.items],
                ctx=ast.Load(),
            )
        if isinstance(value, _Mapping):
            return ast.Dict(
                keys=[ast.Constant(value=key) for key, _ in value.items],
                values=[self._to_expr(item) for _, item in value.items],
            )
        raise _CannotLower("cannot render value")

    def _as_array_expr(self, value):
        if isinstance(value, (_Array, _Raw)):
            return self._to_expr(value)
        raise _CannotLower("expected array expression")

    @staticmethod
    def _scalar_truth(value):
        if not isinstance(value, _Scalar):
            raise _CannotLower("array-valued condition")
        return bool(value.value)

    def _fresh(self, prefix):
        while True:
            self.counter += 1
            candidate = "__numpilot_%s_%d" % (prefix, self.counter)
            if candidate not in self.used_names:
                self.used_names.add(candidate)
                return candidate


def _arguments(function):
    return (
        list(function.args.posonlyargs)
        + list(function.args.args)
        + list(function.args.kwonlyargs)
    )


def _default_values(function):
    defaults = {}
    positional = list(function.args.posonlyargs) + list(function.args.args)
    for argument, default in zip(
        positional[len(positional) - len(function.args.defaults):],
        function.args.defaults,
    ):
        defaults[argument.arg] = default
    for argument, default in zip(
        function.args.kwonlyargs, function.args.kw_defaults
    ):
        if default is not None:
            defaults[argument.arg] = default
    return defaults


def _module_constants(tree):
    constants = {}
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Constant)
            and isinstance(
                statement.value.value, (bool, int, float, complex, str)
            )
        ):
            constants[statement.targets[0].id] = statement.value.value
    return constants


def _numpy_module():
    # Imported lazily so static analysis never imports AsNumpy or CANN.
    import numpy

    return numpy
