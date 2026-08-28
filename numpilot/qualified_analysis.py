"""Inert, call-profile-aware analysis for ordinary NumPy scripts.

This module is the admission gate in front of model-backed migration.  It
parses source text and performs a deliberately small straight-line data-flow
analysis; it never imports the source, executes an operator, or contacts a
model.  API existence, source-audited computation route, exact call contract,
and runtime identity remain independent facts in the resulting receipt.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from numpilot.capability_profile import (
    CallContractQualification,
    CapabilityDecision,
    ComputationRoute,
    QualifiedCallDecision,
    StaticCallFacts,
    aggregate_decisions,
    build_runtime_route_manifest,
    qualify_call,
)
from numpilot.script_migration import _asnumpy_call_contract_issue


QUALIFIED_ANALYSIS_SCHEMA_VERSION = 1
_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_STATIC_ELEMENTS = 100_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DTYPES = frozenset({
    "bool", "bool_", "float16", "float32", "float64", "int8", "int16",
    "int32", "int64", "uint8", "uint16", "uint32", "uint64",
})
_NUMPY_CONSTANTS = frozenset({"e", "euler_gamma", "inf", "nan", "pi"})
_CREATION = frozenset({
    "arange", "array", "asarray", "empty", "empty_like", "eye", "full",
    "full_like", "identity", "linspace", "ones", "ones_like", "zeros",
    "zeros_like",
})
_REDUCTIONS = frozenset({
    "all", "amax", "amin", "any", "max", "mean", "min", "nanmax",
    "nanprod", "nansum", "prod", "sum",
})
_HOST_SCALAR_REDUCTIONS = _REDUCTIONS - {"all", "any"}
_UNARY = frozenset({
    "absolute", "arccos", "arccosh", "arcsin", "arcsinh", "arctan",
    "arctanh", "ceil", "cos", "cosh", "deg2rad", "degrees", "exp",
    "exp2", "expm1", "fabs", "fix", "floor", "isfinite", "isinf",
    "isneginf", "isposinf", "log", "log10", "log1p", "log2",
    "logical_not", "negative", "positive", "rad2deg", "radians", "real",
    "reciprocal", "rint", "sign", "signbit", "sin", "sinc", "sinh",
    "sqrt", "square", "tan", "tanh", "trunc",
})
_BINARY = frozenset({
    "add", "arctan2", "copysign", "divide", "divmod", "dot", "equal",
    "float_power", "floor_divide", "fmax", "fmin", "fmod", "gcd",
    "greater", "greater_equal", "heaviside", "hypot", "inner", "lcm",
    "ldexp", "less", "less_equal", "logaddexp", "logaddexp2",
    "logical_and", "logical_or", "logical_xor", "matmul", "maximum",
    "minimum", "mod", "multiply", "not_equal", "outer", "power",
    "remainder", "subtract", "true_divide", "vdot",
})
_BOOL_OUTPUT = frozenset({
    "all", "any", "equal", "greater", "greater_equal", "isfinite", "isinf",
    "isneginf", "isposinf", "less", "less_equal", "logical_and",
    "logical_not", "logical_or", "logical_xor", "not_equal", "signbit",
})
_CONTROL_FLOW = (
    ast.Assert, ast.AsyncFor, ast.AsyncFunctionDef, ast.AsyncWith, ast.AugAssign,
    ast.Await, ast.ClassDef, ast.comprehension, ast.Delete, ast.DictComp,
    ast.For, ast.FunctionDef, ast.GeneratorExp, ast.Global, ast.If, ast.IfExp,
    ast.Lambda, ast.ListComp, ast.Match, ast.NamedExpr, ast.Nonlocal, ast.Raise,
    ast.Return, ast.SetComp, ast.Try, ast.While, ast.With, ast.Yield,
    ast.YieldFrom,
)


class QualifiedAnalysisError(ValueError):
    """Source cannot be represented by a bounded static analysis receipt."""


@dataclass(frozen=True)
class QualifiedCallsite:
    """One decision plus its source location.

    The public properties intentionally mirror :class:`QualifiedCallDecision`
    so callers do not need to unwrap the location envelope.
    """

    lineno: int
    col_offset: int
    qualification: QualifiedCallDecision

    @property
    def canonical_key(self) -> str:
        return self.qualification.canonical_key

    @property
    def source_api(self) -> str:
        return self.qualification.source_api

    @property
    def route(self) -> ComputationRoute:
        return self.qualification.route

    @property
    def effective_route(self) -> ComputationRoute:
        return self.qualification.effective_route

    @property
    def contract(self) -> CallContractQualification:
        return self.qualification.contract

    @property
    def decision(self) -> CapabilityDecision:
        return self.qualification.decision

    @property
    def reason_code(self) -> str:
        return self.qualification.reason_code

    @property
    def reasons(self) -> Tuple[str, ...]:
        return self.qualification.reasons

    @property
    def next_action(self) -> str:
        return self.qualification.next_action

    @property
    def facts(self) -> StaticCallFacts:
        return self.qualification.facts

    @property
    def performance_eligible(self) -> bool:
        return self.qualification.performance_eligible

    def to_dict(self) -> Dict[str, Any]:
        payload = self.qualification.to_dict()
        payload["line"] = self.lineno
        payload["column"] = self.col_offset
        return payload


@dataclass(frozen=True)
class QualifiedScriptAnalysis:
    source_path: str
    source_sha256: str
    imports: Tuple[str, ...]
    calls: Tuple[QualifiedCallsite, ...]
    issues: Tuple[str, ...]
    required_rewrites: Tuple[str, ...]
    decision: CapabilityDecision
    runtime_identity_status: str
    runtime_identity_sha256: Optional[str]
    performance_valid: bool = False
    speedup: None = None
    schema_version: int = QUALIFIED_ANALYSIS_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "analysis_mode": "STATIC_AST_ONLY",
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "imports": list(self.imports),
            "calls": [call.to_dict() for call in self.calls],
            "issues": list(self.issues),
            "required_rewrites": list(self.required_rewrites),
            "decision": self.decision.value,
            "next_action": _next_action(self.decision),
            "runtime_identity_status": self.runtime_identity_status,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "actual_remote_call": False,
            "execution_performed": False,
            "performance_valid": self.performance_valid,
            "speedup": self.speedup,
        }


@dataclass(frozen=True)
class _ValueProfile:
    kind: str = "unknown"
    dtype: Optional[str] = "unknown"
    shape: Optional[Tuple[int, ...]] = None

    @property
    def rank(self) -> Optional[int]:
        return None if self.shape is None else len(self.shape)

    @property
    def is_array(self) -> bool:
        return self.kind == "array"


_UNKNOWN = _ValueProfile()


def _next_action(decision: CapabilityDecision) -> str:
    return {
        CapabilityDecision.DIRECT: (
            "eligible for hash-bound migration under the matching runtime identity"
        ),
        CapabilityDecision.FALLBACK_CONSENT: (
            "obtain explicit fallback consent and suppress strict-NPU performance"
        ),
        CapabilityDecision.GAP_REQUIRED: (
            "create a source-bound gap request without calling the migration model"
        ),
        CapabilityDecision.CAPABILITY_AUDIT_REQUIRED: (
            "complete the capability/runtime audit before calling the migration model"
        ),
        CapabilityDecision.BLOCKED: (
            "rewrite or explicitly qualify the unsupported source contract"
        ),
    }[decision]


def _unique(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _attribute_path(node: ast.AST) -> Optional[Tuple[str, ...]]:
    parts: List[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _canonical_numpy_call(call: ast.Call) -> Optional[str]:
    path = _attribute_path(call.func)
    if not path or path[0] != "np" or len(path) < 2:
        return None
    return ".".join(path[1:])


def _static_value(node: Optional[ast.AST]) -> Any:
    """Evaluate only bounded literal arithmetic needed by static signatures."""

    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, int, float, bool, type(None))
    ):
        return node.value
    path = _attribute_path(node)
    if path == ("np", "pi"):
        return math.pi
    if path == ("np", "e"):
        return math.e
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _static_value(node.operand)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left = _static_value(node.left)
        right = _static_value(node.right)
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (left, right)
        ):
            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if right != 0:
                    return left / right
            except (ArithmeticError, OverflowError):
                return None
    if isinstance(node, (ast.Tuple, ast.List)):
        values = tuple(_static_value(item) for item in node.elts)
        if all(value is not None for value in values):
            return values
    return None


def _static_shape(node: Optional[ast.AST]) -> Optional[Tuple[int, ...]]:
    value = _static_value(node)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return (value,)
    if (
        isinstance(value, tuple)
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in value
        )
    ):
        return value
    return None


def _dtype_name(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    path = _attribute_path(node)
    if path and len(path) == 2 and path[0] == "np" and path[1] in _DTYPES:
        return "bool" if path[1] == "bool_" else path[1]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value if node.value in _DTYPES else "unknown"
    return "unknown"


def _bound_argument(
    call: ast.Call,
    position: int,
    names: Iterable[str],
) -> Optional[ast.AST]:
    accepted = set(names)
    if position < len(call.args):
        if any(keyword.arg in accepted for keyword in call.keywords):
            return None
        return call.args[position]
    matches = [
        keyword.value for keyword in call.keywords if keyword.arg in accepted
    ]
    return matches[0] if len(matches) == 1 else None


def _keyword(call: ast.Call, name: str) -> Optional[ast.AST]:
    matches = [item.value for item in call.keywords if item.arg == name]
    return matches[0] if len(matches) == 1 else None


def _literal_nested_shape(node: ast.AST) -> Optional[Tuple[int, ...]]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    if not node.elts:
        return (0,)
    child_shapes = [_literal_nested_shape(item) for item in node.elts]
    if all(shape is None for shape in child_shapes):
        return (len(node.elts),)
    if any(shape is None for shape in child_shapes):
        return None
    first = child_shapes[0]
    if any(shape != first for shape in child_shapes):
        return None
    return (len(node.elts),) + first


def _broadcast_profile(left: _ValueProfile, right: _ValueProfile) -> _ValueProfile:
    arrays = [value for value in (left, right) if value.is_array]
    if not arrays:
        dtype = left.dtype if left.dtype != "unknown" else right.dtype
        return _ValueProfile("scalar", dtype, ())
    dtype = next(
        (value.dtype for value in arrays if value.dtype != "unknown"), "unknown"
    )
    if len(arrays) == 1:
        return _ValueProfile("array", dtype, arrays[0].shape)
    shape = _broadcast_shape(left.shape, right.shape)
    return _ValueProfile("array", dtype, shape)


def _broadcast_shape(
    left: Optional[Tuple[int, ...]],
    right: Optional[Tuple[int, ...]],
) -> Optional[Tuple[int, ...]]:
    """Resolve only NumPy's statically known trailing-dimension broadcast."""

    if left is None or right is None:
        return None
    result = []
    for lhs, rhs in zip(reversed(left), reversed(right)):
        if lhs == rhs or lhs == 1 or rhs == 1:
            result.append(max(lhs, rhs))
        else:
            return None
    longer = left if len(left) >= len(right) else right
    result.extend(reversed(longer[:abs(len(left) - len(right))]))
    return tuple(reversed(result))


def _shape_elements(shape: Tuple[int, ...]) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
        if elements > _MAX_STATIC_ELEMENTS:
            break
    return elements


def _is_authorized_array_division(node: ast.BinOp) -> bool:
    """Mirror the one deterministic division rewrite supported by migration."""

    if not isinstance(node.left, ast.Name) or not isinstance(node.right, ast.Call):
        return False
    denominator = node.right
    if _canonical_numpy_call(denominator) != "sum":
        return False
    value = _bound_argument(denominator, 0, ("a",))
    axis = _bound_argument(denominator, 1, ("axis",))
    keepdims = _bound_argument(denominator, 2, ("keepdims",))
    return (
        isinstance(value, ast.Name)
        and value.id == node.left.id
        and isinstance(_static_value(axis), int)
        and not isinstance(_static_value(axis), bool)
        and _static_value(keepdims) is True
    )


class _StaticAnalyzer:
    def __init__(self, tree: ast.Module, *, route_manifest=None):
        self.tree = tree
        self.route_manifest = route_manifest or build_runtime_route_manifest(None)
        self.env: Dict[str, _ValueProfile] = {}
        self.calls: List[QualifiedCallsite] = []
        self.issues: List[str] = []
        self.rewrites: List[str] = []
        self.imports: List[str] = []
        self.numpy_import_count = 0
        self.parents: Dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self.parents[id(child)] = parent

    def analyze(self) -> None:
        self._scan_structure()
        if self.numpy_import_count != 1:
            self.issues.append(
                "source requires exactly one `import numpy as np` binding"
            )
        for statement in self.tree.body:
            self._eval_statement(statement)

    def _scan_structure(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    rendered = "import %s%s" % (
                        alias.name,
                        " as " + alias.asname if alias.asname else "",
                    )
                    self.imports.append(rendered)
                    if not (alias.name == "numpy" and alias.asname == "np"):
                        self.issues.append(
                            "unsupported import `%s` at line %d; only `import numpy as np` is qualified"
                            % (rendered, node.lineno)
                        )
                    else:
                        self.numpy_import_count += 1
            elif isinstance(node, ast.ImportFrom):
                rendered = "from %s import %s" % (
                    node.module or "",
                    ", ".join(alias.name for alias in node.names),
                )
                self.imports.append(rendered)
                self.issues.append(
                    "unmodelled third-party/import-from surface `%s` at line %d"
                    % (node.module or "", node.lineno)
                )
            elif isinstance(node, _CONTROL_FLOW):
                self.issues.append(
                    "control-flow structure %s is outside straight-line qualification at line %d"
                    % (type(node).__name__, getattr(node, "lineno", 0))
                )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                names = [
                    child.id
                    for target in targets
                    for child in ast.walk(target)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
                ]
                if any(name in {"ap", "np", "print"} for name in names):
                    self.issues.append(
                        "reserved analysis name is rebound at line %d" % node.lineno
                    )
                if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                    self.issues.append(
                        "destructuring assignment is outside static qualification at line %d"
                        % node.lineno
                    )
            elif isinstance(node, ast.NamedExpr):
                if (
                    isinstance(node.target, ast.Name)
                    and node.target.id in {"ap", "np", "print"}
                ):
                    self.issues.append(
                        "reserved analysis name is rebound by a named expression at line %d"
                        % node.lineno
                    )

    def _eval_statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            return
        if isinstance(statement, ast.Assign):
            profile = self._eval_expr(statement.value)
            if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                self.env[statement.targets[0].id] = profile
            return
        if isinstance(statement, ast.AnnAssign):
            profile = self._eval_expr(statement.value) if statement.value else _UNKNOWN
            if isinstance(statement.target, ast.Name):
                self.env[statement.target.id] = profile
            return
        if isinstance(statement, ast.Expr):
            self._eval_expr(statement.value)
            return
        if isinstance(statement, ast.If):
            self._eval_expr(statement.test)
            return
        # Bodies of rejected control-flow constructs are intentionally not
        # traversed: static state must not pretend to know which path ran.

    def _eval_expr(self, node: Optional[ast.AST]) -> _ValueProfile:
        if node is None:
            return _UNKNOWN
        if isinstance(node, ast.Name):
            return self.env.get(node.id, _UNKNOWN)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return _ValueProfile("scalar", "bool", ())
            if isinstance(node.value, float):
                return _ValueProfile("scalar", "float64", ())
            if isinstance(node.value, int):
                return _ValueProfile("scalar", "int64", ())
            return _UNKNOWN
        if isinstance(node, ast.Attribute):
            path = _attribute_path(node)
            if path and len(path) == 2 and path[0] == "np":
                if path[1] in _NUMPY_CONSTANTS:
                    if path[1] == "pi":
                        self.rewrites.append("numpy.pi -> math.pi")
                    else:
                        self.issues.append(
                            "NumPy constant numpy.%s has no implemented exact rewrite at line %d"
                            % (path[1], node.lineno)
                        )
                    return _ValueProfile("scalar", "float64", ())
                if path[1] in _DTYPES:
                    return _ValueProfile("metadata", path[1], None)
                self.issues.append(
                    "unmodelled NumPy attribute numpy.%s at line %d"
                    % (path[1], node.lineno)
                )
            elif (
                path
                and len(path) == 2
                and self.env.get(path[0], _UNKNOWN).is_array
            ):
                self.issues.append(
                    "unmodelled ndarray attribute .%s at line %d"
                    % (path[1], node.lineno)
                )
            return _UNKNOWN
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        if isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            result = _broadcast_profile(left, right)
            if left.is_array or right.is_array:
                if (
                    isinstance(node.op, ast.Div)
                    and left.is_array
                    and right.is_array
                    and _is_authorized_array_division(node)
                    and result.shape is not None
                ):
                    self.rewrites.append("array division -> numpy.divide")
                else:
                    self.issues.append(
                        "array operator %s is outside the namespace-only contract at line %d"
                        % (type(node.op).__name__, node.lineno)
                    )
            return result
        if isinstance(node, ast.Subscript):
            receiver = self._eval_expr(node.value)
            self._eval_expr(node.slice)
            self.issues.append(
                "array subscript is outside the audited ordinary-script contract at line %d"
                % node.lineno
            )
            return _ValueProfile(receiver.kind, receiver.dtype, None)
        if isinstance(node, ast.UnaryOp):
            profile = self._eval_expr(node.operand)
            if profile.is_array:
                self.issues.append(
                    "array operator %s is outside the namespace-only contract at line %d"
                    % (type(node.op).__name__, node.lineno)
                )
            return profile
        if isinstance(node, ast.Compare):
            profiles = [self._eval_expr(node.left)] + [
                self._eval_expr(item) for item in node.comparators
            ]
            if any(profile.is_array for profile in profiles):
                self.issues.append(
                    "array operator Compare is outside the namespace-only contract at line %d"
                    % node.lineno
                )
                array = next(profile for profile in profiles if profile.is_array)
                return _ValueProfile("array", "bool", array.shape)
            return _ValueProfile("scalar", "bool", ())
        if isinstance(node, ast.BoolOp):
            profiles = [self._eval_expr(item) for item in node.values]
            if any(profile.is_array for profile in profiles):
                self.issues.append(
                    "array boolean operator %s is outside the audited contract at line %d"
                    % (type(node.op).__name__, node.lineno)
                )
            return profiles[-1] if profiles else _UNKNOWN
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for item in node.elts:
                self._eval_expr(item)
            return _UNKNOWN
        if isinstance(node, ast.Dict):
            for item in list(node.keys) + list(node.values):
                self._eval_expr(item)
            return _UNKNOWN
        if isinstance(node, ast.JoinedStr):
            for item in node.values:
                self._eval_expr(item)
            return _UNKNOWN
        if isinstance(node, ast.FormattedValue):
            return self._eval_expr(node.value)
        if isinstance(node, ast.IfExp):
            self._eval_expr(node.test)
            left = self._eval_expr(node.body)
            right = self._eval_expr(node.orelse)
            return left if left == right else _UNKNOWN
        for child in ast.iter_child_nodes(node):
            self._eval_expr(child)
        return _UNKNOWN

    def _eval_call(self, call: ast.Call) -> _ValueProfile:
        key = _canonical_numpy_call(call)
        if key is not None:
            argument_profiles = [self._eval_expr(item) for item in call.args]
            for keyword in call.keywords:
                self._eval_expr(keyword.value)
            output = self._numpy_output_profile(key, call, argument_profiles)
            self._check_resource_profile(output, call)
            facts_profile = self._facts_profile(
                key, call, output, argument_profiles
            )
            output_use = self._output_use(call)
            facts = StaticCallFacts(
                positional_args=len(call.args),
                keyword_args=tuple(sorted(
                    keyword.arg if keyword.arg is not None else "**kwargs"
                    for keyword in call.keywords
                )),
                source_contract_ok=self._source_contract_ok(
                    key,
                    call,
                    output_use,
                    argument_profiles,
                    output,
                ),
                dtype=facts_profile.dtype,
                rank=facts_profile.rank,
                shape=facts_profile.shape,
                axis=self._axis_fact(key, call),
                equation=self._equation_fact(key, call),
                output_use=output_use,
            )
            try:
                qualification = qualify_call(
                    key, facts, manifest=self.route_manifest
                )
            except (TypeError, ValueError) as exc:
                self.issues.append(
                    "malformed NumPy API path numpy.%s at line %d: %s"
                    % (key, call.lineno, type(exc).__name__)
                )
                return output
            self.calls.append(QualifiedCallsite(
                lineno=call.lineno,
                col_offset=call.col_offset,
                qualification=qualification,
            ))
            return output

        path = _attribute_path(call.func)
        if isinstance(call.func, ast.Attribute):
            receiver = self._eval_expr(call.func.value)
            for argument in call.args:
                self._eval_expr(argument)
            for keyword in call.keywords:
                self._eval_expr(keyword.value)
            method = call.func.attr
            if method == "item" and not call.args and not call.keywords:
                if not receiver.is_array:
                    self.issues.append(
                        "ndarray.item receiver is not a proved device array at line %d"
                        % call.lineno
                    )
                elif receiver.shape is None or _shape_elements(receiver.shape) != 1:
                    self.issues.append(
                        "ndarray.item requires a statically single-element receiver at line %d"
                        % call.lineno
                    )
                return _ValueProfile("scalar", receiver.dtype, ())
            if method == "astype":
                self.issues.append(
                    "ndarray.astype is not an audited AsNumpy method contract at line %d"
                    % call.lineno
                )
                dtype = _dtype_name(call.args[0]) if call.args else "unknown"
                return _ValueProfile(receiver.kind, dtype, receiver.shape)
            self.issues.append(
                "unmodelled method `%s` at line %d" % (method, call.lineno)
            )
            return _UNKNOWN

        for argument in call.args:
            self._eval_expr(argument)
        for keyword in call.keywords:
            self._eval_expr(keyword.value)
        if isinstance(call.func, ast.Name) and call.func.id == "print":
            return _UNKNOWN
        rendered = ".".join(path) if path else getattr(call.func, "id", "<dynamic>")
        self.issues.append(
            "unmodelled call `%s` at line %d" % (rendered, call.lineno)
        )
        if call.args:
            first = self._profile_without_reanalysis(call.args[0])
            if first.is_array:
                return first
        return _UNKNOWN

    def _profile_without_reanalysis(self, node: ast.AST) -> _ValueProfile:
        if isinstance(node, ast.Name):
            return self.env.get(node.id, _UNKNOWN)
        return _UNKNOWN

    def _numpy_output_profile(
        self,
        key: str,
        call: ast.Call,
        arguments: List[_ValueProfile],
    ) -> _ValueProfile:
        leaf = key.rsplit(".", 1)[-1]
        if leaf in _CREATION:
            dtype = self._creation_dtype(leaf, call, arguments)
            shape = self._creation_shape(leaf, call, arguments)
            return _ValueProfile("array", dtype, shape)
        if key.startswith("random."):
            shape_node = _bound_argument(call, 2, ("size",))
            shape = _static_shape(shape_node)
            return _ValueProfile("array", "float64", shape)
        primary = next(
            (profile for profile in arguments if profile.is_array),
            arguments[0] if arguments else _UNKNOWN,
        )
        if key == "linalg.det":
            return _ValueProfile("array", primary.dtype, ())
        if leaf in _REDUCTIONS:
            axis = self._axis_fact(leaf, call)
            keepdims = _static_value(_keyword(call, "keepdims")) is True
            shape = self._reduced_shape(primary.shape, axis, keepdims)
            requested_dtype = _keyword(call, "dtype")
            dtype = (
                "bool"
                if leaf in {"all", "any"}
                else _dtype_name(requested_dtype)
                if requested_dtype is not None
                else primary.dtype
            )
            kind = (
                "scalar"
                if leaf in _HOST_SCALAR_REDUCTIONS
                and axis in {None, "omitted"}
                else "array"
            )
            return _ValueProfile(kind, dtype, shape)
        if leaf in {"cumprod", "cumsum", "nancumprod", "nancumsum", "sort"}:
            requested_dtype = _keyword(call, "dtype")
            dtype = (
                _dtype_name(requested_dtype)
                if requested_dtype is not None
                else primary.dtype
            )
            return _ValueProfile("array", dtype, primary.shape)
        if leaf == "dot" and len(arguments) >= 2:
            left, right = arguments[:2]
            if (
                left.rank == right.rank == 1
                and left.shape
                and right.shape
                and left.shape[0] == right.shape[0]
            ):
                shape: Optional[Tuple[int, ...]] = ()
            elif (
                left.rank == right.rank == 2
                and left.shape
                and right.shape
                and left.shape[1] == right.shape[0]
            ):
                shape = (left.shape[0], right.shape[1])
            else:
                shape = None
            return _ValueProfile("array", left.dtype, shape)
        if leaf == "inner" and len(arguments) >= 2:
            left, right = arguments[:2]
            if (
                left.shape
                and right.shape
                and left.shape[-1] == right.shape[-1]
            ):
                shape = left.shape[:-1] + right.shape[:-1]
            else:
                shape = None
            return _ValueProfile("array", left.dtype, shape)
        if leaf == "vdot" and len(arguments) >= 2:
            left, right = arguments[:2]
            if (
                left.shape is not None
                and right.shape is not None
                and _shape_elements(left.shape) == _shape_elements(right.shape)
            ):
                shape = ()
            else:
                shape = None
            return _ValueProfile("array", left.dtype, shape)
        if leaf == "outer" and len(arguments) >= 2:
            left, right = arguments[:2]
            shape = (
                (left.shape[0], right.shape[0])
                if left.rank == right.rank == 1 and left.shape and right.shape
                else None
            )
            return _ValueProfile("array", left.dtype, shape)
        if leaf == "matmul" and len(arguments) >= 2:
            left, right = arguments[:2]
            shape = (
                (left.shape[0], right.shape[1])
                if (
                    left.rank == right.rank == 2
                    and left.shape
                    and right.shape
                    and left.shape[1] == right.shape[0]
                )
                else None
            )
            return _ValueProfile("array", left.dtype, shape)
        if leaf in _BINARY and len(arguments) >= 2:
            output = _broadcast_profile(arguments[0], arguments[1])
            if leaf in _BOOL_OUTPUT:
                return _ValueProfile(output.kind, "bool", output.shape)
            return output
        if leaf in _UNARY:
            dtype = "bool" if leaf in _BOOL_OUTPUT else primary.dtype
            return _ValueProfile(primary.kind, dtype, primary.shape)
        if leaf in {"clip", "around", "round_"}:
            return _ValueProfile(primary.kind, primary.dtype, primary.shape)
        return _ValueProfile("array", primary.dtype, primary.shape)

    def _creation_dtype(
        self,
        leaf: str,
        call: ast.Call,
        arguments: List[_ValueProfile],
    ) -> Optional[str]:
        keyword_dtype = _dtype_name(_keyword(call, "dtype"))
        if keyword_dtype is not None:
            return keyword_dtype
        positions = {
            "empty": 1, "empty_like": 1, "full": 2, "full_like": 2,
            "identity": 1, "ones": 1, "ones_like": 1, "zeros": 1,
            "zeros_like": 1,
        }
        position = positions.get(leaf)
        if position is not None and position < len(call.args):
            return _dtype_name(call.args[position])
        if leaf in {"empty_like", "full_like", "ones_like", "zeros_like"} and arguments:
            return arguments[0].dtype
        if leaf in {"array", "asarray"}:
            return arguments[0].dtype if arguments else "unknown"
        return None

    def _creation_shape(
        self,
        leaf: str,
        call: ast.Call,
        arguments: List[_ValueProfile],
    ) -> Optional[Tuple[int, ...]]:
        if leaf in {"full", "empty", "ones", "zeros"}:
            return _static_shape(_bound_argument(call, 0, ("shape",)))
        if leaf in {"full_like", "empty_like", "ones_like", "zeros_like"}:
            return arguments[0].shape if arguments else None
        if leaf in {"eye", "identity"}:
            size = _static_value(_bound_argument(call, 0, ("n", "N")))
            if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
                return (size, size)
            return None
        if leaf == "linspace":
            count = _static_value(_bound_argument(call, 2, ("num",)))
            if count is None:
                count = 50
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                return (count,)
            return None
        if leaf == "arange":
            values = [_static_value(item) for item in call.args[:3]]
            if not values or any(value is None for value in values):
                return None
            try:
                if len(values) == 1:
                    start, stop, step = 0, values[0], 1
                elif len(values) == 2:
                    start, stop, step = values[0], values[1], 1
                else:
                    start, stop, step = values
                if not all(isinstance(value, int) for value in (start, stop, step)):
                    return None
                return (len(range(start, stop, step)),)
            except (TypeError, ValueError, OverflowError):
                return None
        if leaf in {"array", "asarray"} and call.args:
            return _literal_nested_shape(call.args[0])
        return None

    def _facts_profile(
        self,
        key: str,
        call: ast.Call,
        output: _ValueProfile,
        arguments: List[_ValueProfile],
    ) -> _ValueProfile:
        leaf = key.rsplit(".", 1)[-1]
        if leaf in _CREATION or key.startswith("random."):
            return output
        if (
            leaf in _REDUCTIONS
            or leaf in {"cumprod", "cumsum", "nancumprod", "nancumsum"}
        ) and _keyword(call, "dtype") is not None:
            primary = next(
                (profile for profile in arguments if profile.is_array), output
            )
            return _ValueProfile(primary.kind, output.dtype, primary.shape)
        return next(
            (profile for profile in arguments if profile.is_array), output
        )

    def _axis_fact(self, key: str, call: ast.Call) -> Any:
        leaf = key.rsplit(".", 1)[-1]
        relevant = leaf in _REDUCTIONS or leaf in {
            "cross", "cumprod", "cumsum", "nancumprod", "nancumsum", "sort",
        }
        if not relevant:
            return "not_applicable"
        position = 2 if leaf == "cross" else 1
        axis_node = _bound_argument(call, position, ("axis",))
        supplied = position < len(call.args) or _keyword(call, "axis") is not None
        if not supplied:
            return "omitted"
        value = _static_value(axis_node)
        if value is None:
            return None if isinstance(axis_node, ast.Constant) else "unknown"
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, tuple) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        ):
            return value
        return "unknown"

    def _equation_fact(self, key: str, call: ast.Call) -> Optional[str]:
        if key != "einsum":
            return "not_applicable"
        value = _static_value(call.args[0]) if call.args else None
        return value if isinstance(value, str) else "unknown"

    def _source_contract_ok(
        self,
        key: str,
        call: ast.Call,
        output_use: str,
        arguments: List[_ValueProfile],
        output: _ValueProfile,
    ) -> Optional[bool]:
        if (
            len(call.args) > 64
            or len(call.keywords) > 64
            or any(isinstance(argument, ast.Starred) for argument in call.args)
            or any(keyword.arg is None for keyword in call.keywords)
        ):
            return False
        if key == "linalg.det":
            return (
                (len(call.args) == 1 and not call.keywords)
                or (
                    not call.args
                    and len(call.keywords) == 1
                    and call.keywords[0].arg == "a"
                )
            )
        if key.startswith("linalg."):
            if len(call.args) != 1 or call.keywords:
                return False
            return None
        if key.startswith("random."):
            signatures = {
                "random.normal": (3, {"loc", "scale", "size"}),
                "random.standard_normal": (1, {"size"}),
                "random.uniform": (3, {"low", "high", "size"}),
            }
            signature = signatures.get(key)
            if signature is None:
                return None if not call.keywords else False
            positional_limit, allowed = signature
            if len(call.args) > positional_limit or any(
                item.arg not in allowed for item in call.keywords
            ):
                return False
            return True
        leaf = key.rsplit(".", 1)[-1]
        if leaf == "einsum":
            return self._einsum_source_contract(call, arguments)
        if (
            leaf == "matmul"
            and self.route_manifest.entries["matmul"].route
            is ComputationRoute.NATIVE_NPU
        ):
            if len(call.args) != 2 or call.keywords or len(arguments) != 2:
                return False
            left, right = arguments
            if not left.is_array or not right.is_array:
                return False
            if left.dtype in {None, "unknown"} or right.dtype in {None, "unknown"}:
                return None
            if left.dtype != "float32" or right.dtype != "float32":
                return False
            if left.shape is None or right.shape is None:
                return None
            return (
                left.rank == 2
                and right.rank == 2
                and all(dimension > 0 for dimension in left.shape + right.shape)
                and left.shape[1] == right.shape[0]
            )
        if leaf in {"all", "any"}:
            # The audited AsNumpy wrappers return ``ndarray`` even when axis
            # is omitted.  Keep this admission deliberately narrower than the
            # NumPy signature until axis/keepdims variants are source-audited.
            return (
                len(call.args) == 1
                and not call.keywords
                and len(arguments) == 1
                and arguments[0].is_array
            )
        if leaf in _BINARY:
            if len(arguments) < 2 or not all(
                profile.is_array for profile in arguments[:2]
            ):
                return False
            if all(profile.shape is not None for profile in arguments[:2]):
                if output.shape is None:
                    return False
            elif output.shape is None:
                return None
        if leaf == "linspace":
            start = _static_value(_bound_argument(call, 0, ("start",)))
            stop = _static_value(_bound_argument(call, 1, ("stop",)))
            count = _static_value(_bound_argument(call, 2, ("num",)))
            dtype = self._creation_dtype(leaf, call, [])
            return (
                isinstance(start, (int, float))
                and not isinstance(start, bool)
                and isinstance(stop, (int, float))
                and not isinstance(stop, bool)
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
                and dtype not in {None, "unknown"}
                and all(item.arg in {"start", "dtype"} for item in call.keywords)
                and len(call.args) <= 3
            )
        if (
            leaf in _HOST_SCALAR_REDUCTIONS
            and self._axis_fact(leaf, call) in {None, "omitted"}
        ):
            if output_use != "terminal":
                return False
            # With axis omitted the audited wrappers expose a host scalar and
            # ignore parts of NumPy's wider keepdims/dtype contract.  Admit
            # only the exact one-array form until those variants are fixed.
            return (
                len(call.args) == 1
                and not call.keywords
                and len(arguments) == 1
                and arguments[0].is_array
            )
        try:
            return _asnumpy_call_contract_issue(key, call) is None
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _einsum_source_contract(
        call: ast.Call,
        arguments: List[_ValueProfile],
    ) -> Optional[bool]:
        if len(call.args) != 3 or call.keywords:
            return False
        equation = _static_value(call.args[0])
        operands = arguments[1:]
        if not all(profile.is_array for profile in operands):
            return False
        if equation == "a,b->ab":
            ranks = tuple(profile.rank for profile in operands)
            if None in ranks:
                return None
            return ranks == (1, 1)
        if equation == "abcd,abced->abce":
            ranks = tuple(profile.rank for profile in operands)
            if None in ranks:
                return None
            if ranks != (4, 5):
                return False
            left, right = (profile.shape for profile in operands)
            if left is None or right is None:
                return None
            return (
                left[0] == right[0]
                and left[1] == right[1]
                and left[2] == right[2]
                and left[3] == right[4]
            )
        # The equation qualification reports the unsupported spelling.
        return True

    def _check_resource_profile(
        self,
        output: _ValueProfile,
        call: ast.Call,
    ) -> None:
        if (
            output.shape is not None
            and _shape_elements(output.shape) > _MAX_STATIC_ELEMENTS
        ):
            self.issues.append(
                "static resource profile exceeds %d elements at line %d"
                % (_MAX_STATIC_ELEMENTS, call.lineno)
            )

    def _output_use(self, call: ast.Call) -> str:
        current: ast.AST = call
        while id(current) in self.parents:
            parent = self.parents[id(current)]
            if isinstance(parent, ast.Attribute) and parent.value is current:
                grandparent = self.parents.get(id(parent))
                if isinstance(grandparent, ast.Call) and grandparent.func is parent:
                    return "terminal" if parent.attr == "item" else "device"
            if isinstance(parent, ast.Call):
                if parent.func is current:
                    return "device"
                path = _attribute_path(parent.func)
                if path == ("print",):
                    return "terminal"
                return "device"
            if isinstance(parent, (ast.BinOp, ast.Compare, ast.UnaryOp)):
                return "device"
            if isinstance(parent, (ast.FormattedValue, ast.JoinedStr)):
                return "terminal"
            if isinstance(parent, ast.Assign):
                if len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
                    return (
                        "device"
                        if self._name_has_device_use(parent.targets[0].id, parent.lineno)
                        else "terminal"
                    )
                return "unknown"
            if isinstance(parent, ast.Expr):
                return "terminal"
            current = parent
        return "terminal"

    def _name_has_device_use(
        self,
        name: str,
        assigned_line: int,
        seen: Optional[set[str]] = None,
    ) -> bool:
        seen = set() if seen is None else set(seen)
        if name in seen:
            return False
        seen.add(name)
        for node in ast.walk(self.tree):
            if not (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id == name
                and getattr(node, "lineno", 0) >= assigned_line
            ):
                continue
            current: ast.AST = node
            while id(current) in self.parents:
                parent = self.parents[id(current)]
                if isinstance(parent, ast.Attribute) and parent.value is current:
                    grandparent = self.parents.get(id(parent))
                    if isinstance(grandparent, ast.Call) and grandparent.func is parent:
                        if parent.attr == "item":
                            break
                        return True
                if isinstance(parent, ast.Call):
                    path = _attribute_path(parent.func)
                    if path == ("print",):
                        break
                    return True
                if isinstance(parent, (ast.BinOp, ast.Compare, ast.UnaryOp)):
                    return True
                if (
                    isinstance(parent, ast.Assign)
                    and parent.value is current
                    and len(parent.targets) == 1
                    and isinstance(parent.targets[0], ast.Name)
                ):
                    if self._name_has_device_use(
                        parent.targets[0].id,
                        parent.lineno,
                        seen,
                    ):
                        return True
                    break
                if (
                    isinstance(parent, ast.AnnAssign)
                    and parent.value is current
                    and isinstance(parent.target, ast.Name)
                ):
                    if self._name_has_device_use(
                        parent.target.id,
                        parent.lineno,
                        seen,
                    ):
                        return True
                    break
                if isinstance(parent, (ast.FormattedValue, ast.JoinedStr, ast.Expr)):
                    break
                current = parent
        return False

    @staticmethod
    def _reduced_shape(
        shape: Optional[Tuple[int, ...]],
        axis: Any,
        keepdims: bool,
    ) -> Optional[Tuple[int, ...]]:
        if shape is None:
            return None
        if axis in {None, "omitted"}:
            return tuple(1 for _ in shape) if keepdims else ()
        if not isinstance(axis, int) or isinstance(axis, bool) or not shape:
            return None
        normalized = axis if axis >= 0 else len(shape) + axis
        if not 0 <= normalized < len(shape):
            return None
        if keepdims:
            result = list(shape)
            result[normalized] = 1
            return tuple(result)
        return shape[:normalized] + shape[normalized + 1:]


def _runtime_state(runtime_identity: Any) -> Tuple[str, Optional[str], bool]:
    if runtime_identity is None:
        return "missing", None, False
    if isinstance(runtime_identity, Mapping):
        status = runtime_identity.get("status")
        identity_sha = runtime_identity.get("identity_sha256")
    else:
        status = getattr(runtime_identity, "status", None)
        identity_sha = getattr(runtime_identity, "identity_sha256", None)
    if hasattr(status, "value"):
        status = status.value
    rendered_status = str(status or "missing")
    valid_sha = (
        identity_sha
        if isinstance(identity_sha, str) and _SHA256.fullmatch(identity_sha)
        else None
    )
    matched = rendered_status.lower() == "matched" and valid_sha is not None
    return rendered_status, valid_sha, matched


def _runtime_downgrade(call: QualifiedCallsite) -> QualifiedCallsite:
    current = call.qualification
    if current.decision is not CapabilityDecision.DIRECT:
        return call
    downgraded = QualifiedCallDecision(
        canonical_key=current.canonical_key,
        source_api=current.source_api,
        route=current.route,
        effective_route=current.effective_route,
        contract=current.contract,
        decision=CapabilityDecision.CAPABILITY_AUDIT_REQUIRED,
        reason_code="RUNTIME_IDENTITY_AUDIT_REQUIRED",
        reasons=(
            "the current runtime identity is missing, drifted, or not fully matched",
        ),
        next_action="complete the runtime identity audit before migration",
        facts=current.facts,
    )
    return QualifiedCallsite(call.lineno, call.col_offset, downgraded)


def analyze_qualified_script(
    source: str,
    source_path: str,
    *,
    runtime_identity: Any = None,
) -> QualifiedScriptAnalysis:
    """Return a machine-readable, fail-closed analysis without side effects."""

    if not isinstance(source, str):
        raise QualifiedAnalysisError("source must be text")
    try:
        encoded = source.encode("utf-8")
    except UnicodeError as exc:
        raise QualifiedAnalysisError("source must be UTF-8") from exc
    if len(encoded) > _MAX_SOURCE_BYTES:
        raise QualifiedAnalysisError(
            "source exceeds the %d-byte static-analysis limit" % _MAX_SOURCE_BYTES
        )
    if (
        not isinstance(source_path, str)
        or not source_path
        or source_path != source_path.strip()
        or "\x00" in source_path
        or len(source_path) > 4096
    ):
        raise QualifiedAnalysisError("source path is invalid")
    try:
        tree = ast.parse(source, filename=source_path)
    except SyntaxError as exc:
        raise QualifiedAnalysisError(
            "source has invalid Python syntax at line %s" % (exc.lineno or "?")
        ) from exc

    if isinstance(runtime_identity, Mapping):
        runtime_audit_id = runtime_identity.get("audit_id")
    else:
        runtime_audit_id = getattr(runtime_identity, "audit_id", None)
    analyzer = _StaticAnalyzer(
        tree,
        route_manifest=build_runtime_route_manifest(runtime_audit_id),
    )
    analyzer.analyze()
    runtime_status, runtime_sha, runtime_matched = _runtime_state(runtime_identity)
    calls = tuple(analyzer.calls)
    if not runtime_matched:
        calls = tuple(_runtime_downgrade(call) for call in calls)

    issues = _unique(analyzer.issues)
    if issues:
        decision = CapabilityDecision.BLOCKED
    elif not calls:
        issues = ("source contains no qualified NumPy calls",)
        decision = CapabilityDecision.BLOCKED
    else:
        decisions = [call.decision for call in calls]
        if not runtime_matched:
            decisions.append(CapabilityDecision.CAPABILITY_AUDIT_REQUIRED)
        decision = aggregate_decisions(decisions)

    return QualifiedScriptAnalysis(
        source_path=source_path,
        source_sha256=hashlib.sha256(encoded).hexdigest(),
        imports=_unique(analyzer.imports),
        calls=calls,
        issues=issues,
        required_rewrites=_unique(analyzer.rewrites),
        decision=decision,
        runtime_identity_status=runtime_status,
        runtime_identity_sha256=runtime_sha,
    )
