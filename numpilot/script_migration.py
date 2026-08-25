"""Simple, whole-file NumPy to AsNumpy migration.

This is the user-facing migration lane for ordinary ``import numpy as np``
scripts.  It deliberately does not execute the source and does not require the
``build_inputs``/``forward`` benchmark contract.  Generated source still comes
from the configured LLM; NumPilot independently proves that its Python AST is
exactly the allowed namespace-only rewrite before publishing it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import tokenize
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from numpilot.agents.analysis import _OpVisitor
from numpilot.capabilities import ExecutionProvenance, parse_profile_constraints
from numpilot.coverage import is_supported
from numpilot.engines.llm import LlmEngine, LlmError


_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_NUMPY_DTYPE_NAMES = frozenset({
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "complex64", "complex128", "bool_",
})
_SAFE_METADATA_PATHS = _NUMPY_DTYPE_NAMES | frozenset({"ndarray"})


class ScriptMigrationError(RuntimeError):
    """An ordinary script cannot be safely scanned or migrated."""


@dataclass(frozen=True)
class ScriptUse:
    api_path: str
    qualified_api: str
    kind: str
    lineno: int
    col_offset: int
    positional_args: int = 0
    keyword_args: Tuple[str, ...] = ()
    synthetic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_path": self.api_path,
            "qualified_api": self.qualified_api,
            "kind": self.kind,
            "lineno": self.lineno,
            "col_offset": self.col_offset,
            "positional_args": self.positional_args,
            "keyword_args": list(self.keyword_args),
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True)
class ScriptScanResult:
    source_path: str
    source_sha256: str
    source: str
    uses: Tuple[ScriptUse, ...]
    issues: Tuple[str, ...]
    rewrite_count: int
    synthetic_divide_count: int
    output_boundary_count: int

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def call_uses(self) -> Tuple[ScriptUse, ...]:
        return tuple(item for item in self.uses if item.kind == "call")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "ok": self.ok,
            "issues": list(self.issues),
            "rewrite_count": self.rewrite_count,
            "synthetic_divide_count": self.synthetic_divide_count,
            "output_boundary_count": self.output_boundary_count,
            "uses": [item.to_dict() for item in self.uses],
        }


@dataclass(frozen=True)
class ScriptApiDecision:
    api_path: str
    qualified_api: str
    kind: str
    status: str
    available: bool
    provenance: str
    evidence: str
    constraints: Tuple[str, ...]
    callsites: Tuple[ScriptUse, ...]
    generatable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_path": self.api_path,
            "qualified_api": self.qualified_api,
            "kind": self.kind,
            "status": self.status,
            "available": self.available,
            "provenance": self.provenance,
            "evidence": self.evidence,
            "constraints": list(self.constraints),
            "callsites": [item.to_dict() for item in self.callsites],
            "generatable": self.generatable,
        }


@dataclass(frozen=True)
class ScriptAnalysis:
    scan: ScriptScanResult
    backend: str
    backend_version: Optional[str]
    audit_id: str
    decisions: Tuple[ScriptApiDecision, ...]

    @property
    def strict(self) -> Tuple[ScriptApiDecision, ...]:
        return tuple(item for item in self.decisions if item.status == "strict_npu")

    @property
    def fallbacks(self) -> Tuple[ScriptApiDecision, ...]:
        return tuple(
            item for item in self.decisions
            if item.status == "fallback_available"
        )

    @property
    def scoped(self) -> Tuple[ScriptApiDecision, ...]:
        return tuple(
            item for item in self.decisions if item.status == "scoped_npu"
        )

    @property
    def unsupported(self) -> Tuple[ScriptApiDecision, ...]:
        return tuple(item for item in self.decisions if item.status == "unsupported")

    @property
    def audit_required(self) -> Tuple[ScriptApiDecision, ...]:
        """Available paths whose execution provenance is not yet qualified."""

        return tuple(
            item
            for item in self.decisions
            if item.status == "capability_audit_required"
        )

    @property
    def can_migrate_directly(self) -> bool:
        return (
            self.scan.ok
            and not self.fallbacks
            and not self.audit_required
            and not self.unsupported
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": "script",
            "scan": self.scan.to_dict(),
            "backend": self.backend,
            "backend_version": self.backend_version,
            "audit_id": self.audit_id,
            "counts": {
                "strict_npu": len(self.strict),
                "scoped_npu": len(self.scoped),
                "fallback_available": len(self.fallbacks),
                "capability_audit_required": len(self.audit_required),
                "unsupported": len(self.unsupported),
            },
            "decisions": [item.to_dict() for item in self.decisions],
        }


@dataclass(frozen=True)
class ScriptMigrationArtifact:
    migrated_source: str
    migrated_sha256: str
    receipt: Mapping[str, Any]
    model: Optional[str]
    response_model: str
    provider: str
    response_content: str


@dataclass(frozen=True, order=True)
class _RewriteLocation:
    """One source-bound AST location; never a text-replacement range."""

    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int

    @classmethod
    def from_node(cls, node: ast.AST) -> "_RewriteLocation":
        return cls(
            lineno=int(getattr(node, "lineno", 0)),
            col_offset=int(getattr(node, "col_offset", 0)),
            end_lineno=int(getattr(node, "end_lineno", 0)),
            end_col_offset=int(getattr(node, "end_col_offset", 0)),
        )

    def render(self) -> str:
        return "%d:%d" % (self.lineno, self.col_offset)


@dataclass(frozen=True)
class _ScriptRewritePlan:
    """The two narrowly proven expression rewrites allowed in script mode."""

    divisions: Tuple[_RewriteLocation, ...]
    output_boundaries: Tuple[_RewriteLocation, ...]


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


def _unique(items: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def detect_source_mode(source: str) -> str:
    """Return ``contract`` only for the existing explicit strict contract."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ScriptMigrationError(
            "source has invalid Python syntax at line %s" % (exc.lineno or "?")
        ) from exc
    counts = {"build_inputs": 0, "forward": 0}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in counts:
            counts[node.name] += 1
    return (
        "contract"
        if counts == {"build_inputs": 1, "forward": 1}
        else "script"
    )


def read_script_source(path: str) -> str:
    source_path = Path(path)
    if source_path.is_symlink() or not source_path.is_file():
        raise ScriptMigrationError("source must be a regular file")
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise ScriptMigrationError(
            "cannot read source: %s" % type(exc).__name__
        ) from exc
    if len(payload) > _MAX_SOURCE_BYTES:
        raise ScriptMigrationError(
            "source exceeds the %d-byte limit" % _MAX_SOURCE_BYTES
        )
    try:
        return payload.decode("utf-8")
    except UnicodeError as exc:
        raise ScriptMigrationError("source must be UTF-8") from exc


def _is_numeric_zero(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and node.value == 0
    )


def _asnumpy_call_contract_issue(
    api_path: str,
    call: ast.Call,
) -> Optional[str]:
    """Reuse the strict forward call contract for one direct ``np.*`` call.

    ``_OpVisitor`` also admits ``np.maximum(x, 0)`` because the legacy
    forward rewriter turns that expression into ``ap.relu(x)``.  Whole-file
    script migration is intentionally namespace-only, so that exception must
    remain rejected here.
    """

    if "." in api_path or not is_supported(api_path):
        # Nested and genuinely missing APIs are classified later so they can
        # still produce an honest unsupported/gap hand-off.
        return None
    validator = _OpVisitor({"np": "numpy"})
    supported, reason = validator._call_supported(  # noqa: SLF001
        api_path,
        "numpy." + api_path,
        call,
    )
    if (
        supported
        and api_path == "maximum"
        and len(call.args) == 2
        and not call.keywords
        and any(_is_numeric_zero(argument) for argument in call.args)
    ):
        return (
            "namespace-only migration cannot use the legacy maximum-to-relu "
            "rewrite for a scalar zero operand"
        )
    return None if supported else (reason or "call contract was rejected")


def _is_direct_numpy_call(node: ast.AST, api_path: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and _attribute_path(node.func) == ("np", api_path)
        and not any(isinstance(argument, ast.Starred) for argument in node.args)
        and not any(keyword.arg is None for keyword in node.keywords)
    )


def _bound_call_argument(
    call: ast.Call,
    position: int,
    keyword: str,
) -> Optional[ast.AST]:
    if position < len(call.args):
        if any(item.arg == keyword for item in call.keywords):
            return None
        return call.args[position]
    matches = [item.value for item in call.keywords if item.arg == keyword]
    return matches[0] if len(matches) == 1 else None


def _is_static_integer(node: Optional[ast.AST]) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def _is_static_array_call(node: ast.AST, array_names: Iterable[str]) -> bool:
    """Prove only the small AsNumpy-return surface needed by the demo.

    This is intentionally not an inference that every ``np.*`` result is an
    ndarray.  Each admitted wrapper has a source-audited ndarray return, and
    operand propagation is limited to names already proved in the same
    module-level straight-line block.  Capability/provenance is checked later.
    """

    if not isinstance(node, ast.Call):
        return False
    known = set(array_names)
    path = _attribute_path(node.func)
    if path == ("np", "full"):
        return True
    if path == ("np", "exp"):
        return (
            len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in known
        )
    if path == ("np", "matmul"):
        return (
            len(node.args) == 2
            and not node.keywords
            and all(
                isinstance(argument, ast.Name) and argument.id in known
                for argument in node.args
            )
        )
    if path == ("np", "divide"):
        return (
            len(node.args) == 2
            and not node.keywords
            and all(
                (
                    isinstance(argument, ast.Name)
                    and argument.id in known
                )
                or _is_static_array_call(argument, known)
                for argument in node.args
            )
        )
    if path == ("np", "sum"):
        value = _bound_call_argument(node, 0, "a")
        axis = _bound_call_argument(node, 1, "axis")
        return (
            isinstance(value, ast.Name)
            and value.id in known
            and _is_static_integer(axis)
        )
    return False


def _is_authorized_array_division(
    node: ast.AST,
    array_names: Iterable[str],
) -> bool:
    """Recognize ``array / np.sum(same_array, axis=int, keepdims=True)``.

    The exact shape is deliberate: AsNumpy currently has no ndarray division
    dunder, while its audited ``divide`` callable accepts two arrays.  Requiring
    a non-None reduction axis plus ``keepdims=True`` proves the denominator is
    an AsNumpy ndarray rather than a Python scalar.
    """

    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return False
    if not isinstance(node.left, ast.Name) or node.left.id not in set(array_names):
        return False
    if not _is_direct_numpy_call(node.right, "sum"):
        return False
    denominator = node.right
    value = _bound_call_argument(denominator, 0, "a")
    axis = _bound_call_argument(denominator, 1, "axis")
    keepdims = _bound_call_argument(denominator, 2, "keepdims")
    return (
        isinstance(value, ast.Name)
        and value.id == node.left.id
        and _is_static_integer(axis)
        and isinstance(keepdims, ast.Constant)
        and keepdims.value is True
        and _is_static_array_call(denominator, array_names)
    )


def _print_output_locations(
    statement: ast.stmt,
    array_names: Iterable[str],
) -> Tuple[_RewriteLocation, ...]:
    """Locate proven arrays at a narrowly defined human-output boundary."""

    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "print"
        and not statement.value.keywords
        and not any(
            isinstance(argument, ast.Starred)
            for argument in statement.value.args
        )
    ):
        return ()
    known = set(array_names)
    locations: List[_RewriteLocation] = []
    for argument in statement.value.args:
        if not isinstance(argument, ast.JoinedStr):
            continue
        for value in argument.values:
            if not (
                isinstance(value, ast.FormattedValue)
                and value.conversion == -1
                and value.format_spec is None
                and isinstance(value.value, ast.Name)
                and value.value.id in known
            ):
                continue
            locations.append(_RewriteLocation.from_node(value.value))
    return tuple(locations)


def _build_script_rewrite_plan(tree: ast.Module) -> _ScriptRewritePlan:
    """Run one fail-closed, module-level straight-line ndarray proof."""

    array_names = set()
    divisions: List[_RewriteLocation] = []
    outputs: List[_RewriteLocation] = []
    straight_line = True
    builtin_print = True
    for statement in tree.body:
        if not straight_line:
            break
        if isinstance(statement, ast.Import):
            continue
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(
                statement.targets[0], ast.Name
            ):
                straight_line = False
                continue
            target = statement.targets[0].id
            if target == "print":
                builtin_print = False
            authorized_division = _is_authorized_array_division(
                statement.value, array_names
            )
            if authorized_division:
                divisions.append(_RewriteLocation.from_node(statement.value))
            if authorized_division or _is_static_array_call(
                statement.value, array_names
            ):
                array_names.add(target)
            else:
                array_names.discard(target)
            continue
        if isinstance(statement, ast.Expr) and builtin_print:
            output_locations = _print_output_locations(statement, array_names)
            if output_locations:
                outputs.extend(output_locations)
                continue
        # A control-flow boundary, definition, arbitrary expression statement,
        # or unfamiliar binding ends the proof.  Later operators stay blocked.
        straight_line = False
    return _ScriptRewritePlan(
        divisions=tuple(divisions),
        output_boundaries=tuple(outputs),
    )


def _python_surface_issues(
    tree: ast.AST,
    rewrite_plan: _ScriptRewritePlan,
) -> Tuple[str, ...]:
    """Reject unqualified ndarray dunders/methods before invoking the LLM."""

    issues: List[str] = []
    allowed_divisions = set(rewrite_plan.divisions)
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.BinOp):
            if (
                isinstance(node.op, ast.Div)
                and _RewriteLocation.from_node(node) in allowed_divisions
            ):
                continue
            issues.append(
                "Python binary operator %s is not audited for AsNumpy arrays "
                "at line %d; use an explicit np.* API"
                % (type(node.op).__name__, line)
            )
        elif isinstance(node, ast.AugAssign):
            issues.append(
                "Python augmented operator %s is not audited for AsNumpy arrays "
                "at line %d; use an explicit np.* API"
                % (type(node.op).__name__, line)
            )
        elif isinstance(node, ast.Compare):
            operators = ",".join(type(item).__name__ for item in node.ops)
            issues.append(
                "Python comparison %s is not audited for AsNumpy arrays at "
                "line %d; use an explicit np.* API" % (operators, line)
            )
        elif isinstance(node, ast.UnaryOp) and not isinstance(
            node.operand, ast.Constant
        ):
            issues.append(
                "Python unary operator %s is not audited for AsNumpy arrays "
                "at line %d; use an explicit np.* API"
                % (type(node.op).__name__, line)
            )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            path = _attribute_path(node.func)
            if path is None or path[0] != "np":
                issues.append(
                    "attribute/method call .%s() is not audited for AsNumpy "
                    "arrays at line %d" % (node.func.attr, line)
                )
    return _unique(issues)


def scan_numpy_script(source: str, source_path: str = "<memory>") -> ScriptScanResult:
    """Scan one inert source string without importing or executing it."""

    if len(source.encode("utf-8")) > _MAX_SOURCE_BYTES:
        raise ScriptMigrationError(
            "source exceeds the %d-byte limit" % _MAX_SOURCE_BYTES
        )
    try:
        tree = ast.parse(source, filename=source_path)
    except SyntaxError as exc:
        raise ScriptMigrationError(
            "source has invalid Python syntax at line %s" % (exc.lineno or "?")
        ) from exc

    rewrite_plan = _build_script_rewrite_plan(tree)
    issues: List[str] = list(_python_surface_issues(tree, rewrite_plan))
    exact_numpy_imports = 0
    top_level_ids = {id(node) for node in tree.body}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (
                    alias.name == "numpy"
                    and alias.asname == "np"
                    and id(node) in top_level_ids
                    and len(node.names) == 1
                ):
                    exact_numpy_imports += 1
                    continue
                issues.append(
                    "only one top-level `import numpy as np` is allowed; "
                    "found import %s%s at line %d"
                    % (
                        alias.name,
                        " as " + alias.asname if alias.asname else "",
                        getattr(node, "lineno", 0),
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__" and id(node) in top_level_ids:
                continue
            issues.append(
                "non-NumPy/from import is outside simple script mode at line %d"
                % getattr(node, "lineno", 0)
            )

    if exact_numpy_imports != 1:
        issues.append(
            "source must contain exactly one top-level `import numpy as np`; found %d"
            % exact_numpy_imports
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "np" and not isinstance(
            node.ctx, ast.Load
        ):
            issues.append("NumPy alias `np` is rebound at line %d" % node.lineno)
        if isinstance(node, ast.arg) and node.arg in {"np", "ap"}:
            issues.append(
                "reserved migration alias `%s` is used as an argument at line %d"
                % (node.arg, node.lineno)
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in {"np", "ap"}:
                issues.append(
                    "reserved migration alias `%s` is rebound at line %d"
                    % (node.name, node.lineno)
                )
        if isinstance(node, ast.ExceptHandler) and node.name in {"np", "ap"}:
            issues.append(
                "reserved migration alias `%s` is rebound at line %d"
                % (node.name, node.lineno)
            )
        if isinstance(node, (ast.Global, ast.Nonlocal)) and any(
            name in {"np", "ap"} for name in node.names
        ):
            issues.append(
                "reserved migration alias is used by global/nonlocal at line %d"
                % node.lineno
            )
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name in {
            "np", "ap"
        }:
            issues.append(
                "reserved migration alias `%s` is rebound by a pattern at line %d"
                % (node.name, node.lineno)
            )
        if isinstance(node, ast.MatchMapping) and node.rest in {"np", "ap"}:
            issues.append(
                "reserved migration alias `%s` is rebound by a pattern at line %d"
                % (node.rest, node.lineno)
            )
        if isinstance(node, ast.Name) and node.id == "ap":
            issues.append(
                "source already uses reserved AsNumpy alias `ap` at line %d"
                % node.lineno
            )

    parents: Dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    call_nodes: Dict[int, ast.Call] = {
        id(node.func): node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    uses: List[ScriptUse] = []
    outer_attribute_ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parent = parents.get(id(node))
        if isinstance(parent, ast.Attribute) and parent.value is node:
            continue
        path = _attribute_path(node)
        if not path or path[0] != "np":
            continue
        outer_attribute_ids.add(id(node))
        api_path = ".".join(path[1:])
        if not api_path:
            continue
        call = call_nodes.get(id(node))
        if call is not None and (
            any(isinstance(argument, ast.Starred) for argument in call.args)
            or any(keyword.arg is None for keyword in call.keywords)
        ):
            issues.append(
                "dynamic *args/**kwargs is outside simple script mode at line %d"
                % getattr(call, "lineno", 0)
            )
        if call is not None and (
            len(call.args) > 64 or len(call.keywords) > 64
        ):
            issues.append(
                "NumPy call signature exceeds the 64-argument limit at line %d"
                % getattr(call, "lineno", 0)
            )
        if call is not None:
            contract_issue = _asnumpy_call_contract_issue(api_path, call)
            if contract_issue:
                issues.append(
                    "AsNumpy call contract rejected numpy.%s at line %d: %s"
                    % (
                        api_path,
                        getattr(call, "lineno", 0),
                        contract_issue,
                    )
                )
        uses.append(ScriptUse(
            api_path=api_path,
            qualified_api="numpy." + api_path,
            kind="call" if call is not None else "reference",
            lineno=getattr(node, "lineno", 0),
            col_offset=getattr(node, "col_offset", 0),
            positional_args=len(call.args) if call is not None else 0,
            keyword_args=tuple(sorted(
                keyword.arg if keyword.arg is not None else "**kwargs"
                for keyword in (call.keywords if call is not None else ())
            )),
        ))

    # A proved ``array / array`` will be emitted as ``ap.divide(left, right)``.
    # Model it as the call that will actually execute so the same backend
    # manifest, signature qualification, fallback-consent, and gap logic apply.
    for location in rewrite_plan.divisions:
        uses.append(ScriptUse(
            api_path="divide",
            qualified_api="numpy.divide",
            kind="call",
            lineno=location.lineno,
            col_offset=location.col_offset,
            positional_args=2,
            keyword_args=(),
            synthetic=True,
        ))

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Name)
            and node.id == "np"
            and isinstance(node.ctx, ast.Load)
        ):
            continue
        parent = parents.get(id(node))
        if not (isinstance(parent, ast.Attribute) and parent.value is node):
            issues.append(
                "NumPy module alias `np` is used dynamically at line %d"
                % node.lineno
            )

    uses.sort(key=lambda item: (item.lineno, item.col_offset, item.api_path))
    rewrite_count = sum(
        isinstance(node, ast.Name)
        and node.id == "np"
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(tree)
    )
    return ScriptScanResult(
        source_path=source_path,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        uses=tuple(uses),
        issues=_unique(issues),
        rewrite_count=rewrite_count,
        synthetic_divide_count=len(rewrite_plan.divisions),
        output_boundary_count=len(rewrite_plan.output_boundaries),
    )


def _resolve_runtime_path(module: Any, path: str) -> Tuple[bool, Any, str]:
    current = module
    try:
        for component in path.split("."):
            current = getattr(current, component)
    except Exception as exc:
        return False, None, "runtime lookup failed: %s: %s" % (
            type(exc).__name__, str(exc)[:160]
        )
    return True, current, "runtime attribute is available"


def _decision_is_generatable(
    api_path: str,
    kind: str,
    *,
    status: str,
    available: bool,
    provenance: str,
) -> bool:
    structural_match = (
        kind == "call"
        and "." not in api_path
        and _SAFE_IDENTIFIER.fullmatch(api_path) is not None
        and api_path not in _SAFE_METADATA_PATHS
    )
    if not structural_match:
        return False
    if status == "fallback_available":
        return available and provenance in {
            ExecutionProvenance.CPU_FALLBACK.value,
            ExecutionProvenance.CONDITIONAL_CPU_FALLBACK.value,
            ExecutionProvenance.HOST_ASSISTED.value,
        }
    return (
        status == "unsupported"
        and not available
        and provenance == ExecutionProvenance.UNKNOWN.value
    )


def classify_script(scan: ScriptScanResult, backend: Any) -> ScriptAnalysis:
    """Classify every use without conflating missing APIs and unknown provenance."""

    grouped: Dict[Tuple[str, str], List[ScriptUse]] = {}
    for use in scan.uses:
        grouped.setdefault((use.api_path, use.kind), []).append(use)

    top_level_calls = sorted({
        path
        for path, kind in grouped
        if kind == "call" and "." not in path and path not in _SAFE_METADATA_PATHS
    })
    manifest = backend.capability_manifest(top_level_calls)
    module = backend.get_module()
    decisions: List[ScriptApiDecision] = []
    for (api_path, kind), callsites in sorted(grouped.items()):
        qualified = "numpy." + api_path
        available = False
        provenance = ExecutionProvenance.UNKNOWN.value
        evidence = ""
        constraints: Tuple[str, ...] = ()
        status = "unsupported"

        if api_path in _SAFE_METADATA_PATHS:
            available, _value, evidence = _resolve_runtime_path(module, api_path)
            if available:
                status = "strict_npu"
                provenance = "metadata"
                evidence = "AsNumpy exposes the NumPy-compatible metadata alias"
        elif kind == "call" and "." not in api_path:
            capability = manifest.operators.get(api_path)
            if capability is not None:
                available = capability.available
                provenance = capability.provenance.value
                evidence = capability.evidence
                constraints = capability.constraints
                if capability.strict_npu:
                    status = "strict_npu"
                elif (
                    available
                    and capability.provenance.is_strict_npu
                    and capability.constraints
                    and capability.signature_variants
                ):
                    qualified_signatures = {
                        (item.positional_args, item.keyword_args)
                        for item in capability.signature_variants
                    }
                    source_signatures = {
                        (item.positional_args, item.keyword_args)
                        for item in callsites
                    }
                    try:
                        parse_profile_constraints(capability.constraints)
                    except ValueError:
                        pass
                    else:
                        if source_signatures <= qualified_signatures:
                            status = "scoped_npu"
                            evidence = (
                                capability.evidence
                                + "; call signature is qualified; dtype/shape/layout "
                                "remain subject to the runtime profile guard"
                            )
                elif (
                    available
                    and capability.provenance in {
                        ExecutionProvenance.CPU_FALLBACK,
                        ExecutionProvenance.CONDITIONAL_CPU_FALLBACK,
                        ExecutionProvenance.HOST_ASSISTED,
                        ExecutionProvenance.SHIM_CPU,
                    }
                ):
                    status = "fallback_available"
                elif available:
                    status = "capability_audit_required"
                    evidence = (
                        capability.evidence
                        + "; API is callable but its exact source/runtime profile "
                        "has not passed a version-bound capability audit"
                    )
        else:
            available, value, evidence = _resolve_runtime_path(module, api_path)
            if available:
                if kind == "call" and not callable(value):
                    available = False
                    evidence = "runtime attribute exists but is not callable"
                else:
                    status = "capability_audit_required"
                    evidence = (
                        "AsNumpy exposes this nested/reference path, but NumPilot "
                        "has no version-bound strict-NPU provenance for it"
                    )

        if not available:
            status = "unsupported"
            provenance = ExecutionProvenance.UNKNOWN.value
            if not evidence:
                evidence = "AsNumpy runtime does not expose this path"
        decisions.append(ScriptApiDecision(
            api_path=api_path,
            qualified_api=qualified,
            kind=kind,
            status=status,
            available=available,
            provenance=provenance,
            evidence=evidence,
            constraints=constraints,
            callsites=tuple(callsites),
            generatable=_decision_is_generatable(
                api_path,
                kind,
                status=status,
                available=available,
                provenance=provenance,
            ),
        ))

    return ScriptAnalysis(
        scan=scan,
        backend=manifest.backend,
        backend_version=manifest.backend_version,
        audit_id=manifest.audit_id,
        decisions=tuple(decisions),
    )


def _validated_rewrite_plan(analysis: ScriptAnalysis) -> _ScriptRewritePlan:
    """Recompute and bind every synthetic rewrite to the capability analysis."""

    source = analysis.scan.source
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != (
        analysis.scan.source_sha256
    ):
        raise ScriptMigrationError("analysis source hash is inconsistent")
    try:
        tree = ast.parse(source, type_comments=True)
    except SyntaxError as exc:
        raise ScriptMigrationError("analysis source syntax is inconsistent") from exc
    plan = _build_script_rewrite_plan(tree)
    if (
        analysis.scan.synthetic_divide_count != len(plan.divisions)
        or analysis.scan.output_boundary_count != len(plan.output_boundaries)
    ):
        raise ScriptMigrationError("analysis synthetic rewrite counts are inconsistent")

    expected_synthetic = tuple(
        (
            location.lineno,
            location.col_offset,
            2,
            (),
        )
        for location in plan.divisions
    )
    actual_synthetic = tuple(
        (
            use.lineno,
            use.col_offset,
            use.positional_args,
            use.keyword_args,
        )
        for use in analysis.scan.uses
        if use.synthetic
        and use.kind == "call"
        and use.api_path == "divide"
        and use.qualified_api == "numpy.divide"
    )
    if actual_synthetic != expected_synthetic or any(
        use.synthetic
        and not (
            use.kind == "call"
            and use.api_path == "divide"
            and use.qualified_api == "numpy.divide"
        )
        for use in analysis.scan.uses
    ):
        raise ScriptMigrationError(
            "analysis synthetic numpy.divide callsites are inconsistent"
        )

    if plan.divisions:
        matching = [
            decision
            for decision in analysis.decisions
            if decision.kind == "call" and decision.api_path == "divide"
        ]
        if len(matching) != 1:
            raise ScriptMigrationError(
                "synthetic numpy.divide capability decision is unavailable"
            )
        decision = matching[0]
        decision_synthetic = tuple(
            (
                use.lineno,
                use.col_offset,
                use.positional_args,
                use.keyword_args,
            )
            for use in decision.callsites
            if use.synthetic
        )
        if (
            decision_synthetic != expected_synthetic
            or not decision.available
            or decision.status
            not in {"strict_npu", "scoped_npu", "fallback_available"}
        ):
            raise ScriptMigrationError(
                "synthetic numpy.divide did not pass capability/callsite audit"
            )
    return plan


class _ExpectedScriptRewrite(ast.NodeTransformer):
    def __init__(self, rewrite_plan: _ScriptRewritePlan):
        self._divisions = set(rewrite_plan.divisions)
        self._output_boundaries = set(rewrite_plan.output_boundaries)

    def visit_Import(self, node: ast.Import) -> ast.AST:
        if (
            len(node.names) == 1
            and node.names[0].name == "numpy"
            and node.names[0].asname == "np"
        ):
            replacement = ast.Import(names=[ast.alias(name="asnumpy", asname="ap")])
            return ast.copy_location(replacement, node)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "np" and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Name(id="ap", ctx=ast.Load()), node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if _RewriteLocation.from_node(node) not in self._divisions:
            return self.generic_visit(node)
        if not isinstance(node.op, ast.Div):
            raise ScriptMigrationError("authorized division location changed operator")
        replacement = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="ap", ctx=ast.Load()),
                attr="divide",
                ctx=ast.Load(),
            ),
            args=[self.visit(node.left), self.visit(node.right)],
            keywords=[],
        )
        return ast.copy_location(replacement, node)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> ast.AST:
        location = _RewriteLocation.from_node(node.value)
        if location not in self._output_boundaries:
            return self.generic_visit(node)
        if (
            node.conversion != -1
            or node.format_spec is not None
            or not isinstance(node.value, ast.Name)
        ):
            raise ScriptMigrationError("authorized output boundary changed shape")
        converted = ast.Call(
            func=ast.Attribute(
                value=self.visit(node.value),
                attr="to_numpy",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )
        replacement = ast.FormattedValue(
            value=converted,
            conversion=-1,
            format_spec=None,
        )
        return ast.copy_location(replacement, node)


def expected_script_tree(
    source: str,
    analysis: Optional[ScriptAnalysis] = None,
) -> ast.Module:
    tree = ast.parse(source, type_comments=True)
    source_plan = _build_script_rewrite_plan(tree)
    if source_plan.divisions or source_plan.output_boundaries:
        if analysis is None:
            raise ScriptMigrationError(
                "synthetic expression rewrites require a source-bound analysis"
            )
        if analysis.scan.source != source:
            raise ScriptMigrationError("rewrite analysis is bound to another source")
        rewrite_plan = _validated_rewrite_plan(analysis)
        if rewrite_plan != source_plan:
            raise ScriptMigrationError("rewrite plan changed after analysis")
    else:
        rewrite_plan = source_plan
    rewritten = _ExpectedScriptRewrite(rewrite_plan).visit(tree)
    ast.fix_missing_locations(rewritten)
    return rewritten


def _comment_tokens(source: str) -> Tuple[str, ...]:
    try:
        return tuple(
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        )
    except (IndentationError, tokenize.TokenError) as exc:
        raise ScriptMigrationError("cannot tokenize migration source") from exc


def audit_glm_script(
    source: str,
    migrated_source: str,
    analysis: Optional[ScriptAnalysis] = None,
) -> None:
    """Require exact AST equality with the one allowed mechanical rewrite."""

    try:
        actual = ast.parse(migrated_source, type_comments=True)
    except SyntaxError as exc:
        raise ScriptMigrationError(
            "GLM returned invalid Python syntax at line %s" % (exc.lineno or "?")
        ) from exc
    expected = expected_script_tree(source, analysis)
    if ast.dump(actual, include_attributes=False) != ast.dump(
        expected, include_attributes=False
    ):
        raise ScriptMigrationError(
            "GLM output changed code outside the exact authorized rewrite plan"
        )
    if _comment_tokens(source) != _comment_tokens(migrated_source):
        raise ScriptMigrationError(
            "GLM output changed comments outside the exact namespace rewrite"
        )
    for node in ast.walk(actual):
        if isinstance(node, ast.Import):
            if any(alias.name == "numpy" for alias in node.names):
                raise ScriptMigrationError("GLM output retains a NumPy import")
        if isinstance(node, ast.Name) and node.id == "np":
            raise ScriptMigrationError("GLM output retains the NumPy alias `np`")


_SCRIPT_SYSTEM_PROMPT = """You are NumPilot's mechanical source migration worker.
Return one complete Python file in a ```python code block and no explanation.
Make exactly these AST-level changes and nothing else:
1. Replace the single `import numpy as np` with `import asnumpy as ap` in the same location.
2. Replace every reference to the bound name `np` with `ap`.
{expression_rules}
Do not add, delete, reorder, simplify, repair, or otherwise change any other statement or expression.
Do not change strings, constants, function/class names, arguments, or control flow.
NumPilot will reject the response unless its Python AST exactly equals this source-bound rewrite plan.
"""


def _script_system_prompt(rewrite_plan: _ScriptRewritePlan) -> str:
    rules: List[str] = []
    if rewrite_plan.divisions:
        rules.append(
            "3. At exactly these source AST locations, replace `left / right` "
            "with `ap.divide(left, right)`: %s. Do not rewrite any other "
            "Python operator."
            % ", ".join(item.render() for item in rewrite_plan.divisions)
        )
    if rewrite_plan.output_boundaries:
        rules.append(
            "%d. Only at these print f-string value locations, replace the "
            "array expression `value` with `value.to_numpy()`: %s. Do not "
            "add output conversion anywhere else."
            % (
                3 + bool(rewrite_plan.divisions),
                ", ".join(
                    item.render() for item in rewrite_plan.output_boundaries
                ),
            )
        )
    if not rules:
        rules.append("3. Do not change any expression.")
    return _SCRIPT_SYSTEM_PROMPT.format(expression_rules="\n".join(rules))

_SCRIPT_RESPONSE = re.compile(
    r"\A[ \t\r\n]*```python[ \t]*\r?\n(?P<code>.*?)\r?\n```[ \t\r\n]*\Z",
    re.DOTALL,
)


def _extract_strict_script_response(content: str) -> str:
    if not isinstance(content, str):
        raise ScriptMigrationError("GLM response content must be text")
    match = _SCRIPT_RESPONSE.fullmatch(content)
    if match is None:
        raise ScriptMigrationError(
            "GLM response must contain exactly one python code block and no prose"
        )
    code = match.group("code")
    if "```" in code:
        raise ScriptMigrationError(
            "GLM response must contain exactly one python code block and no prose"
        )
    if not code.strip():
        raise ScriptMigrationError("GLM returned empty migration code")
    return code.rstrip() + "\n"


def _validate_script_receipt(
    receipt: object,
    response_content: str,
    *,
    expected_model: str,
) -> Dict[str, Any]:
    from numpilot.gaps.provider import (
        DraftGenerationError,
        validate_remote_receipt,
    )

    if not isinstance(expected_model, str) or not expected_model.strip():
        raise ScriptMigrationError("configured migration model identity is unavailable")
    try:
        validated = validate_remote_receipt(receipt, response_content)
    except DraftGenerationError as exc:
        raise ScriptMigrationError(
            "GLM receipt failed validation: %s" % exc
        ) from exc
    if validated.get("requested_model") != expected_model:
        raise ScriptMigrationError(
            "GLM receipt model does not match the configured model"
        )
    response_model = validated.get("response_model")
    if (
        not isinstance(response_model, str)
        or not response_model
        or not response_model.lower().startswith("glm-")
    ):
        raise ScriptMigrationError(
            "GLM receipt has no valid GLM response_model evidence"
        )
    if response_model != validated["requested_model"]:
        raise ScriptMigrationError(
            "GLM receipt response_model does not match requested/configured model"
        )
    return validated


def generate_script_with_glm(
    analysis: ScriptAnalysis,
    *,
    expected_model: str,
    expected_provider: str,
    chat: Optional[Callable[[str, str], Tuple[str, Mapping[str, Any]]]] = None,
) -> ScriptMigrationArtifact:
    """Ask the configured GLM for code, then prove it is the exact allowed AST."""

    if not analysis.scan.ok:
        raise ScriptMigrationError("script scan did not pass")
    if analysis.unsupported:
        raise ScriptMigrationError("unsupported capability cannot enter GLM migration")
    if analysis.audit_required:
        raise ScriptMigrationError(
            "unaudited capability cannot enter GLM migration"
        )
    if (
        not isinstance(expected_provider, str)
        or expected_provider.strip().lower() != "glm"
        or not isinstance(expected_model, str)
        or not expected_model.strip().lower().startswith("glm-")
    ):
        raise ScriptMigrationError(
            "ordinary-script migration requires an explicit GLM provider/model"
        )
    chat = chat or LlmEngine._chat_with_receipt
    rewrite_plan = _validated_rewrite_plan(analysis)
    allowed = sorted({item.api_path for item in analysis.decisions})
    user = (
        "Allowed NumPy paths already classified by NumPilot: %s\n\n"
        "Authorized synthetic numpy.divide callsites: %s\n"
        "Authorized print output-boundary values: %s\n\n"
        "Source SHA-256: %s\n\n```python\n%s\n```"
        % (
            allowed,
            [item.render() for item in rewrite_plan.divisions],
            [item.render() for item in rewrite_plan.output_boundaries],
            analysis.scan.source_sha256,
            analysis.scan.source,
        )
    )
    try:
        content, receipt = chat(_script_system_prompt(rewrite_plan), user)
    except Exception as exc:
        raise LlmError(
            "ordinary-script GLM request failed: %s: %s"
            % (type(exc).__name__, str(exc)[:200])
        ) from exc
    migrated = _extract_strict_script_response(content)
    safe_receipt = _validate_script_receipt(
        receipt,
        content,
        expected_model=expected_model,
    )
    audit_glm_script(analysis.scan.source, migrated, analysis)
    return ScriptMigrationArtifact(
        migrated_source=migrated,
        migrated_sha256=hashlib.sha256(
            migrated.encode("utf-8")
        ).hexdigest(),
        receipt=safe_receipt,
        model=safe_receipt["response_model"],
        response_model=safe_receipt["response_model"],
        provider="glm",
        response_content=content,
    )


def build_gap_request(
    analysis: ScriptAnalysis,
    decisions: Sequence[ScriptApiDecision],
    *,
    target_soc: str,
    cann_version: str,
) -> Dict[str, Any]:
    """Prepare source-bound GapSpecs without claiming generation or qualification."""

    from numpilot.gaps.spec import build_script_gap_specs

    generatable = [item for item in decisions if item.generatable]
    try:
        specs = build_script_gap_specs(
            analysis.scan.source,
            analysis.scan.source_path,
            [item.to_dict() for item in generatable],
            target_soc=target_soc,
            cann_version=cann_version,
        )
    except ValueError as exc:
        raise ScriptMigrationError(
            "cannot prepare a source-bound script GapSpec: %s" % exc
        ) from exc

    return {
        "schema_version": 1,
        "request_type": "script_api_gap",
        "status": (
            "gap_generation_required"
            if specs else "gap_generation_unavailable"
        ),
        "source_path": analysis.scan.source_path,
        "source_sha256": analysis.scan.source_sha256,
        "backend": analysis.backend,
        "backend_version": analysis.backend_version,
        "audit_id": analysis.audit_id,
        "target_soc": target_soc,
        "cann_version": cann_version,
        "gaps": [
            {
                "qualified_api": item.qualified_api,
                "asnumpy_api": "asnumpy." + item.api_path,
                "reason": item.evidence,
                "current_status": item.status,
                "current_available": item.available,
                "current_provenance": item.provenance,
                "generatable": item.generatable,
                "signature_variants": sorted({
                    (
                        call.positional_args,
                        call.keyword_args,
                    )
                    for call in item.callsites
                    if call.kind == "call"
                }),
                "callsites": [call.to_dict() for call in item.callsites],
            }
            for item in decisions
        ],
        "gap_specs": [spec.to_dict() for spec in specs],
        "generated_code": False,
        "qualification_performed": False,
        "activation_performed": False,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = -1
    temporary = ""
    try:
        fd, temporary = tempfile.mkstemp(
            prefix="." + path.name + "-", suffix=".tmp", dir=str(path.parent)
        )
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ScriptMigrationError(
                "managed output appeared during publication; use a fresh --out: %s"
                % path
            ) from exc
        os.unlink(temporary)
        temporary = ""
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def write_script_artifacts(
    out_dir: str,
    analysis: ScriptAnalysis,
    *,
    status: str,
    fallback_accepted: bool,
    artifact: Optional[ScriptMigrationArtifact] = None,
    gap_request: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Publish only an audited final file; failures never publish migrated.py."""

    root = Path(out_dir).absolute()
    migrated_path = root / "migrated.py"
    gap_path = root / "gap-request.json"
    report_path = root / "script-migration-report.json"
    managed_paths = (migrated_path, gap_path, report_path)

    source_path = os.path.abspath(analysis.scan.source_path)
    source_realpath = os.path.realpath(source_path)
    for managed_path in managed_paths:
        managed_realpath = os.path.realpath(os.path.abspath(str(managed_path)))
        aliases_source = source_realpath == managed_realpath
        if not aliases_source and os.path.lexists(managed_path):
            try:
                aliases_source = os.path.samefile(source_path, managed_path)
            except OSError:
                aliases_source = False
        if aliases_source:
            raise ScriptMigrationError(
                "source conflicts with managed output %s; use a fresh --out"
                % managed_path
            )

    existing_managed = [
        str(path) for path in managed_paths if os.path.lexists(path)
    ]
    if existing_managed:
        raise ScriptMigrationError(
            "managed output already exists; use a fresh --out: %s"
            % ", ".join(existing_managed)
        )

    if hashlib.sha256(analysis.scan.source.encode("utf-8")).hexdigest() != (
        analysis.scan.source_sha256
    ):
        raise ScriptMigrationError("analysis source hash is inconsistent")
    if artifact is not None and gap_request is not None:
        raise ScriptMigrationError(
            "one run cannot publish both migrated code and a gap request"
        )
    migrated_statuses = {
        "migrated_strict_npu",
        "migrated_scoped_npu",
        "migrated_with_fallback",
    }
    if (artifact is not None) != (status in migrated_statuses):
        raise ScriptMigrationError("migration status/artifact combination is invalid")

    safe_receipt = None
    migrated_payload = None
    if artifact is not None:
        if analysis.unsupported or analysis.audit_required or not analysis.scan.ok:
            raise ScriptMigrationError("blocked analysis cannot publish migrated code")
        if status == "migrated_with_fallback":
            if not fallback_accepted or not analysis.fallbacks:
                raise ScriptMigrationError("fallback migration lacks explicit consent")
        elif fallback_accepted or analysis.fallbacks:
            raise ScriptMigrationError("non-fallback migration has fallback state")
        if status == "migrated_scoped_npu" and not analysis.scoped:
            raise ScriptMigrationError("scoped migration lacks scoped capability")
        if status == "migrated_strict_npu" and analysis.scoped:
            raise ScriptMigrationError("scoped capability cannot claim universal strict NPU")
        if not isinstance(artifact.model, str) or not artifact.model:
            raise ScriptMigrationError("migration artifact model identity is unavailable")
        if artifact.provider != "glm" or not artifact.model.lower().startswith("glm-"):
            raise ScriptMigrationError("migration artifact is not bound to GLM")
        safe_receipt = _validate_script_receipt(
            artifact.receipt,
            artifact.response_content,
            expected_model=artifact.model,
        )
        if (
            artifact.response_model != safe_receipt["response_model"]
            or artifact.model != safe_receipt["response_model"]
        ):
            raise ScriptMigrationError(
                "migration artifact response_model evidence is inconsistent"
            )
        extracted = _extract_strict_script_response(artifact.response_content)
        if extracted != artifact.migrated_source:
            raise ScriptMigrationError("migration artifact is not bound to model content")
        audit_glm_script(
            analysis.scan.source,
            artifact.migrated_source,
            analysis,
        )
        migrated_payload = artifact.migrated_source.encode("utf-8")
        if hashlib.sha256(migrated_payload).hexdigest() != artifact.migrated_sha256:
            raise ScriptMigrationError("migration artifact hash is inconsistent")
    report: Dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "mode": "script",
        "analysis": analysis.to_dict(),
        "fallback_accepted": bool(fallback_accepted),
        "strict_npu": status == "migrated_strict_npu",
        "performance_valid": False,
        "execution_performed": False,
        "numerical_verification": "not_run",
        "generated_by": "glm" if artifact is not None else None,
        "source_sha256": analysis.scan.source_sha256,
        "ast_audit": "passed" if artifact is not None else "not_run",
        "published_files": (
            ["migrated.py"] if artifact is not None
            else (["gap-request.json"] if gap_request is not None else [])
        ),
    }
    if artifact is not None:
        report.update({
            "migrated_sha256": artifact.migrated_sha256,
            "model": safe_receipt["response_model"],
            "requested_model": safe_receipt["requested_model"],
            "response_model": safe_receipt["response_model"],
            "provider": artifact.provider,
            "llm_receipt": safe_receipt,
        })
    gap_payload = None
    if gap_request is not None:
        gap_payload = (json.dumps(
            _json_ready(dict(gap_request)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n").encode("utf-8")
    report_payload = (json.dumps(
        _json_ready(report), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n").encode("utf-8")

    paths: Dict[str, str] = {}
    if artifact is not None:
        _atomic_write(migrated_path, migrated_payload)
        paths["migrated"] = str(migrated_path)
    if gap_request is not None:
        _atomic_write(gap_path, gap_payload)
        paths["gap_request"] = str(gap_path)
    _atomic_write(report_path, report_payload)
    paths["report"] = str(report_path)
    return paths
