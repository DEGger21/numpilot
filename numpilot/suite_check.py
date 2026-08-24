"""Static, non-executing admission checks for NumPilot migration suites."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from numpilot.agents.analysis import (
    AnalysisAgent,
    numpy_bindings_for_forward,
    resolve_numpy_function,
)
from numpilot.contracts import OffloadPlan
from numpilot.coverage import SUPPORTED_OPS
from numpilot.engines.ast_rewrite import AstRewriteEngine, audit_forward_surface
from numpilot.gaps.spec import blocked_unsupported, build_gap_specs


_MAX_CASE_BYTES = 4 * 1024 * 1024
_MAX_CASES = 128
_ALLOWED_IMPORT_ROOTS = frozenset({"numpy", "__future__"})
_NON_NUMPY_OR_UNAUDITED_DIRECT_APIS = frozenset({
    "angle", "gelu", "relu", "softmax",
})
_SAFE_BUILD_INPUT_NUMPY_CALLS = frozenset({
    "numpy.arange",
    "numpy.array",
    "numpy.asarray",
    "numpy.empty",
    "numpy.empty_like",
    "numpy.eye",
    "numpy.full",
    "numpy.full_like",
    "numpy.geomspace",
    "numpy.identity",
    "numpy.linspace",
    "numpy.logspace",
    "numpy.ones",
    "numpy.ones_like",
    "numpy.random.normal",
    "numpy.random.rand",
    "numpy.random.randn",
    "numpy.random.random",
    "numpy.random.random_sample",
    "numpy.random.standard_normal",
    "numpy.random.uniform",
    "numpy.zeros",
    "numpy.zeros_like",
})


class SuiteCheckError(RuntimeError):
    """The requested suite cannot be inspected safely or deterministically."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _discover_case_files(paths: Sequence[str]) -> Tuple[Path, ...]:
    if not paths:
        raise SuiteCheckError("at least one case file or directory is required")
    discovered: List[Path] = []
    seen = set()
    for raw in paths:
        path = Path(raw).absolute()
        if path.is_symlink():
            raise SuiteCheckError("suite input may not be a symlink: %s" % path)
        if path.is_dir():
            candidates: Iterable[Path] = sorted(path.glob("*.py"))
        elif path.is_file():
            candidates = (path,)
        else:
            raise SuiteCheckError("suite input does not exist: %s" % path)
        for candidate in candidates:
            candidate = candidate.absolute()
            if candidate.is_symlink() or not candidate.is_file():
                raise SuiteCheckError(
                    "case must be an ordinary Python file: %s" % candidate
                )
            if candidate.suffix != ".py":
                raise SuiteCheckError("case must end in .py: %s" % candidate)
            identity = str(candidate)
            if identity not in seen:
                seen.add(identity)
                discovered.append(candidate)
    if not discovered:
        raise SuiteCheckError("suite contains no Python case files")
    if len(discovered) > _MAX_CASES:
        raise SuiteCheckError(
            "suite contains too many cases (%d > %d)"
            % (len(discovered), _MAX_CASES)
        )
    return tuple(sorted(discovered, key=lambda item: str(item)))


def _read_source(path: Path) -> Tuple[bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SuiteCheckError(
            "cannot read case %s: %s" % (path, type(exc).__name__)
        ) from exc
    if len(payload) > _MAX_CASE_BYTES:
        raise SuiteCheckError(
            "case exceeds the %d-byte limit: %s" % (_MAX_CASE_BYTES, path)
        )
    try:
        return payload, payload.decode("utf-8")
    except UnicodeError as exc:
        raise SuiteCheckError("case is not UTF-8: %s" % path) from exc


def _function_counts(tree: ast.Module) -> Mapping[str, int]:
    counts = {"build_inputs": 0, "forward": 0}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in counts:
            counts[node.name] += 1
        elif isinstance(node, ast.AsyncFunctionDef) and node.name in counts:
            # An async definition is counted so that the exact synchronous
            # contract fails with one useful structural diagnostic.
            counts[node.name] += 1
    return counts


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or node.orelse:
        return False
    test = node.test
    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    ):
        return False
    return True


def _structure_violations(tree: ast.Module) -> List[str]:
    violations: List[str] = []
    counts = _function_counts(tree)
    for name in ("build_inputs", "forward"):
        if counts[name] != 1:
            violations.append(
                "source must define exactly one top-level synchronous %s; found %d"
                % (name, counts[name])
            )
    for node in tree.body:
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name in {"build_inputs", "forward"}
        ):
            violations.append("%s must be synchronous" % node.name)
        if isinstance(node, ast.FunctionDef) and node.name in {
            "build_inputs", "forward"
        }:
            if node.decorator_list:
                violations.append("%s may not use decorators" % node.name)
            args = node.args
            if node.name == "build_inputs" and any((
                args.posonlyargs,
                args.args,
                args.kwonlyargs,
                args.vararg,
                args.kwarg,
            )):
                violations.append("build_inputs must accept no arguments")
            if node.name == "forward" and (
                not (args.posonlyargs or args.args)
                or args.kwonlyargs
                or args.vararg is not None
                or args.kwarg is not None
                or args.defaults
                or any(item is not None for item in args.kw_defaults)
            ):
                violations.append(
                    "forward must use one or more required positional arguments only"
                )
    for index, node in enumerate(tree.body):
        allowed = isinstance(node, (ast.Import, ast.ImportFrom))
        allowed = allowed or (
            isinstance(node, ast.FunctionDef)
            and node.name in {"build_inputs", "forward"}
        )
        allowed = allowed or _is_main_guard(node)
        allowed = allowed or (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if not allowed:
            violations.append(
                "top-level %s is outside the inert suite contract at line %d"
                % (type(node).__name__, getattr(node, "lineno", 0))
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules = (node.module or "",)
            if node.level:
                violations.append(
                    "relative import is outside the pure-NumPy suite contract at line %d"
                    % getattr(node, "lineno", 0)
                )
                continue
        else:
            continue
        for module in modules:
            root = module.split(".", 1)[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                violations.append(
                    "non-NumPy import %s is outside the suite contract at line %d"
                    % (module or "<empty>", getattr(node, "lineno", 0))
                )
    return list(dict.fromkeys(violations))


def _build_inputs_violations(tree: ast.Module) -> List[str]:
    """Audit the input factory without executing user-controlled source."""

    builds = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_inputs"
    ]
    forwards = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    if len(builds) != 1 or len(forwards) != 1:
        return []
    build = builds[0]
    forward = forwards[0]
    bindings = numpy_bindings_for_forward(tree, build)
    local_names = set()
    violations: List[str] = []

    def expression_violations(expression: ast.AST) -> List[str]:
        errors: List[str] = []
        forbidden = (
            ast.Await,
            ast.BoolOp,
            ast.Compare,
            ast.DictComp,
            ast.GeneratorExp,
            ast.IfExp,
            ast.Lambda,
            ast.ListComp,
            ast.NamedExpr,
            ast.SetComp,
            ast.Subscript,
            ast.Yield,
            ast.YieldFrom,
        )
        for node in ast.walk(expression):
            line = getattr(node, "lineno", 0)
            if isinstance(node, ast.Call):
                resolved = resolve_numpy_function(node.func, bindings)
                qualified = resolved[1] if resolved is not None else None
                if qualified not in _SAFE_BUILD_INPUT_NUMPY_CALLS:
                    errors.append(
                        "build_inputs call is outside the pure constructor "
                        "allowlist at line %d: %s"
                        % (line, qualified or "non-NumPy call")
                    )
                if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
                    keyword.arg is None for keyword in node.keywords
                ):
                    errors.append(
                        "build_inputs may not dynamically expand call arguments "
                        "at line %d" % line
                    )
            elif isinstance(node, ast.Attribute):
                root = node
                while isinstance(root, ast.Attribute):
                    root = root.value
                if not (
                    isinstance(root, ast.Name)
                    and root.id in bindings
                ):
                    errors.append(
                        "build_inputs attribute access is outside NumPy at line %d"
                        % line
                    )
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in bindings and node.id not in local_names:
                    errors.append(
                        "build_inputs reads an unbound/non-NumPy name at line %d"
                        % line
                    )
            elif isinstance(node, forbidden):
                errors.append(
                    "build_inputs expression %s is outside the static contract "
                    "at line %d" % (type(node).__name__, line)
                )
            elif isinstance(node, ast.UnaryOp) and not (
                isinstance(node.op, (ast.UAdd, ast.USub))
                and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, (int, float, complex))
            ):
                errors.append(
                    "build_inputs unary expression is outside the static contract "
                    "at line %d" % line
                )
        return errors

    returns = []
    statements = list(build.body)
    for index, statement in enumerate(statements):
        if (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                violations.append(
                    "build_inputs assignments must target one local name at line %d"
                    % getattr(statement, "lineno", 0)
                )
            if statement.value is None:
                violations.append("build_inputs assignment must have a value")
            else:
                violations.extend(expression_violations(statement.value))
            if len(targets) == 1 and isinstance(targets[0], ast.Name):
                local_names.add(targets[0].id)
            continue
        if isinstance(statement, ast.Return):
            returns.append(statement)
            if statement.value is not None:
                violations.extend(expression_violations(statement.value))
            continue
        violations.append(
            "build_inputs statement %s is outside the static contract at line %d"
            % (type(statement).__name__, getattr(statement, "lineno", 0))
        )

    if len(returns) != 1:
        violations.append(
            "build_inputs must contain exactly one return; found %d" % len(returns)
        )
    else:
        returned = returns[0].value
        if not isinstance(returned, ast.Tuple) or any(
            isinstance(item, ast.Starred) for item in returned.elts
        ):
            violations.append("build_inputs must return one explicit tuple")
        else:
            forward_arity = len(forward.args.posonlyargs) + len(forward.args.args)
            if len(returned.elts) != forward_arity:
                violations.append(
                    "build_inputs tuple arity %d does not match forward arity %d"
                    % (len(returned.elts), forward_arity)
                )
    if returns and statements[-1] is not returns[-1]:
        violations.append("build_inputs return must be the final statement")
    return list(dict.fromkeys(violations))


def _gap_lane(spec) -> Mapping[str, Any]:
    name = spec.api_name
    candidate = getattr(np, name, None)
    aliases = sorted(
        other
        for other in SUPPORTED_OPS
        if other != name and getattr(np, other, None) is candidate
    )
    reasons: List[str] = []
    if aliases:
        reasons.append(
            "NumPy API aliases an already covered operation: %s"
            % ", ".join(aliases)
        )
    if not isinstance(candidate, np.ufunc):
        reasons.append("missing API is not a top-level NumPy ufunc")
        nin = None
        nout = None
    else:
        nin = int(candidate.nin)
        nout = int(candidate.nout)
        if nin not in (1, 2):
            reasons.append("ufunc input arity is not unary/binary")
        if nout != 1:
            reasons.append("ufunc has multiple outputs")
    signatures = tuple(spec.signature_variants)
    if any(
        signature.get("keyword_args")
        or signature.get("positional_args") != nin
        for signature in signatures
    ):
        reasons.append("observed call signature is outside simple positional ufunc form")
    return {
        "suitable": not reasons,
        "kind": "simple_unary_binary_ufunc" if not reasons else "manual_or_route_review",
        "numpy_ufunc_nin": nin,
        "numpy_ufunc_nout": nout,
        "covered_aliases": aliases,
        "reasons": reasons,
    }


def _forced_surface_audit(source: str, analysis) -> Mapping[str, Any]:
    placement = {op.name: "NPU" for op in analysis.op_uses}
    if not placement:
        return {"ok": False, "ap_calls": {}, "violations": ["forward has no NumPy calls"]}
    migrated = AstRewriteEngine().rewrite(
        source,
        OffloadPlan(placement=placement, threshold_elements=1),
    )
    audit = audit_forward_surface(migrated)
    return {
        "ok": audit.ok,
        "ap_calls": dict(sorted(audit.ap_calls.items())),
        "violations": list(audit.violations),
    }


def _inspect_case(path: Path, *, target_soc: str, cann_version: str) -> Dict[str, Any]:
    payload, source = _read_source(path)
    base: Dict[str, Any] = {
        "path": str(path),
        "source_sha256": _sha256(payload),
        "category": "blocked",
        "eligible": False,
        "runtime_capability_checked": False,
    }
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        base.update({
            "errors": [
                "syntax error at line %s: %s" % (exc.lineno or "?", exc.msg)
            ],
            "warnings": [],
        })
        return base

    structure = _structure_violations(tree)
    structure.extend(_build_inputs_violations(tree))
    analysis = AnalysisAgent().analyze(
        source, str(path), strict_input_types=True
    )
    try:
        specs = build_gap_specs(
            source,
            analysis,
            target_soc=target_soc,
            cann_version=cann_version,
        )
        blocked = blocked_unsupported(analysis)
    except ValueError as exc:
        specs = ()
        blocked = ({"name": "gap_spec", "reason": str(exc)},)
    surface = _forced_surface_audit(source, analysis)
    domain_ops = sorted({
        op.name for op in analysis.op_uses if op.kind != "numpy_api"
    })
    errors = list(structure)
    unavailable_direct = sorted({
        op.name for op in analysis.op_uses
        if op.name in _NON_NUMPY_OR_UNAUDITED_DIRECT_APIS
    })
    if unavailable_direct:
        errors.append(
            "APIs are not audited current AsNumpy targets for the pure-NumPy "
            "suite: %s" % ", ".join(unavailable_direct)
        )
    if domain_ops:
        errors.append(
            "suite cases must use only NumPy APIs; found domain operators: %s"
            % ", ".join(domain_ops)
        )
    if not surface["ok"]:
        errors.extend(surface["violations"])

    unique_ops = sorted({op.name for op in analysis.op_uses})
    unsupported = sorted(analysis.unsupported)
    gap_lane = _gap_lane(specs[0]) if len(specs) == 1 else None
    blocked_details = [dict(item) for item in blocked]
    category = "blocked"
    if not errors and analysis.migratable and not specs and not blocked_details:
        category = "direct_migration_candidate"
    elif (
        not errors
        and len(specs) == 1
        and not blocked_details
        and unsupported == [specs[0].api_name]
        and gap_lane is not None
        and gap_lane["suitable"]
    ):
        category = "single_gap_candidate"
    else:
        if len(specs) > 1:
            errors.append("case contains more than one generatable missing API")
        elif len(specs) == 1 and gap_lane and not gap_lane["suitable"]:
            errors.extend(gap_lane["reasons"])
        if blocked_details:
            errors.extend(
                "%s: %s"
                % (item.get("qualified_api") or item.get("name", "blocked"),
                   item.get("reason", "unsupported"))
                for item in blocked_details
            )
        if not analysis.op_uses:
            errors.append("forward contains no recognized NumPy API calls")

    warnings = [
        "static coverage is not runtime strict-NPU provenance; pin and check the target AsNumpy wheel"
    ]
    if category == "single_gap_candidate":
        warnings.append(
            "dtype, shape, layout, output contract and NumPy oracle still require runtime qualification"
        )
    base.update({
        "category": category,
        "eligible": category != "blocked",
        "callsite_count": len(analysis.op_uses),
        "operators": unique_ops,
        "unsupported": unsupported,
        "generatable_gaps": [spec.to_dict() for spec in specs],
        "blocked_unsupported": blocked_details,
        "forced_forward_surface": surface,
        "gap_lane": gap_lane,
        "errors": list(dict.fromkeys(errors)),
        "warnings": warnings,
    })
    return base


def check_suite(
    paths: Sequence[str],
    *,
    target_soc: str,
    cann_version: str,
    expected_direct: Optional[int] = None,
    expected_single_gap: Optional[int] = None,
) -> Dict[str, Any]:
    """Inspect a migration suite without importing or executing its sources."""

    if not target_soc or not target_soc.strip():
        raise SuiteCheckError("target_soc must be non-empty")
    if not cann_version or not cann_version.strip():
        raise SuiteCheckError("cann_version must be non-empty")
    for label, value in (
        ("expected_direct", expected_direct),
        ("expected_single_gap", expected_single_gap),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise SuiteCheckError("%s must be a non-negative integer" % label)

    files = _discover_case_files(paths)
    cases = [
        _inspect_case(
            path,
            target_soc=target_soc,
            cann_version=cann_version,
        )
        for path in files
    ]
    counts = {
        "direct_migration_candidate": sum(
            item["category"] == "direct_migration_candidate" for item in cases
        ),
        "single_gap_candidate": sum(
            item["category"] == "single_gap_candidate" for item in cases
        ),
        "blocked": sum(item["category"] == "blocked" for item in cases),
    }
    expectation_errors = []
    if expected_direct is not None and counts["direct_migration_candidate"] != expected_direct:
        expectation_errors.append(
            "expected %d direct candidates, found %d"
            % (expected_direct, counts["direct_migration_candidate"])
        )
    if (
        expected_single_gap is not None
        and counts["single_gap_candidate"] != expected_single_gap
    ):
        expectation_errors.append(
            "expected %d single-gap candidates, found %d"
            % (expected_single_gap, counts["single_gap_candidate"])
        )
    if (
        expected_direct is not None
        and expected_single_gap is not None
        and len(cases) != expected_direct + expected_single_gap
    ):
        expectation_errors.append(
            "expected %d total cases, found %d"
            % (expected_direct + expected_single_gap, len(cases))
        )
    status = "pass" if not expectation_errors and not counts["blocked"] else "fail"
    return {
        "schema_version": 1,
        "producer": "numpilot-suite-check-v1",
        "status": status,
        "execution_performed": False,
        "runtime_capability_checked": False,
        "numpy_version": np.__version__,
        "target": {"soc": target_soc, "cann_version": cann_version},
        "expectations": {
            "direct_migration_candidates": expected_direct,
            "single_gap_candidates": expected_single_gap,
        },
        "counts": counts,
        "expectation_errors": expectation_errors,
        "cases": cases,
        "limitations": [
            "this check never executes build_inputs or forward",
            "direct candidates still require exact-wheel strict-NPU provenance and numerical verification",
            "single-gap candidates still require CANNBot generation, AsNumpy integration and six-gate qualification",
        ],
    }


def write_suite_report(path: str, report: Mapping[str, Any]) -> None:
    """Atomically write a private suite report, refusing unsafe targets."""

    target = Path(path).absolute()
    if os.path.lexists(target) and (target.is_symlink() or not target.is_file()):
        raise SuiteCheckError("suite report target is not a regular file: %s" % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = -1
    temporary = ""
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix="." + target.name + "-", suffix=".tmp", dir=str(target.parent)
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = ""
        os.chmod(target, 0o600)
    except OSError as exc:
        raise SuiteCheckError(
            "cannot write suite report %s: %s" % (target, type(exc).__name__)
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
