"""分析智能体：提取 ``forward`` NPU 岛中的 NumPy 算子并比对覆盖集。"""
import ast
import hashlib
from math import prod

from numpilot.capabilities import AnalysisProfileSet, CapabilityMatch
from numpilot.contracts import AnalysisResult, OpUse
from numpilot.coverage import is_supported
from numpilot.operator_specs import collect_static_operator_specs


_ARRAY_BINARY_OPS = frozenset({
    "add", "subtract", "multiply", "divide", "true_divide", "floor_divide",
    "matmul", "dot", "maximum", "minimum", "power", "float_power", "fmod",
    "mod", "remainder", "fmax", "fmin", "copysign", "heaviside", "hypot",
    "arctan2", "logical_and", "logical_or", "logical_xor", "equal", "not_equal",
    "greater", "greater_equal", "less", "less_equal",
})
_UNSUPPORTED_NUMPY_KEYWORDS = frozenset({
    "out", "where", "casting", "order", "subok", "signature", "extobj",
})
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
    def __init__(self, numpy_bindings, domain_specs=None, domain_errors=None):
        # Local name -> canonical NumPy object, e.g. {"np": "numpy",
        # "fft": "numpy.fft.fft"}. Bindings come only from real imports.
        self.numpy_bindings = numpy_bindings
        self.domain_specs = dict(domain_specs or {})
        self.domain_errors = dict(domain_errors or {})
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
        self.generic_visit(node)

    def visit_Import(self, node):
        self._unsupported_python_structure(node, "Import")

    def visit_ImportFrom(self, node):
        self._unsupported_python_structure(node, "ImportFrom")

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

    def _unsupported_python_operator(self, node, operator):
        self.op_uses.append(OpUse(
            name="python_operator." + operator,
            lineno=getattr(node, "lineno", 0),
            supported=False,
            reason="当前 NPUArray 未验证对应 Python dunder，需显式 np.* 调用",
        ))

    def _unsupported_python_structure(self, node, structure):
        self.op_uses.append(OpUse(
            name="python_structure." + structure,
            lineno=getattr(node, "lineno", 0),
            supported=False,
            reason="forward 内部 import 不在 strict NPU 可信执行表面",
        ))

    def _call_supported(self, name, qualified_name, node):
        # The frozen coverage snapshot describes the top-level AsNumpy
        # namespace. A matching leaf is not evidence that a nested API such
        # as numpy.linalg.matmul is exposed at the same path.
        if qualified_name != "numpy." + name or not is_supported(name):
            return False, _MISSING_API_REASON
        risky_keywords = sorted(
            keyword.arg for keyword in node.keywords
            if keyword.arg in _UNSUPPORTED_NUMPY_KEYWORDS
        )
        if risky_keywords:
            return False, "AsNumpy 未验证 NumPy 关键字: " + ", ".join(risky_keywords)
        if name in _ARRAY_BINARY_OPS:
            operands = list(node.args[:2])
            for keyword in node.keywords:
                if keyword.arg in {"x1", "x2"}:
                    operands.append(keyword.value)
            if any(self._is_literal_scalar(value) for value in operands):
                if name == "maximum" and self._is_relu_pattern(node):
                    return True, None
                return False, "当前 AsNumpy 二元 binding 未验证 Python 标量操作数"
        return True, None

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
            visitor = _OpVisitor(
                numpy_bindings,
                domain_specs=domain_specs,
                domain_errors=domain_errors,
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
