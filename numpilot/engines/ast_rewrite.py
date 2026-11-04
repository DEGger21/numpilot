"""默认迁移引擎:确定性 AST 改写。基于覆盖集与卸载方案泛化改写。"""
import ast
from dataclasses import dataclass

from numpilot.agents.analysis import (
    numpy_bindings_for_forward,
    resolve_numpy_function,
)
from numpilot.contracts import OffloadPlan
from numpilot.engines.base import MigrationEngine
from numpilot.operator_specs import collect_static_operator_specs


_SAFE_ARRAY_METADATA = frozenset({
    "dtype", "itemsize", "nbytes", "ndim", "shape", "strides",
})
_SAFE_NUMPY_STATIC_VALUES = frozenset({
    "bool", "bool_", "e", "euler_gamma", "float16", "float32", "float64",
    "inf", "int8", "int16", "int32", "int64", "nan", "ndarray", "pi",
    "uint8", "uint16", "uint32", "uint64",
})
_CONTROL_FLOW_NODES = (
    ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With,
    ast.AsyncWith, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.BoolOp, ast.Subscript,
)
if hasattr(ast, "Match"):  # ast.Match was added in Python 3.10.
    _CONTROL_FLOW_NODES += (ast.Match,)


@dataclass(frozen=True)
class ForwardSurfaceAudit:
    ap_calls: dict
    violations: tuple

    @property
    def ok(self):
        return not self.violations


def _root_name(node):
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_direct_ap_call(node, ap_modules):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in ap_modules
    )


def _is_input_upload(statement, ap_modules, np_modules):
    """识别 AST 引擎唯一允许的边界控制流。"""
    if not isinstance(statement, ast.If) or statement.orelse:
        return False
    test = statement.test
    if not (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Name)
        and test.func.id == "isinstance"
        and len(test.args) == 2
        and isinstance(test.args[0], ast.Name)
        and isinstance(test.args[1], ast.Attribute)
        and _root_name(test.args[1]) in np_modules
        and test.args[1].attr == "ndarray"
        and len(statement.body) == 1
        and isinstance(statement.body[0], ast.Assign)
    ):
        return False
    assignment = statement.body[0]
    value = assignment.value
    return (
        len(assignment.targets) == 1
        and isinstance(assignment.targets[0], ast.Name)
        and assignment.targets[0].id == test.args[0].id
        and isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "from_numpy"
        and isinstance(value.func.value, ast.Attribute)
        and isinstance(value.func.value.value, ast.Name)
        and value.func.value.value.id in ap_modules
        and value.func.value.attr == "ndarray"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == test.args[0].id
    )


def _assigned_names(node):
    return {
        item.id for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
    }


def _fully_depends_on_ap(node, tainted, ap_modules):
    """Require every returned data leaf to come from a direct device call."""

    if _is_direct_ap_call(node, ap_modules):
        return True
    if isinstance(node, ast.Name):
        return node.id in tainted
    if isinstance(node, ast.Attribute) and node.attr in _SAFE_ARRAY_METADATA:
        return _fully_depends_on_ap(node.value, tainted, ap_modules)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return bool(node.elts) and all(
            _fully_depends_on_ap(item, tainted, ap_modules)
            for item in node.elts
        )
    if isinstance(node, ast.Dict):
        return bool(node.values) and all(
            _fully_depends_on_ap(item, tainted, ap_modules)
            for item in node.values
        )
    return False


def _fully_depends_on_device(node, computed, uploaded, ap_modules):
    """Every returned leaf is device-computed or an uploaded-input passthrough."""

    if _is_direct_ap_call(node, ap_modules):
        return True
    if isinstance(node, ast.Name):
        return node.id in computed or node.id in uploaded
    if isinstance(node, ast.Attribute) and node.attr in _SAFE_ARRAY_METADATA:
        return _fully_depends_on_device(
            node.value, computed, uploaded, ap_modules
        )
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return bool(node.elts) and all(
            _fully_depends_on_device(item, computed, uploaded, ap_modules)
            for item in node.elts
        )
    if isinstance(node, ast.Dict):
        return bool(node.values) and all(
            _fully_depends_on_device(item, computed, uploaded, ap_modules)
            for item in node.values
        )
    return False


def _has_device_derived_leaf(node, computed, ap_modules):
    """At least one returned leaf must depend on a direct device computation."""

    if _is_direct_ap_call(node, ap_modules):
        return True
    if isinstance(node, ast.Name):
        return node.id in computed
    if isinstance(node, ast.Attribute) and node.attr in _SAFE_ARRAY_METADATA:
        return _has_device_derived_leaf(node.value, computed, ap_modules)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(
            _has_device_derived_leaf(item, computed, ap_modules)
            for item in node.elts
        )
    if isinstance(node, ast.Dict):
        return any(
            _has_device_derived_leaf(item, computed, ap_modules)
            for item in node.values
        )
    return False


def audit_forward_surface(source: str) -> ForwardSurfaceAudit:
    """Fail-closed 审计迁移后 ``forward`` 的静态执行表面。

    当前 whole-forward MVP 只允许直接 ``ap.<op>``、入口上传
    ``ap.ndarray.from_numpy`` 和 ``isinstance``。残留 NumPy、host escape、动态
    导入/反射、Python ndarray dunder 或未知 helper 都不能形成 strict-NPU 证据。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return ForwardSurfaceAudit({}, (f"迁移产物语法错误: {error}",))

    forwards = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    if len(forwards) != 1:
        return ForwardSurfaceAudit(
            {}, (f"迁移产物必须恰有一个顶层同步 forward，实际 {len(forwards)} 个",)
        )
    forward = forwards[0]
    np_modules = set()
    numpy_functions = set()
    ap_modules = set()
    for node in list(tree.body) + list(ast.walk(forward)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy":
                    np_modules.add(alias.asname or "numpy")
                if alias.name == "asnumpy":
                    ap_modules.add(alias.asname or "asnumpy")
        elif isinstance(node, ast.ImportFrom) and node.module == "numpy":
            numpy_functions.update(alias.asname or alias.name for alias in node.names)
    ap_modules = ap_modules or {"ap"}
    np_modules = np_modules or {"np", "numpy"}

    calls = {}
    violations = []
    host_methods = {"to_numpy", "numpy", "asnumpy", "tolist", "item", "__array__"}
    dynamic_names = {"__import__", "eval", "exec", "compile"}
    allowed_names = {"isinstance"}

    arguments = (
        list(forward.args.posonlyargs)
        + list(forward.args.args)
        + list(forward.args.kwonlyargs)
    )
    local_names = {argument.arg for argument in arguments} | _assigned_names(forward)
    call_function_attributes = {
        id(node.func)
        for node in ast.walk(forward)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    for node in ast.walk(forward):
        if isinstance(node, ast.Call):
            func = node.func
            root = _root_name(func)
            line = getattr(node, "lineno", 0)
            if isinstance(func, ast.Attribute):
                if root in np_modules:
                    violations.append(f"L{line}: forward 残留 host NumPy 调用 {root}.{func.attr}")
                    continue
                if func.attr in host_methods:
                    violations.append(f"L{line}: forward 包含 host escape .{func.attr}()")
                    continue
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id in ap_modules
                ):
                    calls[func.attr] = calls.get(func.attr, 0) + 1
                    continue
                if (
                    func.attr == "from_numpy"
                    and isinstance(func.value, ast.Attribute)
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id in ap_modules
                    and func.value.attr == "ndarray"
                ):
                    continue
                violations.append(
                    f"L{line}: forward 包含未审计的方法/helper 调用 {ast.unparse(func)}"
                )
                continue
            if isinstance(func, ast.Name):
                if func.id in numpy_functions:
                    violations.append(f"L{line}: forward 残留 NumPy 导入函数 {func.id}()")
                elif func.id in dynamic_names:
                    violations.append(f"L{line}: forward 使用动态执行 {func.id}()")
                elif func.id == "getattr" and node.args and _root_name(node.args[0]) in ap_modules:
                    violations.append(f"L{line}: forward 动态解析 AsNumpy API，无法审计")
                elif func.id not in allowed_names:
                    violations.append(f"L{line}: forward 调用未审计 helper {func.id}()")
        elif isinstance(node, ast.BinOp):
            violations.append(
                f"L{getattr(node, 'lineno', 0)}: forward 残留 Python 二元 dunder "
                f"{type(node.op).__name__}"
            )
        elif isinstance(node, ast.Compare):
            violations.append(
                f"L{getattr(node, 'lineno', 0)}: forward 残留 Python 比较 dunder"
            )
        elif isinstance(node, ast.UnaryOp) and not isinstance(node.operand, ast.Constant):
            violations.append(
                f"L{getattr(node, 'lineno', 0)}: forward 残留 Python 一元 dunder "
                f"{type(node.op).__name__}"
            )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append(
                f"L{getattr(node, 'lineno', 0)}: forward 内部导入不在可信迁移表面"
            )
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if (
                node.id not in local_names
                and node.id not in np_modules
                and node.id not in ap_modules
                and node.id not in allowed_names
            ):
                violations.append(
                    f"L{getattr(node, 'lineno', 0)}: forward 读取未审计模块全局 {node.id}"
                )
        elif isinstance(node, ast.Attribute):
            root = _root_name(node)
            if id(node) in call_function_attributes:
                continue
            if root in np_modules:
                allowed = node.attr in _SAFE_NUMPY_STATIC_VALUES
            elif root in ap_modules:
                allowed = node.attr == "ndarray"
            else:
                allowed = node.attr in _SAFE_ARRAY_METADATA
            if not allowed:
                violations.append(
                    f"L{getattr(node, 'lineno', 0)}: forward 读取未审计属性 .{node.attr}"
                )
        elif isinstance(node, _CONTROL_FLOW_NODES):
            violations.append(
                f"L{getattr(node, 'lineno', 0)}: forward 包含未建模控制流/dunder "
                f"{type(node).__name__}"
            )

    # 对当前 straight-line whole-forward MVP 做最小 use-def 证明：所有设备调用都
    # 必须被消费，且返回值只能来自设备计算，或来自已上传到设备的输入透传。这样
    # 运行一个 NPU decoy 后返回预计算 host 全局，不能通过数值与 runtime-count
    # 门禁；同时允许 benchmark 契约把某个输入原样作为 reference 输出返回。
    tainted = set()
    uploaded = set()
    returns = 0
    for statement in forward.body:
        if _is_input_upload(statement, ap_modules, np_modules):
            uploaded.add(statement.test.args[0].id)
            continue
        if isinstance(statement, ast.Expr):
            if isinstance(statement.value, ast.Constant) and isinstance(
                statement.value.value, str
            ):
                continue
            if any(_is_direct_ap_call(item, ap_modules) for item in ast.walk(statement)):
                violations.append(
                    f"L{getattr(statement, 'lineno', 0)}: ap.* 结果被丢弃，不能证明输出数据流"
                )
            continue
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            names = _assigned_names(statement)
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                violations.append(
                    f"L{getattr(statement, 'lineno', 0)}: forward 仅允许单一名称赋值"
                )
                tainted.difference_update(names)
                continue
            if value is not None and _fully_depends_on_ap(
                value, tainted, ap_modules
            ):
                tainted.update(names)
            else:
                tainted.difference_update(names)
            continue
        if isinstance(statement, ast.Return):
            returns += 1
            if statement.value is None or not _fully_depends_on_device(
                statement.value, tainted, uploaded, ap_modules
            ):
                violations.append(
                    f"L{getattr(statement, 'lineno', 0)}: forward 返回值叶子既不来自"
                    "直接 ap.* 结果，也不是已上传设备的 forward 入参透传"
                )
            elif calls and not _has_device_derived_leaf(
                statement.value, tainted, ap_modules
            ):
                violations.append(
                    f"L{getattr(statement, 'lineno', 0)}: forward 有 ap.* 计算但所有"
                    "返回值都只是输入透传，无法排除 NPU decoy"
                )
            continue
        if isinstance(statement, ast.If):
            if not _is_input_upload(statement, ap_modules, np_modules):
                violations.append(
                    f"L{getattr(statement, 'lineno', 0)}: 计算控制流未建模，无法证明 ap.* 数据流"
                )
            continue
        violations.append(
            f"L{getattr(statement, 'lineno', 0)}: forward 语句 {type(statement).__name__} 未建模"
        )
    if returns != 1:
        violations.append(f"forward 必须恰有一个可证明的数据流返回，实际 {returns} 个")

    return ForwardSurfaceAudit(calls, tuple(dict.fromkeys(violations)))


def count_ap_calls(source: str) -> dict:
    """统计迁移后源码中 `ap.<op>(...)` 的调用点计数 {op: n}。

    引擎无关:AST / LLM 引擎的产物都能用它得到"真的替换到 ap 的算子清单"。
    仅计入 ap 别名的属性调用(如 import asnumpy as ap 后的 ap.matmul(...))。
    """
    return audit_forward_surface(source).ap_calls


def expected_forward_ap_calls(source: str, plan: OffloadPlan) -> dict:
    """从原始 ``forward`` 和 plan 推导 strict MVP 允许的精确改写清单。"""
    tree = ast.parse(source)
    npu_ops = {name for name, place in plan.placement.items() if place == "NPU"}
    forwards = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    if len(forwards) != 1:
        return {}
    forward = forwards[0]
    numpy_bindings = numpy_bindings_for_forward(tree, forward)
    static_specs = collect_static_operator_specs(tree).specs
    argument_names = {
        item.arg
        for item in (
            list(forward.args.posonlyargs)
            + list(forward.args.args)
            + list(forward.args.kwonlyargs)
        )
    }
    shadowed = argument_names | _assigned_names(forward)
    domain_specs = {
        name: spec for name, spec in static_specs.items() if name not in shadowed
    }

    counts = {}

    class _ExpectedVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            resolved = resolve_numpy_function(node.func, numpy_bindings)
            if resolved is not None:
                name, qualified = resolved
            else:
                name, qualified = None, None
            if name in npu_ops and qualified == "numpy." + name:
                if (
                    name == "maximum"
                    and len(node.args) == 2
                    and not node.keywords
                    and (
                        _Rewriter._is_zero(node.args[0])
                        or _Rewriter._is_zero(node.args[1])
                    )
                ):
                    name = "relu"
                counts[name] = counts.get(name, 0) + 1
            elif isinstance(node.func, ast.Name):
                spec = domain_specs.get(node.func.id)
                if spec is not None and spec.name in npu_ops:
                    counts[spec.name] = counts.get(spec.name, 0) + 1
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            return None

        def visit_AsyncFunctionDef(self, node):
            return None

        def visit_ClassDef(self, node):
            return None

        def visit_Lambda(self, node):
            return None

    visitor = _ExpectedVisitor()
    for statement in forward.body:
        visitor.visit(statement)
    return counts


class _Rewriter(ast.NodeTransformer):
    def __init__(
        self,
        numpy_bindings,
        domain_specs,
        npu_ops,
        numpy_alias,
        ap_alias,
    ):
        self.numpy_bindings = numpy_bindings
        self.domain_specs = dict(domain_specs)
        self.npu_ops = npu_ops              # set[str], 需改到 ap 的算子
        self.numpy_alias = numpy_alias
        self.ap_alias = ap_alias
        self.used_ap = False
        self._in_forward = False
        self._current_forward_used_ap = False

    def visit_FunctionDef(self, node):
        """只迁移 forward 的函数体,并把 H2D 边界放在 forward 入口。"""
        if node.name != "forward":
            return node

        previous_in_forward = self._in_forward
        previous_forward_used_ap = self._current_forward_used_ap
        self._in_forward = True
        self._current_forward_used_ap = False
        node.body = [self.visit(stmt) for stmt in node.body]
        forward_used_ap = self._current_forward_used_ap
        self._in_forward = previous_in_forward
        self._current_forward_used_ap = previous_forward_used_ap

        if forward_used_ap:
            uploads = self._input_uploads(node)
            # 函数 docstring 必须仍是第一条语句,否则会丢失 __doc__ 语义。
            insert_at = 1 if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ) else 0
            node.body[insert_at:insert_at] = uploads
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        if self._in_forward:
            return node
        return self.generic_visit(node)

    def visit_Lambda(self, node):
        if self._in_forward:
            return node
        return self.generic_visit(node)

    def _input_uploads(self, node):
        """为具名参数生成幂等上传:NumPy ndarray 上传,设备数组原样保留。"""
        args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        return [self._input_upload(arg.arg) for arg in args]

    def _input_upload(self, name):
        return ast.If(
            test=ast.Call(
                func=ast.Name(id="isinstance", ctx=ast.Load()),
                args=[
                    ast.Name(id=name, ctx=ast.Load()),
                    ast.Attribute(
                        value=ast.Name(id=self.numpy_alias, ctx=ast.Load()),
                        attr="ndarray",
                        ctx=ast.Load(),
                    ),
                ],
                keywords=[],
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id=self.ap_alias, ctx=ast.Load()),
                                attr="ndarray",
                                ctx=ast.Load(),
                            ),
                            attr="from_numpy",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=name, ctx=ast.Load())],
                        keywords=[],
                    ),
                )
            ],
            orelse=[],
        )

    def _mark_ap_used(self):
        self.used_ap = True
        self._current_forward_used_ap = True

    def visit_Call(self, node):
        if not self._in_forward:
            return node
        if isinstance(node.func, ast.Name):
            spec = self.domain_specs.get(node.func.id)
            if spec is not None and spec.name in self.npu_ops:
                self.generic_visit(node)
                self._mark_ap_used()
                node.func = ast.copy_location(
                    ast.Attribute(
                        value=ast.Name(id=self.ap_alias, ctx=ast.Load()),
                        attr=spec.name,
                        ctx=ast.Load(),
                    ),
                    node.func,
                )
                return node
        resolved = resolve_numpy_function(node.func, self.numpy_bindings)
        if resolved is not None:
            name, qualified = resolved
        else:
            name, qualified = None, None
        if name not in self.npu_ops or qualified != "numpy." + str(name):
            self.generic_visit(node)
            return node
        # relu 模式:np.maximum(x, 0) / np.maximum(0, x) → ap.relu(x)
        # asnumpy 的二元算子要求两侧都是 NPUArray,不接受标量 0;而 relu 在覆盖集内,
        # 语义等价且真机可用。故把这一 numpy 惯用法识别改写为 ap.relu。
        relu = self._as_relu(node, name)
        if relu is not None:
            return relu
        self.generic_visit(node)
        self._mark_ap_used()
        node.func = ast.copy_location(
            ast.Attribute(
                value=ast.Name(id=self.ap_alias, ctx=ast.Load()),
                attr=name,
                ctx=ast.Load(),
            ),
            node.func,
        )
        return node

    def _as_relu(self, node, name):
        if not (
            name == "maximum"
            and "maximum" in self.npu_ops
        ):
            return None
        if len(node.args) != 2 or node.keywords:
            return None
        a, b = node.args
        if self._is_zero(b):
            x = a
        elif self._is_zero(a):
            x = b
        else:
            return None
        self._mark_ap_used()
        new_x = self.visit(x)                 # 递归改写内部(matmul/add 等)
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(value=ast.Name(id=self.ap_alias, ctx=ast.Load()),
                                   attr="relu", ctx=ast.Load()),
                args=[new_x], keywords=[],
            ),
            node,
        )

    @staticmethod
    def _is_zero(n):
        # 匹配字面量 0 / 0.0,排除 bool(False == 0 在 Python 里为真)
        return (
            isinstance(n, ast.Constant)
            and not isinstance(n.value, bool)
            and n.value == 0
        )


class AstRewriteEngine(MigrationEngine):
    name = "ast"

    def rewrite(self, source: str, plan: OffloadPlan) -> str:
        tree = ast.parse(source)
        forwards = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        ]
        if len(forwards) != 1:
            return ast.unparse(tree)
        numpy_bindings = numpy_bindings_for_forward(tree, forwards[0])
        static_specs = collect_static_operator_specs(tree).specs
        arguments = (
            list(forwards[0].args.posonlyargs)
            + list(forwards[0].args.args)
            + list(forwards[0].args.kwonlyargs)
        )
        shadowed = {item.arg for item in arguments} | _assigned_names(forwards[0])
        domain_specs = {
            name: spec for name, spec in static_specs.items() if name not in shadowed
        }
        npu_ops = {op for op, place in plan.placement.items() if place == "NPU"}
        numpy_alias = self._module_numpy_alias(tree, numpy_bindings)
        add_numpy_import = numpy_alias is None
        if add_numpy_import:
            numpy_alias = self._fresh_alias(tree, "_numpilot_numpy")
        ap_alias = self._fresh_alias(tree, "ap")
        rewriter = _Rewriter(
            numpy_bindings,
            domain_specs,
            npu_ops,
            numpy_alias,
            ap_alias,
        )
        new_tree = rewriter.visit(tree)
        ast.fix_missing_locations(new_tree)
        if rewriter.used_ap:
            if add_numpy_import:
                self._insert_import(new_tree, "numpy", numpy_alias)
            self._insert_import(new_tree, "asnumpy", ap_alias)
        return ast.unparse(new_tree)

    @staticmethod
    def _np_aliases(tree):
        forwards = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        ]
        if len(forwards) != 1:
            return set()
        bindings = numpy_bindings_for_forward(tree, forwards[0])
        return {
            name for name, qualified in bindings.items()
            if qualified == "numpy"
        }

    @staticmethod
    def _module_numpy_alias(tree, numpy_bindings):
        for statement in tree.body:
            if not isinstance(statement, ast.Import):
                continue
            for alias in statement.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                if numpy_bindings.get(local_name) == "numpy":
                    return local_name
        return None

    @staticmethod
    def _fresh_alias(tree, preferred):
        used = {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        used.update(
            node.arg for node in ast.walk(tree)
            if isinstance(node, ast.arg)
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                used.update(
                    alias.asname or alias.name.split(".", 1)[0]
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                used.update(alias.asname or alias.name for alias in node.names)
        candidate = preferred
        suffix = 2
        while candidate in used:
            candidate = "%s_%d" % (preferred, suffix)
            suffix += 1
        return candidate

    @staticmethod
    def _insert_import(tree, module, alias):
        imp = ast.Import(names=[ast.alias(name=module, asname=alias)])
        last_import_idx = -1
        for i, node in enumerate(tree.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                last_import_idx = i
        tree.body.insert(last_import_idx + 1, imp)
        ast.fix_missing_locations(tree)
