"""默认迁移引擎:确定性 AST 改写。基于覆盖集与卸载方案泛化改写。"""
import ast
from dataclasses import dataclass

from numpilot.contracts import OffloadPlan
from numpilot.engines.base import MigrationEngine


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


def _depends_on_ap(node, tainted, ap_modules):
    return any(
        _is_direct_ap_call(item, ap_modules)
        or (
            isinstance(item, ast.Name)
            and isinstance(item.ctx, ast.Load)
            and item.id in tainted
        )
        for item in ast.walk(node)
    )


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
        elif isinstance(node, _CONTROL_FLOW_NODES):
            violations.append(
                f"L{getattr(node, 'lineno', 0)}: forward 包含未建模控制流/dunder "
                f"{type(node).__name__}"
            )

    # 对当前 straight-line whole-forward MVP 做最小 use-def 证明：所有设备调用都
    # 必须被消费，且每个返回值至少依赖一个设备调用。这样运行一个 NPU decoy 后
    # 返回预计算 host 全局或原输入，不能通过数值与 runtime-count 门禁。
    tainted = set()
    returns = 0
    for statement in forward.body:
        if _is_input_upload(statement, ap_modules, np_modules):
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
            if value is not None and _depends_on_ap(value, tainted, ap_modules):
                tainted.update(names)
            else:
                tainted.difference_update(names)
            continue
        if isinstance(statement, ast.Return):
            returns += 1
            if statement.value is None or not _depends_on_ap(
                statement.value, tainted, ap_modules
            ):
                violations.append(
                    f"L{getattr(statement, 'lineno', 0)}: forward 返回值不依赖任何直接 ap.* 结果"
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
    np_aliases = AstRewriteEngine._np_aliases(tree)
    npu_ops = {name for name, place in plan.placement.items() if place == "NPU"}
    forwards = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    if len(forwards) != 1:
        return {}

    counts = {}

    class _ExpectedVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in np_aliases
                and func.attr in npu_ops
            ):
                name = func.attr
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
            self.generic_visit(node)

    _ExpectedVisitor().visit(forwards[0])
    return counts


class _Rewriter(ast.NodeTransformer):
    def __init__(self, np_aliases, npu_ops):
        self.np_aliases = np_aliases        # {"np"}
        self.npu_ops = npu_ops              # set[str], 需改到 ap 的算子
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

    def _input_uploads(self, node):
        """为具名参数生成幂等上传:NumPy ndarray 上传,设备数组原样保留。"""
        np_alias = sorted(self.np_aliases)[0]
        args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        return [self._input_upload(arg.arg, np_alias) for arg in args]

    @staticmethod
    def _input_upload(name, np_alias):
        return ast.If(
            test=ast.Call(
                func=ast.Name(id="isinstance", ctx=ast.Load()),
                args=[
                    ast.Name(id=name, ctx=ast.Load()),
                    ast.Attribute(
                        value=ast.Name(id=np_alias, ctx=ast.Load()),
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
                                value=ast.Name(id="ap", ctx=ast.Load()),
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
        # relu 模式:np.maximum(x, 0) / np.maximum(0, x) → ap.relu(x)
        # asnumpy 的二元算子要求两侧都是 NPUArray,不接受标量 0;而 relu 在覆盖集内,
        # 语义等价且真机可用。故把这一 numpy 惯用法识别改写为 ap.relu。
        relu = self._as_relu(node)
        if relu is not None:
            return relu
        self.generic_visit(node)
        return node

    def _as_relu(self, node):
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "maximum"
            and "maximum" in self.npu_ops
            and isinstance(func.value, ast.Name)
            and func.value.id in self.np_aliases
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
                func=ast.Attribute(value=ast.Name(id="ap", ctx=ast.Load()),
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

    def visit_Attribute(self, node):
        if not self._in_forward:
            return node
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.np_aliases
            and node.attr in self.npu_ops
        ):
            self._mark_ap_used()
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id="ap", ctx=ast.Load()),
                              attr=node.attr, ctx=node.ctx),
                node,
            )
        return node


class AstRewriteEngine(MigrationEngine):
    name = "ast"

    def rewrite(self, source: str, plan: OffloadPlan) -> str:
        tree = ast.parse(source)
        np_aliases = self._np_aliases(tree)
        npu_ops = {op for op, place in plan.placement.items() if place == "NPU"}
        rewriter = _Rewriter(np_aliases, npu_ops)
        new_tree = rewriter.visit(tree)
        ast.fix_missing_locations(new_tree)
        if rewriter.used_ap:
            self._insert_ap_import(new_tree)
        return ast.unparse(new_tree)

    @staticmethod
    def _np_aliases(tree):
        aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "numpy":
                        aliases.add(a.asname or "numpy")
        return aliases or {"np"}

    @staticmethod
    def _insert_ap_import(tree):
        imp = ast.Import(names=[ast.alias(name="asnumpy", asname="ap")])
        last_import_idx = -1
        for i, node in enumerate(tree.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                last_import_idx = i
        tree.body.insert(last_import_idx + 1, imp)
        ast.fix_missing_locations(tree)
