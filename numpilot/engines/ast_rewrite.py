"""默认迁移引擎:确定性 AST 改写。基于覆盖集与卸载方案泛化改写。"""
import ast

from numpilot.contracts import OffloadPlan
from numpilot.engines.base import MigrationEngine


def count_ap_calls(source: str) -> dict:
    """统计迁移后源码中 `ap.<op>(...)` 的调用点计数 {op: n}。

    引擎无关:AST / LLM 引擎的产物都能用它得到"真的替换到 ap 的算子清单"。
    仅计入 ap 别名的属性调用(如 import asnumpy as ap 后的 ap.matmul(...))。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    ap_aliases = {
        a.asname or a.name
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for a in node.names if a.name == "asnumpy"
    } or {"ap"}
    counts = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ap_aliases
        ):
            counts[node.func.attr] = counts.get(node.func.attr, 0) + 1
    return counts


class _Rewriter(ast.NodeTransformer):
    def __init__(self, np_aliases, npu_ops):
        self.np_aliases = np_aliases        # {"np"}
        self.npu_ops = npu_ops              # set[str], 需改到 ap 的算子
        self.used_ap = False

    def visit_Call(self, node):
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
        self.used_ap = True
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
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.np_aliases
            and node.attr in self.npu_ops
        ):
            self.used_ap = True
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
