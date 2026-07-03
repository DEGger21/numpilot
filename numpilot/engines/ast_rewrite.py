"""默认迁移引擎:确定性 AST 改写。基于覆盖集与卸载方案泛化改写。"""
import ast

from numpilot.contracts import OffloadPlan
from numpilot.engines.base import MigrationEngine


class _Rewriter(ast.NodeTransformer):
    def __init__(self, np_aliases, npu_ops):
        self.np_aliases = np_aliases        # {"np"}
        self.npu_ops = npu_ops              # set[str], 需改到 ap 的算子
        self.used_ap = False

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
