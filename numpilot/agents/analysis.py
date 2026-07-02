"""分析智能体:AST 静态解析,提取 numpy 算子清单并比对覆盖集。"""
import ast
from math import prod

from numpilot.contracts import AnalysisResult, OpUse
from numpilot.coverage import is_supported


class _OpVisitor(ast.NodeVisitor):
    def __init__(self, np_aliases):
        self.np_aliases = np_aliases      # {"np", "numpy"}
        self.op_uses = []

    def visit_Call(self, node):
        name = self._numpy_func_name(node.func)
        if name is not None:
            self.op_uses.append(
                OpUse(
                    name=name,
                    lineno=getattr(node, "lineno", 0),
                    supported=is_supported(name),
                    est_elements=self._est_elements(name, node),
                )
            )
        self.generic_visit(node)

    def _numpy_func_name(self, func):
        # np.matmul(...) -> "matmul"; np.fft.fft(...) -> "fft"(取最后一段, 命名空间不在覆盖集自然判缺口)
        if isinstance(func, ast.Attribute):
            root = func
            while isinstance(root.value, ast.Attribute):
                root = root.value
            if isinstance(root.value, ast.Name) and root.value.id in self.np_aliases:
                return func.attr
        return None

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


class AnalysisAgent:
    def analyze(self, source: str, source_path: str = "<mem>") -> AnalysisResult:
        tree = ast.parse(source)
        np_aliases = self._collect_numpy_aliases(tree)
        visitor = _OpVisitor(np_aliases)
        visitor.visit(tree)
        unsupported = sorted({op.name for op in visitor.op_uses if not op.supported})
        return AnalysisResult(
            source_path=source_path,
            op_uses=visitor.op_uses,
            unsupported=unsupported,
            migratable=(len(unsupported) == 0 and len(visitor.op_uses) > 0),
        )

    @staticmethod
    def _collect_numpy_aliases(tree):
        aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "numpy":
                        aliases.add(a.asname or "numpy")
        return aliases or {"np", "numpy"}
