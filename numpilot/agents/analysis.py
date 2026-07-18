"""分析智能体：提取 ``forward`` NPU 岛中的 NumPy 算子并比对覆盖集。"""
import ast
from math import prod

from numpilot.contracts import AnalysisResult, OpUse
from numpilot.coverage import is_supported


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


class _OpVisitor(ast.NodeVisitor):
    def __init__(self, np_aliases):
        self.np_aliases = np_aliases      # {"np", "numpy"}
        self.op_uses = []

    def visit_Call(self, node):
        name = self._numpy_func_name(node.func)
        if name is not None:
            supported, reason = self._call_supported(name, node)
            self.op_uses.append(
                OpUse(
                    name=name,
                    lineno=getattr(node, "lineno", 0),
                    supported=supported,
                    est_elements=self._est_elements(name, node),
                    reason=reason,
                )
            )
        self.generic_visit(node)

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

    def _call_supported(self, name, node):
        if not is_supported(name):
            return False, "API 不在静态覆盖集"
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
        # build_inputs 和顶层展示逻辑按 MVP 约定保留在 CPU；只有 forward 属于
        # placement/capability 范围，避免报告出“构造输入也卸载到 NPU”的假证据。
        forwards = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "forward"
        ]
        for forward in forwards:
            for statement in forward.body:
                visitor.visit(statement)
        unsupported = sorted({op.name for op in visitor.op_uses if not op.supported})
        return AnalysisResult(
            source_path=source_path,
            op_uses=visitor.op_uses,
            unsupported=unsupported,
            migratable=(len(forwards) == 1 and not unsupported and bool(visitor.op_uses)),
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
