"""确定性的 Program IR：把 ``forward`` 的调用与数据依赖抽成 DAG。

The architecture's Understanding stage carries three artifacts: the AST, a
program IR, and a DAG.  The AST is already produced by :mod:`ast`; this module
adds the other two in one pass over the top-level ``forward`` body:

* a node per input, call, constant and return;
* an edge for every argument that flows into a call or a returned value.

The IR is evidence, not a gate: it never decides that code is safe.  It gives
the planner and the report a stable, inspectable shape of the program without
executing user source.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass(frozen=True)
class IRNode:
    id: str
    kind: str  # "input" | "constant" | "call" | "return"
    label: str
    lineno: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "lineno": self.lineno,
        }


@dataclass(frozen=True)
class ProgramIR:
    """A small, serialisable data-flow graph for one ``forward``."""

    nodes: tuple = field(default_factory=tuple)
    edges: tuple = field(default_factory=tuple)  # ((src_id, dst_id), ...)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
        }


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class _Builder(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.env = {}
        self.counter = 0

    def _add(self, kind, label, lineno=0):
        self.counter += 1
        node = IRNode("n%d" % self.counter, kind, label, lineno)
        self.nodes.append(node)
        return node

    def _value_node(self, expression):
        """Return the node id produced by ``expression``, if it has one."""
        if isinstance(expression, ast.Name):
            return self.env.get(expression.id)
        if isinstance(expression, ast.Call):
            call = self._add_call(expression)
            return call
        return None

    def _add_call(self, node):
        label = _dotted(node.func) or "<call>"
        call = self._add("call", label, getattr(node, "lineno", 0))
        for argument in node.args:
            source = self._value_node(argument)
            if source is not None:
                self.edges.append((source.id, call.id))
        for keyword in node.keywords:
            source = self._value_node(keyword.value)
            if source is not None:
                self.edges.append((source.id, call.id))
        return call

    def visit_FunctionDef(self, node):
        if node.name != "forward":
            return
        arguments = list(node.args.posonlyargs) + list(node.args.args)
        for argument in arguments:
            source = self._add("input", argument.arg, getattr(node, "lineno", 0))
            self.env[argument.arg] = source
        for statement in node.body:
            self.visit(statement)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        produced = self._value_node(node.value)
        if produced is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.env[target.id] = produced

    def visit_AnnAssign(self, node):
        produced = self._value_node(node.value) if node.value is not None else None
        if produced is not None and isinstance(node.target, ast.Name):
            self.env[node.target.id] = produced

    def visit_Return(self, node):
        if node.value is None:
            return
        produced = self._value_node(node.value)
        if produced is None:
            return
        returned = self._add("return", "return", getattr(node, "lineno", 0))
        self.edges.append((produced.id, returned.id))

    def visit_Call(self, node):
        # Calls reached through generic traversal (conditions, expression
        # statements) still appear as nodes, just without a bound name.
        self._add_call(node)

    def generic_visit(self, node):
        # Keep walking expressions so calls inside branches and loops appear in
        # the graph; the IR stays a conservative superset of executed paths.
        super().generic_visit(node)


def build_program_ir(source: str, filename: str = "<program>") -> ProgramIR:
    """Build the IR for the top-level ``forward``; empty when absent."""
    tree = ast.parse(source, filename=filename)
    builder = _Builder()
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            builder.visit(statement)
    edges = tuple(dict.fromkeys(builder.edges))
    return ProgramIR(nodes=tuple(builder.nodes), edges=edges)
