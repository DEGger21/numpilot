"""Deterministic AsNumpy targets for Python operators on ndarray values.

The analyzer records Python dunders as unsupported because the strict AsNumpy
surface does not accept ndarray operators directly.  Most of them still have a
known elementwise AsNumpy equivalent; naming it in the Planner taskbook keeps
the Code Migrator from guessing.  Operators without a registered equivalent
stay ``None`` and fail closed in the taskbook.
"""

from __future__ import annotations


PYTHON_OPERATOR_TARGETS = {
    # Binary arithmetic.
    "Add": "add",
    "Sub": "subtract",
    "Mult": "multiply",
    "Div": "divide",
    "Pow": "power",
    "MatMult": "matmul",
    "Mod": "remainder",
    "FloorDiv": "floor_divide",
    # Unary arithmetic.
    "USub": "negative",
    "UAdd": "positive",
    # Elementwise comparisons (single-op comparisons only).
    "Lt": "less",
    "LtE": "less_equal",
    "Gt": "greater",
    "GtE": "greater_equal",
    "Eq": "equal",
    "NotEq": "not_equal",
}

_PREFIX = "python_operator."


def target_for_operator(name: str) -> str | None:
    """Return the canonical AsNumpy leaf for a ``python_operator.*`` name.

    The return value is unprefixed (``"add"``), matching the alias targets in
    the gap inventory.  The translation taskbook prefixes it with
    ``asnumpy.``.
    """

    if not name.startswith(_PREFIX):
        return None
    return PYTHON_OPERATOR_TARGETS.get(name[len(_PREFIX):])
