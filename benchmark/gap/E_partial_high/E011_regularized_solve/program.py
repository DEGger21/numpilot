"""E011_regularized_solve: Linear solve with factor diagnostics.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Positive-definite fixtures keep decompositions well conditioned.
The compute path contains several independently missing linear algebra APIs.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(511)
    base = rng.standard_normal((8, 8))
    identity = np.eye(8)
    matrix = base.T @ base + identity
    return {
        'a': matrix,
        'b': rng.standard_normal((8, 3)),
        'identity': identity,
    }


def forward(a, b, identity):
    regularized = np.add(a, identity * 0.25)
    solution = np.linalg.solve(regularized, b)
    factor = np.linalg.cholesky(regularized)
    diagonal = np.diag(factor)
    eigenvalues = np.linalg.eigvalsh(regularized)
    condition = np.linalg.cond(regularized)
    residual = regularized @ solution - b
    permutation = np.argsort(diagonal)
    selected = np.take(diagonal, permutation)
    rhs_order = np.argsort(solution, axis=0)
    return {
        "rhs_order": rhs_order,
        'solution': solution,
        'eigenvalues': eigenvalues,
        'condition': np.asarray(condition),
        'residual': residual,
        'diagonal': selected,
    }
