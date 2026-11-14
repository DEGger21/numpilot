"""E015_triangular_system: Cholesky triangular systems and reconstruction.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Positive-definite fixtures keep decompositions well conditioned.
The compute path contains several independently missing linear algebra APIs.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(515)
    base = rng.standard_normal((8, 8))
    identity = np.eye(8)
    matrix = base.T @ base + identity
    return {
        'a': matrix,
        'b': rng.standard_normal((8, 3)),
        'identity': identity,
    }


def forward(a, b, identity):
    factor = np.linalg.cholesky(a)
    lower = np.tril(factor)
    transpose_1 = np.transpose(factor)
    upper = np.triu(transpose_1)
    middle = np.linalg.solve(lower, b)
    solution = np.linalg.solve(upper, middle)
    reconstruction = lower @ upper
    error = np.subtract(reconstruction, a)
    diagonal = np.diagonal(lower)
    condition = np.linalg.cond(lower)
    row_order = np.argsort(diagonal)
    return {
        "row_order": row_order,
        'solution': solution,
        'factor': lower,
        'error': error,
        'diagonal': diagonal,
        'condition': np.asarray(condition),
    }
