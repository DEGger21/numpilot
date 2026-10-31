"""E013_least_squares: Least squares with pseudoinverse cross-check.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Positive-definite fixtures keep decompositions well conditioned.
The compute path contains several independently missing linear algebra APIs.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(513)
    base = rng.standard_normal((8, 8))
    identity = np.eye(8)
    matrix = base.T @ base + identity
    return {
        'a': matrix,
        'b': rng.standard_normal((8, 3)),
        'identity': identity,
    }


def forward(a, b, identity):
    solution, residuals, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    inverse = np.linalg.pinv(a)
    alternative = inverse @ b
    error = np.subtract(solution, alternative)
    condition = np.linalg.cond(a)
    rank_array = np.asarray(rank)
    order = np.argsort(singular)
    sorted_singular = np.take(singular, order)
    component_order = np.argsort(solution, axis=0)
    return {
        "component_order": component_order,
        'solution': solution,
        'residuals': residuals,
        'rank': rank_array,
        'singular': sorted_singular,
        'error': error,
        'condition': np.asarray(condition),
    }
