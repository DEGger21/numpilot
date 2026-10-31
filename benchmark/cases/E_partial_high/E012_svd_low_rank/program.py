"""E012_svd_low_rank: Truncated SVD reconstruction.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Positive-definite fixtures keep decompositions well conditioned.
The compute path contains several independently missing linear algebra APIs.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(512)
    base = rng.standard_normal((8, 8))
    identity = np.eye(8)
    matrix = base.T @ base + identity
    return {
        'a': matrix,
        'b': rng.standard_normal((8, 3)),
        'identity': identity,
    }


def forward(a, b, identity):
    u, singular, vh = np.linalg.svd(a, full_matrices=False)
    diagonal = np.diag(singular[:3])
    left = np.take(u, [0, 1, 2], axis=1)
    right = np.take(vh, [0, 1, 2], axis=0)
    approximation = left @ diagonal @ right
    error = np.subtract(a, approximation)
    rank = np.linalg.matrix_rank(approximation)
    quantiles = np.quantile(error, [0.1, 0.5, 0.9])
    retained = np.take(singular, [0, 1, 2])
    singular_gaps = np.diff(singular)
    return {
        "retained": retained,
        "singular_gaps": singular_gaps,
        'out': approximation,
        'singular': singular,
        'error': error,
        'rank': np.asarray(rank),
        'quantiles': quantiles,
    }
