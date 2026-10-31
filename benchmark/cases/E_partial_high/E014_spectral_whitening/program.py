"""E014_spectral_whitening: Symmetric eigensystem whitening.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Positive-definite fixtures keep decompositions well conditioned.
The compute path contains several independently missing linear algebra APIs.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(514)
    base = rng.standard_normal((8, 8))
    identity = np.eye(8)
    matrix = base.T @ base + identity
    return {
        'a': matrix,
        'b': rng.standard_normal((8, 3)),
        'identity': identity,
    }


def forward(a, b, identity):
    values, vectors = np.linalg.eigh(a)
    safe = np.maximum(values, 1e-05)
    diagonal = np.diag(safe ** (-0.5))
    transposed = np.transpose(vectors)
    whitener = vectors @ diagonal @ transposed
    white = whitener @ a @ whitener
    rank = np.linalg.matrix_rank(white)
    condition = np.linalg.cond(white)
    diagonal_white = np.diagonal(white)
    eigen_gaps = np.diff(values)
    return {
        "eigen_gaps": eigen_gaps,
        'whitener': whitener,
        'white': white,
        'diagonal': diagonal_white,
        'rank': np.asarray(rank),
        'condition': np.asarray(condition),
    }
