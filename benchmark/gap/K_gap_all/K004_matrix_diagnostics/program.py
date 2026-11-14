"""K004_matrix_diagnostics: Several missing decompositions sharing an SPD matrix.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
The SPD fixture avoids singular-system exception paths.
Only missing linalg/shape/index APIs appear in the compute functions.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(1104)
    base = rng.normal(size=(9, 9))
    matrix = base.T @ base + np.eye(9) * 2
    return {
        'matrix': matrix,
        'rhs': rng.normal(size=(9, 3)),
    }


def cholesky_path(matrix):
    """Cholesky reconstruction exposes factor orientation mistakes."""
    factor = np.linalg.cholesky(matrix)
    diagonal = np.diagonal(factor)
    reconstruction = factor @ np.transpose(factor)
    condition = np.asarray(np.linalg.cond(factor))
    return factor, diagonal, reconstruction, condition


def solution_paths(matrix, rhs, factor):
    """Compare a full solve to two triangular solves."""
    solution = np.linalg.solve(matrix, rhs)
    intermediate = np.linalg.solve(np.tril(factor), rhs)
    triangular = np.linalg.solve(np.triu(np.transpose(factor)), intermediate)
    residual = matrix @ solution - rhs
    projected = np.take(solution, [0, 4, 8], axis=0)
    return solution, triangular, residual, projected


def svd_path(matrix):
    """Truncate singular components and retain exact matrix rank."""
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    diagonal = np.diag(singular[:4])
    left = np.take(u, [0, 1, 2, 3], axis=1)
    right = np.take(vh, [0, 1, 2, 3], axis=0)
    low_rank = left @ diagonal @ right
    rank = np.asarray(np.linalg.matrix_rank(low_rank))
    return u, singular, vh, low_rank, rank


def eigen_path(matrix):
    """Whiten with a symmetric eigensystem."""
    values, vectors = np.linalg.eigh(matrix)
    inverse_scale = np.diag(values ** -0.5)
    transform = vectors @ inverse_scale @ np.transpose(vectors)
    whitened = transform @ matrix @ transform
    condition = np.asarray(np.linalg.cond(whitened))
    return values, vectors, whitened, condition


def inverse_paths(matrix, rhs):
    """Pseudoinverse and least squares provide independent paths."""
    inverse = np.linalg.pinv(matrix)
    pseudo_solution = inverse @ rhs
    least_solution, residuals, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=None)
    least_residual = matrix @ least_solution - rhs
    return inverse, pseudo_solution, least_solution, least_residual


def structural_views(matrix):
    """Structural APIs create independently observable shape changes."""
    upper = np.triu(matrix, k=1)
    lower = np.tril(matrix, k=-1)
    diagonal = np.diagonal(matrix)
    repeated = np.repeat(diagonal[None, :], 3, axis=0)
    tiled = np.tile(repeated, (2, 1))
    blocks = np.block([[upper, lower], [lower, upper]])
    flattened = np.ravel(blocks)
    restored = np.reshape(flattened, blocks.shape)
    return upper, lower, tiled, restored


def forward(matrix, rhs):
    factor, diagonal, reconstruction, factor_condition = cholesky_path(matrix)
    solution, triangular, residual, projected = solution_paths(matrix, rhs, factor)
    u, singular, vh, low_rank, rank = svd_path(matrix)
    values, vectors, whitened, spectral_condition = eigen_path(matrix)
    inverse, pseudo_solution, least_solution, least_residual = inverse_paths(matrix, rhs)
    upper, lower, tiled_diagonal, blocks = structural_views(matrix)
    error_quantiles = np.quantile(residual, [0.1, 0.5, 0.9])
    row_medians = np.median(matrix, axis=1)
    return {
        'factor': factor,
        'diagonal': diagonal,
        'reconstruction': reconstruction,
        'factor_condition': factor_condition,
        'solution': solution,
        'triangular': triangular,
        'residual': residual,
        'projected': projected,
        'u': u,
        'singular': singular,
        'vh': vh,
        'low_rank': low_rank,
        'rank': rank,
        'values': values,
        'vectors': vectors,
        'whitened': whitened,
        'spectral_condition': spectral_condition,
        'inverse': inverse,
        'pseudo_solution': pseudo_solution,
        'least_solution': least_solution,
        'least_residual': least_residual,
        'upper': upper,
        'lower': lower,
        'tiled_diagonal': tiled_diagonal,
        'blocks': blocks,
        'error_quantiles': error_quantiles,
        'row_medians': row_medians,
    }
