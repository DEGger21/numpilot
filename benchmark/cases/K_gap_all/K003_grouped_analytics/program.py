"""K003_grouped_analytics: Grouped weighted observations and empirical distributions.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Unequal group sizes and nonuniform weights exercise variable-size contracts.
No stage relies on a supported compute API to hide a missing operation.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(1103)
    return {
        'keys': rng.integers(0, 13, 211, dtype=np.int32),
        'values': rng.normal(1, 2, 211),
        'weights': rng.uniform(0.1, 1.5, 211),
        'edges': np.linspace(-6, 8, 18),
    }


def aggregate(keys, values, weights):
    """Weighted per-key totals with explicit inverse indexing."""
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    totals = np.bincount(inverse, weights=values * weights)
    mass = np.bincount(inverse, weights=weights)
    means = totals / mass
    return unique, inverse, counts, totals, means


def rank_observations(keys, values):
    """Lexicographic ordering and stable value ranks."""
    order = np.lexsort((values, keys))
    sorted_keys = np.take(keys, order)
    sorted_values = np.take(values, order)
    value_order = np.argsort(values, kind="stable")
    ranks = np.argsort(value_order)
    return order, sorted_keys, sorted_values, ranks


def bin_values(values, weights, edges):
    """Cross-check two histogram representations."""
    histogram, returned_edges = np.histogram(values, bins=edges)
    bins = np.digitize(values, edges[1:-1])
    weighted = np.bincount(bins, weights=weights, minlength=edges.size - 1)
    insertions = np.searchsorted(returned_edges, values, side="right")
    return histogram, bins, weighted, insertions


def robust_summary(values):
    """Quantiles, an order-statistic trim, and endpoint samples."""
    quartiles = np.percentile(values, [25, 50, 75])
    median = np.asarray(np.median(values))
    partitioned = np.partition(values, (10, values.size - 11))
    trimmed = partitioned[10:-10]
    indices = np.argpartition(values, (0, values.size - 1))[[0, -1]]
    extrema = np.take(values, indices)
    return quartiles, median, trimmed, extrema


def compare_groups(keys, unique):
    """Set operations on distinct halves of the observation stream."""
    left = np.unique(keys[::2])
    right = np.unique(keys[1::2])
    common = np.intersect1d(left, right)
    only_left = np.setdiff1d(left, right)
    only_right = np.setdiff1d(right, left)
    union = np.union1d(unique, np.concatenate((left, right)))
    return common, only_left, only_right, union


def empirical_curve(values):
    """Quantile interpolation with derivatives and enclosed area."""
    levels = np.arange(0, 1.01, 0.1)
    quantiles = np.quantile(values, levels)
    grid = np.arange(0, 1.001, 0.025)
    resampled = np.interp(grid, levels, quantiles)
    slope = np.gradient(resampled, grid)
    area = np.asarray(np.trapezoid(resampled, grid))
    return resampled, slope, area, grid


def forward(keys, values, weights, edges):
    unique, inverse, counts, totals, means = aggregate(keys, values, weights)
    order, sorted_keys, sorted_values, ranks = rank_observations(keys, values)
    hist, bins, weighted_hist, insertions = bin_values(values, weights, edges)
    quartiles, median, trimmed, extrema = robust_summary(values)
    common, only_left, only_right, union = compare_groups(sorted_keys, unique)
    resampled, slope, area, grid = empirical_curve(values)
    repeated_means = np.take(means, inverse)
    residuals = values - repeated_means
    packed = np.packbits(residuals > 0, bitorder="big")
    return {
        'unique': unique,
        'counts': counts,
        'totals': totals,
        'means': means,
        'order': order,
        'sorted_keys': sorted_keys,
        'sorted_values': sorted_values,
        'ranks': ranks,
        'histogram': hist,
        'bins': bins,
        'weighted_histogram': weighted_hist,
        'insertions': insertions,
        'quartiles': quartiles,
        'median': median,
        'trimmed': trimmed,
        'extrema': extrema,
        'common': common,
        'only_left': only_left,
        'only_right': only_right,
        'union': union,
        'resampled': resampled,
        'slope': slope,
        'area': area,
        'grid': grid,
        'residuals': residuals,
        'packed': packed,
    }
