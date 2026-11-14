"""E006_histogram_cdf: Empirical distribution and bin lookup.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
The compute path combines at least three missing APIs with a supported anchor.
Named intermediate outputs expose indexing, shape and numerical behavior.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(506)
    x = rng.standard_normal(96, dtype=np.float32)
    x[::7] = -3.0
    x[::11] = 3.0
    x[1:4] = [-0.5, 0.0, 0.5]
    return {
        'x': x,
    }


def forward(x):
    bounded = np.clip(x, -3, 3)
    counts, edges = np.histogram(bounded, bins=12, range=(-3, 3))
    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    indices = np.digitize(bounded, edges[1:-1])
    mass = np.bincount(indices, minlength=12)
    cdf = np.cumsum(counts) / bounded.size
    median = np.interp(0.5, cdf, centers)
    quantiles = np.quantile(bounded, [0.1, 0.5, 0.9])
    bin_centers = np.convolve(edges, [0.5, 0.5], mode="valid")
    return {
        "bin_centers": bin_centers,
        'counts': counts,
        'widths': widths,
        'mass': mass,
        'cdf': cdf,
        'median': np.asarray(median),
        'quantiles': quantiles,
    }
