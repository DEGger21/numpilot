"""E008_robust_order_stats: Trimmed distribution and outlier positions.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
The compute path combines at least three missing APIs with a supported anchor.
Named intermediate outputs expose indexing, shape and numerical behavior.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(508)
    x = rng.standard_normal(96, dtype=np.float32)
    x[:5] = -20.0
    x[-5:] = 20.0
    x[20:25] = 0.5
    return {
        'x': x,
    }


def forward(x):
    bounded = np.clip(x, -8, 8)
    partitioned = np.partition(bounded, (8, bounded.size - 9))
    trimmed = partitioned[8:-8]
    median = np.median(trimmed)
    deviation = np.absolute(bounded - median)
    mad = np.median(deviation)
    indices = np.argwhere(deviation > 2 * mad)
    quartiles = np.percentile(trimmed, [25, 50, 75])
    unique = np.unique(partitioned)
    outlier_values = np.take(bounded, indices.ravel())
    outlier_order = np.argsort(outlier_values)
    return {
        "outlier_values": outlier_values,
        "outlier_order": outlier_order,
        'trimmed': trimmed,
        'outliers': indices,
        'quartiles': quartiles,
        'mad': np.asarray(mad),
        'support': unique,
    }
