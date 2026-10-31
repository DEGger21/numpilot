"""E007_grouped_samples: Grouping quantized observations.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
The compute path combines at least three missing APIs with a supported anchor.
Named intermediate outputs expose indexing, shape and numerical behavior.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(507)
    x = rng.standard_normal(96, dtype=np.float32)
    x[:16] = 0.0
    x[16:24] = 1.0
    x[-3:] = [-4.5, 3.5, 5.0]
    return {
        'x': x,
    }


def forward(x):
    labels = np.rint(x * 3).astype(np.int32)
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    totals = np.bincount(inverse, weights=x)
    means = totals / counts
    order = np.argsort(inverse, kind='stable')
    sorted_x = np.take(x, order)
    cuts = np.cumsum(counts)[:-1]
    groups = np.split(sorted_x, cuts)
    medians = np.array([np.median(group) for group in groups])
    largest_groups = np.argpartition(counts, -3)[-3:]
    representatives = np.take(means, largest_groups)
    return {
        "largest_groups": largest_groups,
        "representatives": representatives,
        'labels': unique,
        'means': means,
        'medians': medians,
        'counts': counts,
    }
