"""D1015_sorted_bin_indices: Calibrated signals mapped to sorted thresholds.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
searchsorted uses right insertion, including exact endpoint ties.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(415)
    return {
        'x': rng.standard_normal(83, dtype=np.float32),
        'bins': np.linspace(-2, 2, 13, dtype=np.float32),
    }


def forward(x, bins):
    scaled = np.multiply(x, 0.75)
    shifted = np.add(scaled, 0.2)
    bounded = np.clip(shifted, -2.0, 2.0)
    index = np.searchsorted(bins, bounded, side='right')
    magnitude = np.absolute(bounded)
    negative_1 = np.negative(magnitude)
    weight = np.exp(negative_1)
    weighted = np.multiply(bounded, weight)
    return {
        'index': index,
        'weighted': weighted,
        'weight': weight,
    }
