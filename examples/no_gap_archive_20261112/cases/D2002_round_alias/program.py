"""D2002_round_alias: Quantized normalized activation.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Equivalent: np.round(normalized, decimals=2) -> np.around(normalized, decimals=2).
Only the named alias/composition is absent from the pinned export catalog.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(452)
    return {
        'x': rng.standard_normal((8, 13), dtype=np.float32),
        'y': rng.standard_normal((8, 13), dtype=np.float32),
    }


def forward(x, y):
    mean_1 = np.mean(x, axis=1, keepdims=True)
    center = np.subtract(x, mean_1)
    square_2 = np.square(center)
    mean_3 = np.mean(square_2, axis=1, keepdims=True)
    add_4 = np.add(mean_3, 1e-05)
    scale = np.sqrt(add_4)
    normalized = np.divide(center, scale)
    quantized = np.round(normalized, decimals=2)
    residual = np.subtract(quantized, normalized)
    return {
        'out': quantized,
        'error': residual,
        'reference': y,
    }
