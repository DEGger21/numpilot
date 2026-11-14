"""D1008_tile_calibration: Tile a learned calibration row over batches.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
tile materializes six calibration rows.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(408)
    return {
        'x': rng.standard_normal((6, 10), dtype=np.float32),
        'row': rng.standard_normal((1, 10), dtype=np.float32),
    }


def forward(x, row):
    absolute_1 = np.absolute(row)
    gain = np.add(absolute_1, 0.5)
    mean_2 = np.mean(gain, axis=1, keepdims=True)
    gain = np.divide(gain, mean_2)
    calibration = np.tile(gain, (6, 1))
    corrected = np.multiply(x, calibration)
    mean_3 = np.mean(corrected, axis=1, keepdims=True)
    centered = np.subtract(corrected, mean_3)
    output = np.tanh(centered)
    square_4 = np.square(output)
    energy = np.sum(square_4, axis=1)
    return {
        'out': output,
        'energy': energy,
    }
