"""D2001_absolute_alias: Absolute-value alias inside a residual score.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Equivalent: np.abs(residual) -> np.absolute(residual).
Only the named alias/composition is absent from the pinned export catalog.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(451)
    return {
        'x': rng.standard_normal((8, 13), dtype=np.float32),
        'y': rng.standard_normal((8, 13), dtype=np.float32),
    }


def forward(x, y):
    residual = np.subtract(x, y)
    magnitude = np.abs(residual)
    bounded = np.minimum(magnitude, 1.0)
    square_1 = np.square(bounded)
    quadratic = np.multiply(square_1, 0.5)
    linear = np.subtract(magnitude, bounded)
    loss = np.add(quadratic, linear)
    bounded_residual = np.clip(residual, -1.0, 1.0)
    return {
        "bounded_residual": bounded_residual,
        'loss': loss,
        'row_loss': np.mean(loss, axis=1),
    }
