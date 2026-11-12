"""D1001_signed_compression: Two signed cube-root feature branches.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
cbrt appears twice, with float32 tensors of identical shape.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(401)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }


def forward(x, y):
    mean_1 = np.mean(x, axis=1, keepdims=True)
    x_center = np.subtract(x, mean_1)
    mean_2 = np.mean(y, axis=1, keepdims=True)
    y_center = np.subtract(y, mean_2)
    absolute_3 = np.absolute(x_center)
    maximum_4 = np.maximum(absolute_3, 0.25)
    x_scaled = np.divide(x_center, maximum_4)
    absolute_5 = np.absolute(y_center)
    maximum_6 = np.maximum(absolute_5, 0.25)
    y_scaled = np.divide(y_center, maximum_6)
    left = np.cbrt(x_scaled)
    right = np.cbrt(y_scaled)
    joint = np.add(left, right)
    out = np.tanh(joint)
    square_7 = np.square(out)
    energy = np.mean(square_7, axis=1)
    return {
        'out': out,
        'energy': energy,
    }
