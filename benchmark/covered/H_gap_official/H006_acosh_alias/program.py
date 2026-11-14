"""H006_acosh_alias: Inverse hyperbolic cosine on its real domain.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Alias reference; NumPy >= 2.0.
np.acosh(x) = np.arccosh(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    distance = np.linspace(0, 8, 24, dtype=np.float32).reshape(4, 6)
    return {
        'x': 1 + distance,
        'y': 1 + distance * distance,
    }


def forward(x, y):
    first = np.acosh(x)
    second = np.acosh(y)
    return {
        'out': first + second,
        'first': first,
    }
