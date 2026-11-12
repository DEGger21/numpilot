"""H002_round_alias: Round-to-even decimal alias.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Alias reference; NumPy >= 2.0.
np.round(x, decimals=n) = np.around(x, decimals=n).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    return {
        "x": np.resize(np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32), (4_000_000,)),
        "y": np.resize(np.array([0.125, 0.375, 0.625, 0.875], dtype=np.float64), (4_000_000,)),
    }


def forward(x, y):
    integer = np.round(x)
    hundredths = np.round(y, decimals=2)
    return {
        'integer': integer,
        'hundredths': hundredths,
    }
