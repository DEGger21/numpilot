"""H010_standard_deviation: Population standard deviations with explicit axes.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Composition reference; NumPy >= 2.0.
std(x,axis=a) = sqrt(mean(square(x-mean(x,a,True)),axis=a)).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(810)
    return {
        'x': rng.standard_normal((9, 11), dtype=np.float32),
    }


def forward(x):
    row = np.std(x, axis=1)
    column = np.std(x, axis=0)
    return {
        'row': row,
        'column': column,
    }
