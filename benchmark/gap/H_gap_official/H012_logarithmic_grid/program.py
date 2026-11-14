"""H012_logarithmic_grid: Two logarithmically spaced grids.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Composition reference; NumPy >= 2.0.
logspace(a,b,n,base=c) = power(c,linspace(a,b,n)).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    return {
        'start': np.array(-2.0),
        'stop': np.array(2.0),
    }


def forward(start, stop):
    decades = np.logspace(start, stop, 17, base=10.0)
    octaves = np.logspace(start, stop, 17, base=2.0)
    return {
        'decades': decades,
        'octaves': octaves,
    }
