"""H013_geometric_grid: Positive geometric scale sequence.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Composition reference; NumPy >= 2.0.
For positive inputs: geomspace(a,b,n) = power(10,linspace(log10(a),log10(b),n)).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np


def build_inputs():
    return {
        "start": np.array(0.125),
        "stop": np.array(128.0),
    }


def forward(start, stop):
    ascending = np.geomspace(start, stop, 4_000_000)
    descending = np.geomspace(stop, start, 4_000_000)
    return {
        'ascending': ascending,
        'descending': descending,
    }
