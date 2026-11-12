"""H013_geometric_grid: Positive geometric scale sequence.

Inputs are constructed on the CPU; forward returns named NPU arrays.
Composition reference; NumPy >= 2.0.
For positive inputs: geomspace(a,b,n) = power(10,linspace(log10(a),log10(b),n)).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    return {'start': np.array(0.125), 'stop': np.array(128.0)}


def forward(start, stop):
    if isinstance(start, np.ndarray):
        start = ap.ndarray.from_numpy(start)
    if isinstance(stop, np.ndarray):
        stop = ap.ndarray.from_numpy(stop)
    # ascending: start * (stop / start) ** linspace(0, 1, 21)
    asc_ratio = ap.divide(stop, start)
    asc_lin = ap.linspace(0.0, 1.0, 21)
    asc_pow = ap.power(asc_ratio, asc_lin)
    ascending = ap.multiply(start, asc_pow)
    # descending: stop * (start / stop) ** linspace(0, 1, 21)
    desc_ratio = ap.divide(start, stop)
    desc_lin = ap.linspace(0.0, 1.0, 21)
    desc_pow = ap.power(desc_ratio, desc_lin)
    descending = ap.multiply(stop, desc_pow)
    return {'ascending': ascending, 'descending': descending}