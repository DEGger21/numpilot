"""H012_logarithmic_grid: Two logarithmically spaced grids.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Composition reference; NumPy >= 2.0.
logspace(a,b,n,base=c) = power(c,linspace(a,b,n)).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    return {
        "start": np.array(-2.0),
        "stop": np.array(2.0),
    }


def forward(start, stop):
    if isinstance(start, np.ndarray):
        start = ap.ndarray.from_numpy(start)
    if isinstance(stop, np.ndarray):
        stop = ap.ndarray.from_numpy(stop)
    # decades = power(full_like(linspace(0,1,4_000_000), 10.0, dtype=linspace(0,1,4_000_000).dtype),
    #                 add(start, multiply(subtract(stop, start), linspace(0,1,4_000_000))))
    decades_like = ap.linspace(0.0, 1.0, 4_000_000)
    decades_dtype_ref = ap.linspace(0.0, 1.0, 4_000_000)
    decades_base = ap.full_like(decades_like, 10.0, dtype=decades_dtype_ref.dtype)
    decades_span = ap.subtract(stop, start)
    decades_frac = ap.multiply(decades_span, ap.linspace(0.0, 1.0, 4_000_000))
    decades_exponent = ap.add(start, decades_frac)
    decades = ap.power(decades_base, decades_exponent)
    # octaves = power(full_like(linspace(0,1,4_000_000), 2.0, dtype=linspace(0,1,4_000_000).dtype),
    #                 add(start, multiply(subtract(stop, start), linspace(0,1,4_000_000))))
    octaves_like = ap.linspace(0.0, 1.0, 4_000_000)
    octaves_dtype_ref = ap.linspace(0.0, 1.0, 4_000_000)
    octaves_base = ap.full_like(octaves_like, 2.0, dtype=octaves_dtype_ref.dtype)
    octaves_span = ap.subtract(stop, start)
    octaves_frac = ap.multiply(octaves_span, ap.linspace(0.0, 1.0, 4_000_000))
    octaves_exponent = ap.add(start, octaves_frac)
    octaves = ap.power(octaves_base, octaves_exponent)
    return {'decades': decades, 'octaves': octaves}