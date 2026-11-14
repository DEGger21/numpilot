"""G005_median_three_sources: Equal-contract robust row centers.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Three static call sites share exactly one contract:
median|axis=1|keepdims=True|float32|(4,8)
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(705)
    return {
        'a': rng.standard_normal((4, 8), dtype=np.float32),
        'b': rng.standard_normal((4, 8), dtype=np.float32),
        'c': rng.standard_normal((4, 8), dtype=np.float32),
    }


def forward(a, b, c):
    ma = np.median(a, axis=1, keepdims=True)
    mb = np.median(b, axis=1, keepdims=True)
    mc = np.median(c, axis=1, keepdims=True)
    ca = np.subtract(a, ma)
    cb = np.subtract(b, mb)
    cc = np.subtract(c, mc)
    out = np.add(np.square(ca), np.add(np.square(cb), np.square(cc)))
    row_energy = np.mean(out, axis=1)
    return {
        'out': out,
        'row_energy': row_energy,
        'center_a': ma,
    }
