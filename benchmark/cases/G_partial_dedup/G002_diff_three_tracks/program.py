"""G002_diff_three_tracks: Three equal-contract temporal derivatives.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Three static call sites share exactly one contract:
diff|n=1|axis=1|float32|(4,8)
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(702)
    return {
        'a': rng.standard_normal((4, 8), dtype=np.float32),
        'b': rng.standard_normal((4, 8), dtype=np.float32),
        'c': rng.standard_normal((4, 8), dtype=np.float32),
    }


def forward(a, b, c):
    da = np.diff(a, n=1, axis=1)
    db = np.diff(b, n=1, axis=1)
    dc = np.diff(c, n=1, axis=1)
    velocity = np.add(da, db)
    acceleration_proxy = np.subtract(dc, velocity)
    power = np.square(acceleration_proxy)
    out = np.sqrt(np.add(power, 1e-5))
    summary = np.sum(out, axis=1)
    return {
        'out': out,
        'velocity': velocity,
        'summary': summary,
    }
