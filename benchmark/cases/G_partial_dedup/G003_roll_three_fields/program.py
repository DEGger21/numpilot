"""G003_roll_three_fields: Identical periodic shifts in three fields.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Three static call sites share exactly one contract:
roll|shift=1|axis=1|float32|(4,8)
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(703)
    return {
        'a': rng.standard_normal((4, 8), dtype=np.float32),
        'b': rng.standard_normal((4, 8), dtype=np.float32),
        'c': rng.standard_normal((4, 8), dtype=np.float32),
    }


def forward(a, b, c):
    ra = np.roll(a, shift=1, axis=1)
    rb = np.roll(b, shift=1, axis=1)
    rc = np.roll(c, shift=1, axis=1)
    flux_a = np.subtract(a, ra)
    flux_b = np.subtract(b, rb)
    flux_c = np.subtract(c, rc)
    flux = np.add(np.add(flux_a, flux_b), flux_c)
    out = np.multiply(flux, 0.25)
    return {
        'out': out,
        'energy': np.sum(np.square(out), axis=1),
    }
