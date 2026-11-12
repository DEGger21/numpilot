"""G004_cbrt_three_channels: Signed compression of three channels.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Three static call sites share exactly one contract:
cbrt|float32|(4,8)
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(704)
    return {
        "a": rng.standard_normal((16384, 256), dtype=np.float32),
        "b": rng.standard_normal((16384, 256), dtype=np.float32),
        "c": rng.standard_normal((16384, 256), dtype=np.float32),
    }


def forward(a, b, c):
    ra = np.cbrt(a)
    rb = np.cbrt(b)
    rc = np.cbrt(c)
    mixture = np.add(np.multiply(ra, 0.5), np.multiply(rb, 0.3))
    mixture = np.add(mixture, np.multiply(rc, 0.2))
    out = np.tanh(mixture)
    agreement = np.multiply(np.multiply(ra, rb), rc)
    return {
        'out': out,
        'agreement': agreement,
        'compressed_a': ra,
    }
