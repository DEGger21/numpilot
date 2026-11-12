"""I001_cube_root: Signed cube root including signed zero.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I1: one unary elementwise output.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
Source: ../../../../examples/gap_cbrt.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    x = np.linspace(-27, 27, 4_000_000, dtype=np.float32)
    x[2_000_000] = -0.0
    return {
        "x": x,
    }


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    abs_x = ap.absolute(x)
    exp_x = ap.full_like(x, 0.3333333333333333, dtype=x.dtype)
    pow_x = ap.power(abs_x, exp_x)
    __numpilot_v_1 = ap.copysign(pow_x, x)
    return {'out': __numpilot_v_1}