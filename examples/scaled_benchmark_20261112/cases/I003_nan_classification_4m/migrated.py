"""I003_nan_classification: NaN classification with infinities and signed zero.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I1: one unary elementwise bool output.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
Source: ../../../../examples/gap_isnan.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    return {
        "x": np.resize(np.array([0, -0.0, np.nan, np.inf, -np.inf, -1, 1], dtype=np.float32), (4_000_000,)),
    }


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.not_equal(x, x)
    return {'out': __numpilot_v_1}