"""F007_sort_kind: Stable sort keyword is part of the contract.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Witness NumPy output: [[1,1,2,3],[-1,0,0,2]].
Pinned 1929a3ba wrapper sort rejects kind: TypeError before its body.
This signature mismatch is checkable without a CANN runtime.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(707)
    return {
        "x": rng.integers(-100, 100, (2048, 2048)).astype(np.float32),
    }


def forward(x):
    out = np.sort(x, axis=1, kind="stable")
    return {
        'out': out,
        'prefix': np.cumsum(out, axis=1),
    }
