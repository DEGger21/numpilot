"""B010_empty_sum: Additive identity over an empty axis.

Inputs are constructed on the CPU; forward returns named NPU arrays.
Trap: empty array with a nonempty output.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    return {'x': np.empty((3, 0, 4), dtype=np.float32)}


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    out = ap.sum(x, axis=1)
    product = ap.prod(x, axis=1)
    return {'out': out, 'product': product}