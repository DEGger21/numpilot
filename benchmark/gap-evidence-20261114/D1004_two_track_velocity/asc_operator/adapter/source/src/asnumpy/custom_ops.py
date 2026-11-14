"""Custom AsNumpy wrapper for ``numpy.diff`` (generated-operator overlay)."""
from __future__ import annotations

import numpy as np

from ._core import diff as _native_diff
from .utils import ndarray

__all__ = ["diff"]


def diff(a, n=1, axis=-1):
    """NumPy-compatible first difference on the NPU (float32, rank 2, axis=1)."""
    if not isinstance(a, ndarray):
        raise TypeError("asnumpy.diff expects a single AsNumpy float32 tensor input")
    if np.dtype(a.dtype) != np.dtype(np.float32):
        raise TypeError("asnumpy.diff supports float32 input only; implicit casting is not allowed")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or int(n) != 1:
        raise ValueError("asnumpy.diff candidate admits only n=1")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("asnumpy.diff axis must be an integer")
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] < 2:
        raise ValueError("asnumpy.diff candidate admits rank 2 input with at least two columns")
    normalized = int(axis) + 2 if int(axis) < 0 else int(axis)
    if normalized != 1:
        raise ValueError("asnumpy.diff candidate admits only axis=1 or axis=-1")
    return ndarray(_native_diff(a, normalized))
