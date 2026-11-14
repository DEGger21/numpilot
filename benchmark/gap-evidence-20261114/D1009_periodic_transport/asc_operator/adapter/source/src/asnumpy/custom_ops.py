"""Custom AsNumpy wrapper for ``numpy.roll`` (generated-operator overlay)."""
from __future__ import annotations

import numpy as np

from ._core import roll as _native_roll
from .utils import ndarray

__all__ = ["roll"]


def roll(a, shift, axis):
    """NumPy-compatible roll on the NPU (float32, rank 2, shift=1, axis=1)."""
    if not isinstance(a, ndarray):
        raise TypeError("asnumpy.roll expects a single AsNumpy float32 tensor input")
    if np.dtype(a.dtype) != np.dtype(np.float32):
        raise TypeError("asnumpy.roll supports float32 input only; implicit casting is not allowed")
    if isinstance(shift, (bool, np.bool_)) or not isinstance(shift, (int, np.integer)) or int(shift) != 1:
        raise ValueError("asnumpy.roll candidate admits only shift=1")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("asnumpy.roll axis must be an integer")
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("asnumpy.roll candidate admits non-empty rank 2 float32 input only")
    normalized = int(axis) + 2 if int(axis) < 0 else int(axis)
    if normalized != 1:
        raise ValueError("asnumpy.roll admits only axis=1 or axis=-1")
    return ndarray(_native_roll(a, 1, 1))
