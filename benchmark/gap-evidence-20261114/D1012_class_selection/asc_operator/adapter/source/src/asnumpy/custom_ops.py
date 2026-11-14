"""Custom AsNumpy wrapper for ``numpy.argmax`` (generated-operator overlay)."""
from __future__ import annotations

import numpy as np

from ._core import argmax as _native_argmax
from .utils import ndarray

__all__ = ["argmax"]


def argmax(a, axis=1):
    """NumPy-compatible argmax on the NPU (float32, rank 2, axis=1, int64 output)."""
    if not isinstance(a, ndarray):
        raise TypeError("asnumpy.argmax expects a single AsNumpy float32 tensor input")
    if np.dtype(a.dtype) != np.dtype(np.float32):
        raise TypeError("asnumpy.argmax supports float32 input only; implicit casting is not allowed")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("asnumpy.argmax axis must be an integer")
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("asnumpy.argmax candidate admits non-empty rank 2 float32 input only")
    normalized = int(axis) + 2 if int(axis) < 0 else int(axis)
    if normalized != 1:
        raise ValueError("asnumpy.argmax admits only axis=1 or axis=-1")
    return ndarray(_native_argmax(a, 1))
