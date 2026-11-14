"""Custom AsNumpy wrapper for ``numpy.repeat`` (generated-operator overlay)."""
from __future__ import annotations

import numpy as np

from ._core import repeat as _native_repeat
from .utils import ndarray

__all__ = ["repeat"]


def repeat(a, repeats, axis=1):
    """NumPy-compatible repeat on the NPU (float32, rank 2, repeats=3, axis=1)."""
    if not isinstance(a, ndarray):
        raise TypeError("asnumpy.repeat expects a single AsNumpy float32 tensor input")
    if np.dtype(a.dtype) != np.dtype(np.float32):
        raise TypeError("asnumpy.repeat supports float32 input only; implicit casting is not allowed")
    if isinstance(repeats, (bool, np.bool_)) or not isinstance(repeats, (int, np.integer)) or int(repeats) != 3:
        raise ValueError("asnumpy.repeat candidate admits only repeats=3")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("asnumpy.repeat axis must be an integer")
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("asnumpy.repeat candidate admits non-empty rank 2 float32 input only")
    normalized = int(axis) + 2 if int(axis) < 0 else int(axis)
    if normalized != 1:
        raise ValueError("asnumpy.repeat admits only axis=1 or axis=-1")
    return ndarray(_native_repeat(a, 3, 1))
