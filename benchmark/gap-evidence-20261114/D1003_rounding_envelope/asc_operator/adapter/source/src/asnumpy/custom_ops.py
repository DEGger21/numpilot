"""Custom AsNumpy wrapper for ``numpy.nextafter`` (generated-operator overlay)."""
from __future__ import annotations

import numpy as np

from ._core import nextafter as _native_nextafter
from .utils import ndarray

__all__ = ["nextafter"]


def nextafter(a, b):
    """NumPy-compatible nextafter on the NPU (float32, rank 2)."""
    if not isinstance(a, ndarray):
        raise TypeError("asnumpy.nextafter expects AsNumpy float32 tensor inputs")
    if not isinstance(b, ndarray):
        raise TypeError("asnumpy.nextafter expects AsNumpy float32 tensor inputs")
    if np.dtype(a.dtype) != np.dtype(np.float32) or np.dtype(b.dtype) != np.dtype(np.float32):
        raise TypeError("asnumpy.nextafter supports float32 inputs only; implicit casting is not allowed")
    if tuple(int(dim) for dim in a.shape) != tuple(int(dim) for dim in b.shape):
        raise ValueError("asnumpy.nextafter requires identical input shapes")
    shape = tuple(int(dim) for dim in a.shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("asnumpy.nextafter candidate admits non-empty rank 2 float32 input only")
    return ndarray(_native_nextafter(a, b))
